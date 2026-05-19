import csv
import itertools
import multiprocessing as mp
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

EXEC = "./GA"
TEST_FILE = "jobshop1.txt"

POPULATION_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
RERUNS = [1, 3, 5]
ITERATIONS = [64, 256, 1024]
MUTATION_PROBS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
MUTATION_TYPES = [1, 2, 3]
CROSSOVER_TYPES = [1, 2]

REPEATS = 5

TIMEOUT = 1000

# IMPORTANT:
# Do NOT use all cores initially.
# Your GA is probably already multithreaded/cache-heavy.
WORKERS = max(1, mp.cpu_count() - 2)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path("benchmark_data")
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_RUNS = OUTPUT_DIR / "raw_runs.csv"
FAILURES = OUTPUT_DIR / "failures.csv"

# ============================================================
# LOAD TEST INSTANCE
# ============================================================

with open(TEST_FILE, "r") as f:
    TEST_DATA = f.read()

# ============================================================
# INITIALIZE CSV
# ============================================================

if not RAW_RUNS.exists():
    with open(RAW_RUNS, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "population",
            "rerun",
            "iterations",
            "mutation_prob",
            "mutation_type",
            "crossover_type",
            "seed",
            "runtime_sec",
            "final_score",
        ])

if not FAILURES.exists():
    with open(FAILURES, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "population",
            "rerun",
            "iterations",
            "mutation_prob",
            "mutation_type",
            "crossover_type",
            "seed",
            "error",
        ])

# ============================================================
# JOB GENERATION
# ============================================================

def generate_jobs():

    for params in itertools.product(
        POPULATION_SIZES,
        RERUNS,
        ITERATIONS,
        MUTATION_PROBS,
        MUTATION_TYPES,
        CROSSOVER_TYPES
    ):

        for _ in range(REPEATS):

            seed = random.randint(0, 2**31 - 1)

            yield (*params, seed)

# ============================================================
# RUN SINGLE EXPERIMENT
# ============================================================

def run_job(job):

    (
        pop,
        rerun,
        iterations,
        mut_prob,
        mut_type,
        cross_type,
        seed
    ) = job

    # ========================================================
    # IMPORTANT:
    # PRESERVE ORIGINAL INPUT FORMAT
    # ========================================================

    input_data = (
        f"{pop} "
        f"{rerun} "
        f"{iterations} "
        f"{mut_prob} "
        f"{mut_type} "
        f"{cross_type}\n"
        f"{TEST_DATA}"
    )

    try:

        start = time.perf_counter()

        proc = subprocess.run(
            EXEC,
            input=input_data.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT
        )

        runtime = time.perf_counter() - start

        if proc.returncode != 0:

            return {
                "success": False,
                "job": job,
                "error": proc.stderr.decode(
                    errors="ignore"
                )
            }

        lines = proc.stdout.decode(
            errors="ignore"
        ).splitlines()

        scores = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            try:
                scores.append(float(line))
            except:
                pass

        if not scores:

            return {
                "success": False,
                "job": job,
                "error": "No valid scores"
            }

        final_score = statistics.mean(scores)

        return {
            "success": True,
            "row": [
                pop,
                rerun,
                iterations,
                mut_prob,
                mut_type,
                cross_type,
                seed,
                runtime,
                final_score,
            ]
        }

    except subprocess.TimeoutExpired:

        return {
            "success": False,
            "job": job,
            "error": "Timeout"
        }

    except Exception as e:

        return {
            "success": False,
            "job": job,
            "error": str(e)
        }

# ============================================================
# MAIN
# ============================================================

def main():

    jobs = list(generate_jobs())

    print(f"Total jobs: {len(jobs)}")
    print(f"Workers: {WORKERS}")

    with mp.Pool(WORKERS) as pool:

        iterator = pool.imap_unordered(
            run_job,
            jobs,
            chunksize=4
        )

        for result in tqdm(iterator, total=len(jobs)):

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if result["success"]:

                with open(RAW_RUNS, "a", newline="") as f:

                    writer = csv.writer(f)

                    writer.writerow(result["row"])

            # ------------------------------------------------
            # FAILURE
            # ------------------------------------------------

            else:

                (
                    pop,
                    rerun,
                    iterations,
                    mut_prob,
                    mut_type,
                    cross_type,
                    seed
                ) = result["job"]

                with open(FAILURES, "a", newline="") as f:

                    writer = csv.writer(f)

                    writer.writerow([
                        pop,
                        rerun,
                        iterations,
                        mut_prob,
                        mut_type,
                        cross_type,
                        seed,
                        result["error"]
                    ])

if __name__ == "__main__":
    main()
