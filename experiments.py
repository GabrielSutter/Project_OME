import subprocess
import itertools
import multiprocessing as mp
import time
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm   # <-- progress bar

EXEC = "./GA"
TEST_FILE = "jobshop1.txt"

population_sizes = [10, 20, 30]
reruns = [1, 3, 5]
iterations = [50, 100, 150]
mutation_probs = [0.01, 0.05, 0.1]
mutation_types = [1, 2, 3]
crossover_types = [1, 2]

TIMEOUT = 120

with open(TEST_FILE, "r") as f:
    TEST_DATA = f.read()


def run_instance(params):
    pop, rerun, iters, mut_prob, mut_type, cross_type = params

    input_data = f"{pop} {rerun} {iters} {mut_prob} {mut_type} {cross_type}\n"
    input_data += TEST_DATA

    try:
        start = time.time()

        proc = subprocess.run(
            EXEC,
            input=input_data.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT
        )

        elapsed = time.time() - start

        output_lines = proc.stdout.decode().strip().splitlines()
        scores = [float(x) for x in output_lines if x.strip()]

        return {
            "params": params,
            "mean": statistics.mean(scores),
            "best": min(scores),
            "worst": max(scores),
            "std": statistics.pstdev(scores) if len(scores) > 1 else 0,
            "time": elapsed,
        }

    except Exception as e:
        return {
            "params": params,
            "mean": float("inf"),
            "best": float("inf"),
            "worst": float("inf"),
            "std": float("inf"),
            "time": float("inf"),
            "error": str(e),
        }


def main():
    all_params = list(itertools.product(
        population_sizes,
        reruns,
        iterations,
        mutation_probs,
        mutation_types,
        crossover_types
    ))

    total = len(all_params)
    print(f"Running {total} experiments...\n")

    results = []

    with mp.Pool(mp.cpu_count()) as pool:
        for result in tqdm(
            pool.imap_unordered(run_instance, all_params),
            total=total,
            desc="Experiments",
            smoothing=0.1
        ):
            results.append(result)

    # --- Sort results ---
    results.sort(key=lambda x: x["mean"])

    print("\nTop 10 configurations:")
    for r in results[:10]:
        print(r)

    # --- Plot mutation vs crossover ---
    grouped = {}
    for r in results:
        mut = r["params"][4]
        cross = r["params"][5]
        grouped.setdefault((mut, cross), []).append(r["mean"])

    labels = []
    values = []
    for key, vals in grouped.items():
        labels.append(f"M{key[0]}-C{key[1]}")
        values.append(sum(vals) / len(vals))

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.ylabel("Average Makespan")
    plt.title("Mutation vs Crossover Performance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("mutation_vs_crossover.png", dpi=150)
    plt.close()

    # --- Save results ---
    with open("results_full.txt", "w") as f:
        for r in results:
            f.write(str(r) + "\n")


if __name__ == "__main__":
    main()