import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT = "raw_runs.csv"

OUT = Path("analysis")
OUT.mkdir(exist_ok=True)

MUTATION_NAMES = {
    1: "Swap",
    2: "Insert",
    3: "Invert"
}

CROSSOVER_NAMES = {
    1: "Interleaving",
    2: "Segment"
}


def savefig(name):
    plt.tight_layout()
    plt.savefig(
        OUT / name,
        dpi=200
    )
    plt.close()


def pareto_frontier(df):

    df = df.sort_values(
        ["runtime_sec", "final_score"]
    )

    frontier = []

    best_score = float("inf")

    for _, row in df.iterrows():

        score = row["final_score"]

        if score < best_score:

            frontier.append(row)

            best_score = score

    return pd.DataFrame(frontier)


def make_plots(df):

    plt.figure(figsize=(8,5))

    grouped = df.groupby(
        "population"
    )["final_score"].mean()

    grouped.plot(marker="o")

    plt.xlabel("Population")
    plt.ylabel("Average Makespan")

    plt.title(
        "Population Size vs Solution Quality"
    )

    savefig(
        "population_vs_quality.png"
    )



    plt.figure(figsize=(8,5))

    grouped = df.groupby(
        "iterations"
    )["final_score"].mean()

    grouped.plot(marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("Average Makespan")

    plt.title(
        "Iterations vs Solution Quality"
    )

    savefig(
        "iterations_vs_quality.png"
    )



    plt.figure(figsize=(8,5))

    grouped = df.groupby(
        "mutation_prob"
    )["final_score"].mean()

    grouped.plot(marker="o")

    plt.xlabel("Mutation Probability")
    plt.ylabel("Average Makespan")

    plt.title(
        "Mutation Probability Sweet Spot"
    )

    savefig(
        "mutation_prob_vs_quality.png"
    )



    plt.figure(figsize=(8,5))

    grouped = df.groupby(
        "mutation_name"
    )["final_score"].mean()

    grouped.plot.bar()

    plt.ylabel("Average Makespan")

    plt.title(
        "Mutation Operator Performance"
    )

    savefig(
        "mutation_types.png"
    )



    plt.figure(figsize=(8,5))

    grouped = df.groupby(
        "crossover_name"
    )["final_score"].mean()

    grouped.plot.bar()

    plt.ylabel("Average Makespan")

    plt.title(
        "Crossover Performance"
    )

    savefig(
        "crossover_types.png"
    )



    plt.figure(figsize=(8,6))

    plt.scatter(
        df["runtime_sec"],
        df["final_score"],
        alpha=0.3,
        s=10
    )

    plt.xlabel("Runtime (s)")
    plt.ylabel("Final Makespan")

    plt.title(
        "Runtime vs Quality"
    )

    savefig(
        "runtime_vs_quality.png"
    )



    heat = (
        df.groupby(
            ["mutation_name","crossover_name"]
        )
        ["final_score"]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(7,5))

    plt.imshow(
        heat,
        aspect="auto"
    )

    plt.xticks(
        range(len(heat.columns)),
        heat.columns
    )

    plt.yticks(
        range(len(heat.index)),
        heat.index
    )

    plt.colorbar(
        label="Average Makespan"
    )

    plt.title(
        "Mutation/Crossover Heatmap"
    )

    savefig(
        "operator_heatmap.png"
    )



    frontier = pareto_frontier(df)

    plt.figure(figsize=(8,6))

    plt.scatter(
        df["runtime_sec"],
        df["final_score"],
        alpha=0.15,
        s=10
    )

    plt.plot(
        frontier["runtime_sec"],
        frontier["final_score"],
        linewidth=2
    )

    plt.xlabel("Runtime")

    plt.ylabel("Makespan")

    plt.title(
        "Pareto Frontier"
    )

    savefig(
        "pareto_frontier.png"
    )



def build_summary(df):

    summary = (

        df.groupby([
            "population",
            "iterations",
            "mutation_name",
            "crossover_name"
        ])

        .agg(

            mean_score=(
                "final_score",
                "mean"
            ),

            std_score=(
                "final_score",
                "std"
            ),

            best_score=(
                "final_score",
                "min"
            ),

            mean_runtime=(
                "runtime_sec",
                "mean"
            )

        )

        .reset_index()

    )

    score_norm = (
        summary["mean_score"]
        /
        summary["mean_score"].min()
    )

    runtime_norm = (
        summary["mean_runtime"]
        /
        summary["mean_runtime"].min()
    )

    stability = (
        summary["std_score"]
        /
        summary["std_score"].max()
    )

    summary["combined_rank"] = (

        0.6*score_norm +

        0.25*runtime_norm +

        0.15*stability

    )

    summary = summary.sort_values(
        "combined_rank"
    )

    summary.to_csv(
        OUT / "summary.csv",
        index=False
    )

    summary.head(50).to_csv(
        OUT / "top50.csv",
        index=False
    )

    return summary


def main():

    df = pd.read_csv(INPUT)

    df["mutation_name"] = (

        df["mutation_type"]
        .map(MUTATION_NAMES)

    )

    df["crossover_name"] = (

        df["crossover_type"]
        .map(CROSSOVER_NAMES)

    )

    make_plots(df)

    summary = build_summary(df)

    print()

    print(
        "Top configurations:"
    )

    print(
        summary.head(20)
    )

    print()

    print(
        "Analysis saved to:",
        OUT
    )


if __name__ == "__main__":
    main()
