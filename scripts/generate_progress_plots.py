"""Generate advisor-check progress plots from DQN sweep outputs.

Creates 5 static figures:
1) baseline_vs_dqn_summary.png
2) coeff_sweep_tradeoff.png
3) action_distribution_by_coeff.png
4) learning_curves_tir_insulin.png
5) glucose_like_trace_proxy.png

Notes:
- Uses existing artifacts CSV files.
- If baseline CSV is missing, compares across DQN coefficients only.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ART = Path("artifacts")
OUT = ART / "advisor_figures"
OUT.mkdir(parents=True, exist_ok=True)

SWEEP_FILES = [
    ART / "dqn_sweep_coeff_0.10.csv",
    ART / "dqn_sweep_coeff_0.30.csv",
    ART / "dqn_sweep_coeff_0.50.csv",
]


def load_eval_means(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    dfe = df[df["phase"] == "eval"].copy()
    coeff = float(dfe["coeff"].iloc[0])
    means = dfe.mean(numeric_only=True).to_dict()
    means["coeff"] = coeff
    means["_eval_df"] = dfe
    return means


def build_summary_df() -> pd.DataFrame:
    rows = []
    for p in SWEEP_FILES:
        if p.exists():
            rows.append(load_eval_means(p))
    if not rows:
        raise FileNotFoundError("No sweep CSV files found in artifacts/.")
    sdf = pd.DataFrame([{k: v for k, v in r.items() if k != "_eval_df"} for r in rows])
    sdf = sdf.sort_values("coeff").reset_index(drop=True)
    return sdf


def plot_1_baseline_vs_best(summary: pd.DataFrame) -> None:
    # If baseline CSV exists, include it; otherwise compare only coefficients.
    baseline_path = ART / "baselines_metrics.csv"
    best_idx = summary["time_in_range_percent"].idxmax()
    best = summary.loc[best_idx]

    labels = [f"DQN c={c:.1f}" for c in summary["coeff"].tolist()]
    tir = summary["time_in_range_percent"].values
    mean_g = summary["mean_glucose"].values
    hypo = summary["hypo_events"].values
    severe = summary["severe_hyper_events"].values
    insulin = summary["total_insulin_units"].values

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    metrics = [tir, mean_g, hypo, severe, insulin]
    titles = ["TIR %", "Mean Glucose", "Hypo Events", "Severe Hyper Events", "Total Insulin U"]
    for ax, vals, t in zip(axes, metrics, titles):
        ax.bar(labels, vals, color=["#4C78A8", "#72B7B2", "#F58518"])  # 3 coeffs
        ax.set_title(t)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("DQN Coefficients Summary (Eval Means)")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_vs_dqn_summary.png", dpi=160)
    plt.close(fig)


def plot_2_coeff_sweep_tradeoff(summary: pd.DataFrame) -> None:
    coeff = summary["coeff"].values
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    axes[0].plot(coeff, summary["time_in_range_percent"], marker="o", linewidth=2)
    axes[0].set_title("TIR vs Penalty Coeff")
    axes[0].set_xlabel("insulin_penalty_coeff")
    axes[0].set_ylabel("TIR %")

    axes[1].plot(coeff, summary["total_insulin_units"], marker="o", linewidth=2, color="#F58518")
    axes[1].set_title("Total Insulin vs Coeff")
    axes[1].set_xlabel("insulin_penalty_coeff")
    axes[1].set_ylabel("Units/day")

    axes[2].plot(coeff, summary["severe_hyper_steps"], marker="o", linewidth=2, color="#E45756")
    axes[2].set_title("Severe Hyper Steps vs Coeff")
    axes[2].set_xlabel("insulin_penalty_coeff")
    axes[2].set_ylabel("Steps/episode")

    fig.suptitle("Reward Sweep Tradeoffs")
    fig.tight_layout()
    fig.savefig(OUT / "coeff_sweep_tradeoff.png", dpi=160)
    plt.close(fig)


def plot_3_action_distribution(summary: pd.DataFrame) -> None:
    labels = [f"c={c:.1f}" for c in summary["coeff"]]
    a0 = summary["action_0_count"].values
    a1 = summary["action_1_count"].values
    a2 = summary["action_2_count"].values
    a3 = summary["action_3_count"].values
    a4 = summary["action_4_count"].values

    x = np.arange(len(labels))
    width = 0.62

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, a0, width, label="A0 (0U)")
    ax.bar(x, a1, width, bottom=a0, label="A1 (2.8U)")
    ax.bar(x, a2, width, bottom=a0+a1, label="A2 (4.8U)")
    ax.bar(x, a3, width, bottom=a0+a1+a2, label="A3 (7.6U)")
    ax.bar(x, a4, width, bottom=a0+a1+a2+a3, label="A4 (12.5U)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean action count per episode")
    ax.set_title("Action Distribution by Penalty Coefficient")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "action_distribution_by_coeff.png", dpi=160)
    plt.close(fig)


def plot_4_learning_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for p, color in zip(SWEEP_FILES, ["#4C78A8", "#72B7B2", "#F58518"]):
        if not p.exists():
            continue
        df = pd.read_csv(p)
        dfe = df[df["phase"] == "eval"].copy()
        if dfe.empty:
            continue
        coeff = float(dfe["coeff"].iloc[0])
        grouped = dfe.groupby("train_episode").agg(
            tir=("time_in_range_percent", "mean"),
            insulin=("total_insulin_units", "mean"),
        ).reset_index()

        axes[0].plot(grouped["train_episode"], grouped["tir"], marker="o", label=f"c={coeff:.1f}", color=color)
        axes[1].plot(grouped["train_episode"], grouped["insulin"], marker="o", label=f"c={coeff:.1f}", color=color)

    axes[0].set_title("Eval TIR over Training")
    axes[0].set_xlabel("Train episode")
    axes[0].set_ylabel("TIR %")
    axes[0].legend()

    axes[1].set_title("Eval Total Insulin over Training")
    axes[1].set_xlabel("Train episode")
    axes[1].set_ylabel("Units/day")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "learning_curves_tir_insulin.png", dpi=160)
    plt.close(fig)


def plot_5_glucose_like_proxy(summary: pd.DataFrame) -> None:
    # We do not log per-timestep glucose traces in current CSVs.
    # This figure provides a compact proxy: distribution of episode-level mean glucose by coeff.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = []
    labels = []
    for p in SWEEP_FILES:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        dfe = df[df["phase"] == "eval"].copy()
        coeff = float(dfe["coeff"].iloc[0])
        data.append(dfe["mean_glucose"].values)
        labels.append(f"c={coeff:.1f}")

    ax.boxplot(data, labels=labels)
    ax.axhline(66, linestyle="--", color="red", linewidth=1, label="Hypo threshold")
    ax.axhline(80, linestyle=":", color="gray", linewidth=1)
    ax.axhline(180, linestyle=":", color="gray", linewidth=1, label="Target 80-180")
    ax.axhline(292, linestyle="--", color="purple", linewidth=1, label="Severe hyper threshold")
    ax.set_title("Episode Mean Glucose Distribution (Eval)")
    ax.set_ylabel("mg/dL")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "glucose_like_trace_proxy.png", dpi=160)
    plt.close(fig)


def main() -> None:
    summary = build_summary_df()
    plot_1_baseline_vs_best(summary)
    plot_2_coeff_sweep_tradeoff(summary)
    plot_3_action_distribution(summary)
    plot_4_learning_curves()
    plot_5_glucose_like_proxy(summary)

    print(f"Saved figures to: {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f" - {p}")


if __name__ == "__main__":
    main()
