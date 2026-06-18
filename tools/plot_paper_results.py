import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SACRED_DIR = RESULTS_DIR / "sacred"
TRAJ_DIR = RESULTS_DIR / "microgrid_trajs" / "facmac_smac"
OUTPUT_DIR = RESULTS_DIR / "paper_figures"


RUN_LABELS = {
    24: "Reward shaping v2",
    25: "Reward rollback 160k",
    27: "Stable train 160k",
    28: "Stable train 400k",
    29: "Current rerun 160k",
}


CURVE_METRICS = [
    ("test_grid_purchase_cost_mean", "Test Grid Purchase Cost", False),
    ("test_return_mean", "Test Return", True),
    ("test_soc_violation_mean", "Test SOC Violation", False),
    ("critic_loss", "Critic Loss", False),
]


FINAL_BAR_METRICS = [
    ("test_grid_purchase_cost_mean", "Final Test Grid Purchase Cost", False),
    ("test_return_mean", "Final Test Return", True),
    ("test_soc_violation_mean", "Final Test SOC Violation", False),
]


BEST_RUN_ID = 29
TRAIN_TEST_METRICS = [
    ("return_mean", "test_return_mean", "Return"),
    ("grid_purchase_cost_mean", "test_grid_purchase_cost_mean", "Grid Purchase Cost"),
    ("soc_violation_mean", "test_soc_violation_mean", "SOC Violation"),
]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_id: int) -> Optional[Dict]:
    run_dir = SACRED_DIR / str(run_id)
    info_path = run_dir / "info.json"
    config_path = run_dir / "config.json"
    run_path = run_dir / "run.json"
    if not info_path.exists() or not config_path.exists() or not run_path.exists():
        return None
    return {
        "run_id": run_id,
        "label": RUN_LABELS.get(run_id, f"Run {run_id}"),
        "info": load_json(info_path),
        "config": load_json(config_path),
        "run": load_json(run_path),
    }


def get_series(info: Dict, metric: str) -> Optional[Tuple[List[float], List[float]]]:
    t_key = f"{metric}_T"
    values = info.get(metric)
    steps = info.get(t_key)
    if not values or not steps:
        return None
    numeric_steps: List[float] = []
    numeric_values: List[float] = []
    n = min(len(values), len(steps))
    for step, value in zip(steps[:n], values[:n]):
        step_num = coerce_numeric(step)
        value_num = coerce_numeric(value)
        if step_num is None or value_num is None:
            continue
        numeric_steps.append(step_num)
        numeric_values.append(value_num)
    if not numeric_steps or not numeric_values:
        return None
    return numeric_steps, numeric_values


def coerce_numeric(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and "value" in value and isinstance(value["value"], (int, float)):
        return float(value["value"])
    return None


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
        }
    )


def plot_metric_curves(runs: List[Dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, title, higher_better) in zip(axes, CURVE_METRICS):
        for run_data in runs:
            series = get_series(run_data["info"], metric)
            if series is None:
                continue
            steps, values = series
            ax.plot(steps, values, label=run_data["label"])
        ax.set_title(title)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(title)
        if metric == "critic_loss":
            ax.set_yscale("log")
        note = "Higher is better" if higher_better else "Lower is better"
        ax.text(0.02, 0.95, note, transform=ax.transAxes, va="top", fontsize=8)
        ax.legend()

    fig.suptitle("Training Curves Across Key Runs", fontsize=14)
    fig.savefig(OUTPUT_DIR / "training_curves_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_metric_bars(runs: List[Dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    labels = [run_data["label"] for run_data in runs]

    for ax, (metric, title, higher_better) in zip(axes, FINAL_BAR_METRICS):
        finals = []
        for run_data in runs:
            series = get_series(run_data["info"], metric)
            finals.append(series[1][-1] if series else float("nan"))
        bars = ax.bar(labels, finals, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(labels)])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=15)
        note = "Higher is better" if higher_better else "Lower is better"
        ax.text(0.02, 0.95, note, transform=ax.transAxes, va="top", fontsize=8)
        for bar, value in zip(bars, finals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    fig.suptitle("Final Metric Comparison", fontsize=14)
    fig.savefig(OUTPUT_DIR / "final_metrics_bar_chart.png", bbox_inches="tight")
    plt.close(fig)


def plot_best_run_train_test(run_data: Dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for ax, (train_metric, test_metric, title) in zip(axes, TRAIN_TEST_METRICS):
        train_series = get_series(run_data["info"], train_metric)
        test_series = get_series(run_data["info"], test_metric)
        if train_series:
            ax.plot(train_series[0], train_series[1], label=f"Train {title}")
        if test_series:
            ax.plot(test_series[0], test_series[1], label=f"Test {title}")
        ax.set_title(title)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(title)
        ax.legend()

    fig.suptitle(f"Train vs Test Curves for {run_data['label']}", fontsize=14)
    fig.savefig(OUTPUT_DIR / "best_run_train_vs_test.png", bbox_inches="tight")
    plt.close(fig)


def find_latest_trajectory_csv() -> Optional[Path]:
    if not TRAJ_DIR.exists():
        return None
    csv_files = sorted(TRAJ_DIR.glob("*/*.csv"), key=lambda p: p.stat().st_mtime)
    return csv_files[-1] if csv_files else None


def plot_trajectory(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    episode_id = int(df["episode"].min())
    episode_df = df[df["episode"] == episode_id].copy()
    step_df = (
        episode_df.groupby("col_idx", as_index=False)
        .agg(
            price=("price", "mean"),
            load=("load", "mean"),
            pv=("pv", "mean"),
            pg_total=("pg_total", "mean"),
            buy_cost=("buy_cost", "mean"),
        )
        .sort_values("col_idx")
    )

    soc_df = (
        episode_df.pivot_table(index="col_idx", columns="agent", values="soc", aggfunc="mean")
        .sort_index()
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    axes[0].plot(step_df["col_idx"], step_df["price"], label="Price")
    axes[0].plot(step_df["col_idx"], step_df["load"], label="Load")
    axes[0].plot(step_df["col_idx"], step_df["pv"], label="PV")
    axes[0].set_title("Price, Load and PV")
    axes[0].set_xlabel("Time Step")
    axes[0].legend()

    for agent in soc_df.columns:
        axes[1].plot(soc_df.index, soc_df[agent], label=agent)
    axes[1].axhline(0.2, color="red", linestyle="--", linewidth=1, label="SOC Lower Bound")
    axes[1].axhline(0.8, color="green", linestyle="--", linewidth=1, label="SOC Upper Bound")
    axes[1].set_title("Battery SOC Trajectory")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("SOC")
    axes[1].legend(ncol=3)

    axes[2].plot(step_df["col_idx"], step_df["pg_total"], label="Grid Power")
    axes[2].plot(step_df["col_idx"], step_df["buy_cost"], label="Buy Cost")
    axes[2].set_title("Grid Interaction")
    axes[2].set_xlabel("Time Step")
    axes[2].legend()

    fig.suptitle(f"Example Test Episode Trajectory ({csv_path.parent.name})", fontsize=14)
    fig.savefig(OUTPUT_DIR / "example_test_episode_trajectory.png", bbox_inches="tight")
    plt.close(fig)


def save_run_summary(runs: List[Dict]) -> None:
    rows = []
    for run_data in runs:
        row = {"run_id": run_data["run_id"], "label": run_data["label"]}
        for metric, _, _ in FINAL_BAR_METRICS:
            series = get_series(run_data["info"], metric)
            row[metric] = series[1][-1] if series else None
        critic_series = get_series(run_data["info"], "critic_loss")
        row["critic_loss"] = critic_series[1][-1] if critic_series else None
        row["t_max"] = run_data["config"].get("t_max")
        row["action_bins"] = run_data["config"].get("env_args", {}).get("action_bins")
        row["lr"] = run_data["config"].get("lr")
        row["critic_lr"] = run_data["config"].get("critic_lr")
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "run_summary.csv", index=False)


def main() -> None:
    ensure_output_dir()
    set_style()

    runs = [load_run(run_id) for run_id in RUN_LABELS]
    runs = [run_data for run_data in runs if run_data is not None]
    if not runs:
        raise RuntimeError("No valid Sacred runs found.")

    plot_metric_curves(runs)
    plot_final_metric_bars(runs)

    best_run = next((run_data for run_data in runs if run_data["run_id"] == BEST_RUN_ID), runs[-1])
    plot_best_run_train_test(best_run)

    csv_path = find_latest_trajectory_csv()
    if csv_path:
        plot_trajectory(csv_path)

    save_run_summary(runs)

    print(f"Saved paper figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
