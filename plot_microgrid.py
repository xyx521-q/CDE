import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

sys.dont_write_bytecode = True
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pick_episode(df: pd.DataFrame, episode: int | None) -> int:
    if episode is not None:
        return int(episode)
    return int(df["episode"].min())


def _series_by_agent(df_ep: pd.DataFrame, col: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    out = {}
    df_ep = df_ep.sort_values(["agent", "col_idx"])
    x = sorted(df_ep["col_idx"].unique())
    for agent, g in df_ep.groupby("agent"):
        s = g.groupby("col_idx")[col].first().reindex(x)
        out[str(agent)] = s.to_numpy()
    return np.array(x, dtype=int), out


def _series_shared(df_ep: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    g = df_ep.sort_values(["col_idx", "agent"])
    s = g.groupby("col_idx")[col].first().sort_index()
    return s.index.to_numpy(dtype=int), s.to_numpy()


def plot_microgrid(csv_path: Path, out_dir: Path, episode: int | None):
    df = pd.read_csv(csv_path)
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce").astype("Int64")
    df["col_idx"] = pd.to_numeric(df["col_idx"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["episode", "col_idx"]).copy()
    df["episode"] = df["episode"].astype(int)
    df["col_idx"] = df["col_idx"].astype(int)

    ep = _pick_episode(df, episode)
    df_ep = df[df["episode"] == ep].copy()
    if df_ep.empty:
        raise ValueError(f"episode {ep} not found in {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    x_soc, soc_by_agent = _series_by_agent(df_ep, "soc")
    _, soc_init_by_agent = _series_by_agent(df_ep, "soc_init")

    fig = plt.figure(figsize=(10, 4))
    for agent, y in soc_by_agent.items():
        plt.plot(x_soc, y, label=f"{agent}")
    plt.ylim(0, 1)
    plt.xlabel("Hour index")
    plt.ylabel("SoC")
    plt.title(f"SoC Trajectory (episode={ep})")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=min(4, max(1, len(soc_by_agent))))
    fig.tight_layout()
    fig.savefig(out_dir / f"episode_{ep:06d}_soc.png", dpi=200)
    plt.close(fig)

    x_pg, pg = _series_shared(df_ep, "pg_total")
    _, price = _series_shared(df_ep, "price")
    fig = plt.figure(figsize=(10, 4))
    ax1 = plt.gca()
    ax1.plot(x_pg, pg, color="tab:blue", label="pg_total")
    ax1.set_xlabel("Hour index")
    ax1.set_ylabel("pg_total", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x_pg, price, color="tab:orange", label="price")
    ax2.set_ylabel("price", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title(f"Grid Exchange & Price (episode={ep})")
    fig.tight_layout()
    fig.savefig(out_dir / f"episode_{ep:06d}_pg_price.png", dpi=200)
    plt.close(fig)

    x_cost, buy_cost = _series_shared(df_ep, "buy_cost")
    x_reward, reward_sum = (
        df_ep.groupby("col_idx")["reward"].sum().sort_index().index.to_numpy(dtype=int),
        df_ep.groupby("col_idx")["reward"].sum().sort_index().to_numpy(),
    )
    fig = plt.figure(figsize=(10, 4))
    plt.plot(x_cost, buy_cost, label="buy_cost", color="tab:red")
    plt.plot(x_reward, reward_sum, label="team_reward", color="tab:green")
    plt.xlabel("Hour index")
    plt.ylabel("Value")
    plt.title(f"Cost & Team Reward (episode={ep})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"episode_{ep:06d}_cost_reward.png", dpi=200)
    plt.close(fig)

    x_act, pd_by_agent = _series_by_agent(df_ep, "pd")
    _, pb_by_agent = _series_by_agent(df_ep, "pb")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for agent, y in pd_by_agent.items():
        axes[0].plot(x_act, y, label=agent)
    axes[0].set_ylabel("DG power (pd)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=min(4, max(1, len(pd_by_agent))))
    for agent, y in pb_by_agent.items():
        axes[1].plot(x_act, y, label=agent)
    axes[1].set_xlabel("Hour index")
    axes[1].set_ylabel("Battery power (pb)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Actions (episode={ep})")
    fig.tight_layout()
    fig.savefig(out_dir / f"episode_{ep:06d}_actions.png", dpi=200)
    plt.close(fig)

    return ep


def plot_aggregate(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce").astype("Int64")
    df["col_idx"] = pd.to_numeric(df["col_idx"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["episode", "col_idx"]).copy()
    df["episode"] = df["episode"].astype(int)
    df["col_idx"] = df["col_idx"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    soc_stats = (
        df.groupby(["agent", "col_idx"])["soc"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["agent", "col_idx"])
    )

    fig = plt.figure(figsize=(10, 4))
    for agent, g in soc_stats.groupby("agent"):
        plt.plot(g["col_idx"].to_numpy(), g["mean"].to_numpy(), label=str(agent))
    plt.ylim(0, 1)
    plt.xlabel("Hour index")
    plt.ylabel("SoC (mean over episodes)")
    plt.title("Average SoC Profile (test episodes)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_soc_mean.png", dpi=200)
    plt.close(fig)

    shared = df.sort_values(["episode", "col_idx", "agent"]).groupby(["episode", "col_idx"]).first().reset_index()
    pg_stats = shared.groupby("col_idx")["pg_total"].agg(["mean", "std"]).reset_index().sort_values("col_idx")
    price_stats = shared.groupby("col_idx")["price"].agg(["mean", "std"]).reset_index().sort_values("col_idx")

    fig = plt.figure(figsize=(10, 4))
    ax1 = plt.gca()
    ax1.plot(pg_stats["col_idx"].to_numpy(), pg_stats["mean"].to_numpy(), color="tab:blue")
    ax1.set_xlabel("Hour index")
    ax1.set_ylabel("pg_total (mean)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(price_stats["col_idx"].to_numpy(), price_stats["mean"].to_numpy(), color="tab:orange")
    ax2.set_ylabel("price (mean)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title("Average Grid Exchange & Price Profile (test episodes)")
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_pg_price_mean.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode", type=int, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    plot_aggregate(csv_path=csv_path, out_dir=out_dir)
    ep = plot_microgrid(csv_path=csv_path, out_dir=out_dir, episode=args.episode)
    print(str(out_dir.resolve()))
    print(ep)


if __name__ == "__main__":
    main()
