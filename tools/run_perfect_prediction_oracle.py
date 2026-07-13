import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.maenv import MGEnv  # noqa: E402


def simulate_episode(x, env, row_idx, initial_socs):
    t_horizon = env.episode_limit
    n_agents = env.n_agents
    dg_max = env.params["dg_max"]
    battery_caps = env.params["battery_caps"]
    raw_ch = env.params["raw_ch"]
    raw_dis = env.params["raw_dis"]
    env_param = env.params["env_param"]
    bessa = env.params["bessa"]
    bessb = env.params["bessb"]
    bessc = env.params["bessc"]
    costa = env.params["costa"]
    costb = env.params["costb"]
    costc = env.params["costc"]

    actions = x.reshape(t_horizon, n_agents, 2)
    socs = initial_socs.astype(float).copy()

    total_reward = 0.0
    total_grid_purchase_cost = 0.0
    total_baseline_grid_purchase_cost = 0.0
    total_grid_purchase_saving = 0.0
    total_generate_cost = 0.0
    total_bess_cost = 0.0
    total_env_cost = 0.0
    total_curtailment_penalty = 0.0
    total_soc_reserve_penalty = 0.0
    mean_soc_sum = 0.0

    for t in range(t_horizon):
        price = float(env.data_dict["price"][row_idx, t])
        load_pv = float(env.data_dict["load_pv"][row_idx, t])

        pd_frac = np.clip(actions[t, :, 0], 0.0, 1.0)
        pb_gate = np.clip(actions[t, :, 1], -1.0, 1.0)

        pds = pd_frac * dg_max

        # Map [-1, 1] into the dynamically feasible battery power range so SOC stays valid.
        pb_low = -((0.8 - socs) * battery_caps / raw_ch)
        pb_high = (socs - 0.2) * battery_caps / raw_dis
        pbs = pb_low + 0.5 * (pb_gate + 1.0) * (pb_high - pb_low)

        delta_soc = np.where(
            pbs < 0,
            -raw_ch * pbs / battery_caps,
            -raw_dis * pbs / battery_caps,
        )
        socs = np.clip(socs + delta_soc, 0.2, 0.8)

        pg = load_pv - float(np.sum(pds)) - float(np.sum(pbs))

        x_bess = pbs + 3.0 * battery_caps * (1.0 - socs)
        bess_poly = -np.abs(bessa * x_bess**2 + bessb * x_bess + bessc)
        generate_costs = -(costa * pds**2 + costb * pds + costc)
        env_reward = -(pds * env_param)
        soc_reserve_penalty = -env.soc_reserve_weight * np.maximum(
            env.soc_reserve_target - socs, 0.0
        )

        baseline_pg = max(load_pv, 0.0)
        baseline_grid_purchase_cost = baseline_pg * price
        actual_grid_purchase_cost = max(pg, 0.0) * price
        curtailment = max(-pg, 0.0)
        curtailment_penalty = -env.curtailment_penalty_weight * curtailment
        if env.buy_reward_mode == "relative_saving":
            buy_cost = env.buy_reward_scale * (
                (baseline_grid_purchase_cost - actual_grid_purchase_cost)
                / max(baseline_grid_purchase_cost, 1e-6)
            )
        else:
            buy_cost = -actual_grid_purchase_cost

        eco_reward = (
            env.generate_cost_weight * generate_costs
            + env.bess_cost_weight * bess_poly
            + env.buy_cost_weight * buy_cost
            + curtailment_penalty
        )
        reward_list = (
            eco_reward + env.env_reward_weight * env_reward + soc_reserve_penalty
        )

        if env.reward_aggregate == "mean":
            total_reward += float(np.mean(reward_list)) * env.reward_scale
        else:
            total_reward += float(np.sum(reward_list)) * env.reward_scale

        total_grid_purchase_cost += float(actual_grid_purchase_cost)
        total_baseline_grid_purchase_cost += float(baseline_grid_purchase_cost)
        total_grid_purchase_saving += float(
            baseline_grid_purchase_cost - actual_grid_purchase_cost
        )
        total_generate_cost += float(np.sum(-generate_costs))
        total_bess_cost += float(np.sum(-bess_poly))
        total_env_cost += float(np.sum(-env_reward))
        total_curtailment_penalty += float(curtailment_penalty)
        total_soc_reserve_penalty += float(np.sum(-soc_reserve_penalty))
        mean_soc_sum += float(np.mean(socs))

    return {
        "episode_return": total_reward,
        "grid_purchase_cost": total_grid_purchase_cost,
        "baseline_grid_purchase_cost": total_baseline_grid_purchase_cost,
        "grid_purchase_saving": total_grid_purchase_saving,
        "grid_purchase_saving_ratio": total_grid_purchase_saving
        / max(total_baseline_grid_purchase_cost, 1e-6),
        "generate_cost": total_generate_cost,
        "bess_cost": total_bess_cost,
        "env_cost": total_env_cost,
        "curtailment_penalty": total_curtailment_penalty,
        "soc_reserve_penalty": total_soc_reserve_penalty,
        "mean_soc": mean_soc_sum / t_horizon,
    }


def objective_fn(x, env, row_idx, initial_socs):
    metrics = simulate_episode(x, env, row_idx, initial_socs)
    return -metrics["episode_return"]


def make_initial_guess(env, row_idx, initial_socs, mode):
    t_horizon = env.episode_limit
    n_agents = env.n_agents
    dg_max = env.params["dg_max"]
    load_pv = env.data_dict["load_pv"][row_idx]
    price = env.data_dict["price"][row_idx]

    x0 = np.zeros((t_horizon, n_agents, 2), dtype=np.float64)
    if mode == "gen_cover":
        positive_net = np.maximum(load_pv, 0.0)
        pd = np.clip((positive_net[:, None] / n_agents) / dg_max[None, :], 0.0, 1.0)
        x0[:, :, 0] = pd
    elif mode == "price_shift":
        median_price = float(np.median(price))
        positive_net = np.maximum(load_pv, 0.0)
        pd = np.clip((positive_net[:, None] / n_agents) / dg_max[None, :], 0.0, 1.0)
        x0[:, :, 0] = pd
        x0[:, :, 1] = np.where(price[:, None] >= median_price, 0.7, -0.7)
    return x0.reshape(-1)


def optimize_day(env, row_idx, initial_socs, maxiter):
    bounds = []
    n_vars = env.episode_limit * env.n_agents * 2
    for k in range(n_vars):
        if k % 2 == 0:
            bounds.append((0.0, 1.0))
        else:
            bounds.append((-1.0, 1.0))

    candidates = ["zeros", "gen_cover", "price_shift"]
    best = None
    best_res = None
    for mode in candidates:
        x0 = make_initial_guess(env, row_idx, initial_socs, mode)
        res = minimize(
            objective_fn,
            x0,
            args=(env, row_idx, initial_socs),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-6, "maxfun": 50000},
        )
        metrics = simulate_episode(res.x, env, row_idx, initial_socs)
        if best is None or metrics["episode_return"] > best["episode_return"]:
            best = metrics
            best_res = res

    best["solver_success"] = bool(best_res.success)
    best["solver_message"] = str(best_res.message)
    best["solver_fun"] = float(best_res.fun)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="microgrid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-days", type=int, default=20)
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--initial-soc", type=float, default=0.35)
    parser.add_argument("--maxiter", type=int, default=120)
    args = parser.parse_args()

    config_path = ROOT / "src" / "config" / "envs" / f"{args.env_config}.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_args = cfg["env_args"]
    env_args["seed"] = args.seed
    env_args["result_dir"] = str(ROOT / "results")
    env = MGEnv(env_args, algo_name="perfect_prediction_oracle")

    if args.split == "test":
        row_indices = np.sort(env.test_indices)
    elif args.split == "train":
        row_indices = np.sort(env.train_indices)
    else:
        row_indices = np.arange(env.data_shape[0])
    row_indices = row_indices[: args.num_days]

    initial_socs = np.full(env.n_agents, args.initial_soc, dtype=float)
    results = []
    for idx, row_idx in enumerate(row_indices, start=1):
        print(f"[{idx}/{len(row_indices)}] optimizing row {row_idx} ...", flush=True)
        metrics = optimize_day(env, int(row_idx), initial_socs, args.maxiter)
        metrics["row_idx"] = int(row_idx)
        results.append(metrics)
        print(
            f"  return={metrics['episode_return']:.4f} "
            f"grid_cost={metrics['grid_purchase_cost']:.2f} "
            f"saving_ratio={metrics['grid_purchase_saving_ratio']:.4f}",
            flush=True,
        )

    summary = {
        "seed": args.seed,
        "split": args.split,
        "num_days": len(results),
        "initial_soc": args.initial_soc,
        "episode_return_mean": float(np.mean([r["episode_return"] for r in results])),
        "grid_purchase_cost_mean": float(
            np.mean([r["grid_purchase_cost"] for r in results])
        ),
        "baseline_grid_purchase_cost_mean": float(
            np.mean([r["baseline_grid_purchase_cost"] for r in results])
        ),
        "grid_purchase_saving_mean": float(
            np.mean([r["grid_purchase_saving"] for r in results])
        ),
        "grid_purchase_saving_ratio_mean": float(
            np.mean([r["grid_purchase_saving_ratio"] for r in results])
        ),
        "generate_cost_mean": float(np.mean([r["generate_cost"] for r in results])),
        "bess_cost_mean": float(np.mean([r["bess_cost"] for r in results])),
        "env_cost_mean": float(np.mean([r["env_cost"] for r in results])),
        "curtailment_penalty_mean": float(
            np.mean([r["curtailment_penalty"] for r in results])
        ),
        "soc_reserve_penalty_mean": float(
            np.mean([r["soc_reserve_penalty"] for r in results])
        ),
        "mean_soc_mean": float(np.mean([r["mean_soc"] for r in results])),
    }

    out_dir = ROOT / "results" / "oracle_perfect_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.env_config}_{args.split}_{len(results)}d_seed{args.seed}"
    with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / f"{stem}_episodes.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
