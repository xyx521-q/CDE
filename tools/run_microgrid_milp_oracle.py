import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]


def _as_agent_array(values):
    return np.asarray(values, dtype=float)


def apply_parameter_overrides(params, dg_max_override=None, battery_cap_override=None):
    overridden = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in params.items()
    }
    if dg_max_override is not None:
        overridden["dg_max"] = np.full_like(
            overridden["dg_max"], float(dg_max_override), dtype=float
        )
    if battery_cap_override is not None:
        overridden["battery_caps"] = np.full_like(
            overridden["battery_caps"], float(battery_cap_override), dtype=float
        )
    return overridden


def load_case(env_config, seed, split, num_days):
    config_path = ROOT / "src" / "config" / "envs" / f"{env_config}.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_args = cfg["env_args"]
    data_path = ROOT / "src" / "envs" / "data"
    price = np.loadtxt(data_path / "price.csv", delimiter=",")
    load = np.loadtxt(data_path / "load.csv", delimiter=",")
    pv = np.loadtxt(data_path / "pv.csv", delimiter=",")
    load_pv = load - pv

    agents = env_args["data"]
    params = {
        "dg_max": _as_agent_array([a["dg_max"] for a in agents]),
        "battery_caps": _as_agent_array([a["battery_cap"] for a in agents]),
        "raw_ch": _as_agent_array([a["raw_ch"] for a in agents]),
        "raw_dis": _as_agent_array([a["raw_dis"] for a in agents]),
        "costa": _as_agent_array([a["costa"] for a in agents]),
        "costb": _as_agent_array([a["costb"] for a in agents]),
        "costc": _as_agent_array([a["costc"] for a in agents]),
        "env_param": _as_agent_array([a["env_param"] for a in agents]),
        "split_ratio": _as_agent_array([a["split_ratio"] for a in agents]),
        "curtailment_penalty_weight": float(
            env_args.get("curtailment_penalty_weight", 0.0)
        ),
    }

    rng = np.random.default_rng(seed)
    perm = rng.permutation(price.shape[0])
    split_point = int(price.shape[0] * 0.8)
    if split == "test":
        row_indices = np.sort(perm[split_point:])
    elif split == "train":
        row_indices = np.sort(perm[:split_point])
    else:
        row_indices = np.arange(price.shape[0])

    return {
        "price": price,
        "load_pv": load_pv,
        "row_indices": row_indices[:num_days],
        "params": params,
    }


def solve_dispatch(
    load_pv,
    price,
    params,
    initial_soc=0.35,
    terminal_soc=0.35,
    cycle_cost=0.0,
    curtailment_penalty_weight=None,
    mip_rel_gap=1e-8,
    time_limit=None,
):
    load_pv = np.asarray(load_pv, dtype=float)
    price = np.asarray(price, dtype=float)
    n_steps = int(load_pv.shape[0])
    n_agents = int(params["battery_caps"].shape[0])
    caps = np.asarray(params["battery_caps"], dtype=float)
    dg_max = np.asarray(params["dg_max"], dtype=float)
    raw_ch = np.asarray(params["raw_ch"], dtype=float)
    raw_dis = np.asarray(params["raw_dis"], dtype=float)
    gen_linear_cost = np.asarray(params.get("costb", np.zeros(n_agents)), dtype=float)
    env_linear_cost = np.asarray(
        params.get("env_param", np.zeros(n_agents)), dtype=float
    )
    if curtailment_penalty_weight is None:
        curtailment_penalty_weight = float(
            params.get("curtailment_penalty_weight", 0.0)
        )
    resolved_curtailment_penalty_weight = float(curtailment_penalty_weight)

    grid_start = 0
    pd_start = grid_start + n_steps
    charge_start = pd_start + n_steps * n_agents
    discharge_start = charge_start + n_steps * n_agents
    curtailment_start = discharge_start + n_steps * n_agents
    soc_start = curtailment_start + n_steps
    is_charging_start = soc_start + (n_steps + 1) * n_agents
    n_vars = is_charging_start + n_steps * n_agents

    def idx_grid(t):
        return grid_start + t

    def idx_pd(t, a):
        return pd_start + t * n_agents + a

    def idx_charge(t, a):
        return charge_start + t * n_agents + a

    def idx_discharge(t, a):
        return discharge_start + t * n_agents + a

    def idx_curtailment(t):
        return curtailment_start + t

    def idx_soc(t, a):
        return soc_start + t * n_agents + a

    def idx_is_charging(t, a):
        return is_charging_start + t * n_agents + a

    c = np.zeros(n_vars, dtype=float)
    lb = np.zeros(n_vars, dtype=float)
    ub = np.full(n_vars, np.inf, dtype=float)
    integrality = np.zeros(n_vars, dtype=int)

    for t in range(n_steps):
        c[idx_grid(t)] = price[t]
        c[idx_curtailment(t)] = float(curtailment_penalty_weight)
        for a in range(n_agents):
            c[idx_pd(t, a)] = gen_linear_cost[a] + env_linear_cost[a]
            c[idx_charge(t, a)] = cycle_cost
            c[idx_discharge(t, a)] = cycle_cost
            ub[idx_pd(t, a)] = dg_max[a]
            ub[idx_charge(t, a)] = caps[a]
            ub[idx_discharge(t, a)] = caps[a]
        ub[idx_curtailment(t)] = np.inf

    for t in range(n_steps):
        for a in range(n_agents):
            ub[idx_is_charging(t, a)] = 1.0
            integrality[idx_is_charging(t, a)] = 1

    for t in range(n_steps + 1):
        for a in range(n_agents):
            lb[idx_soc(t, a)] = 0.2
            ub[idx_soc(t, a)] = 0.8

    constraints = []
    lower = []
    upper = []

    def add_constraint(entries, lo, hi):
        constraints.append(entries)
        lower.append(lo)
        upper.append(hi)

    for a in range(n_agents):
        add_constraint({idx_soc(0, a): 1.0}, initial_soc, initial_soc)
        if terminal_soc is not None:
            add_constraint({idx_soc(n_steps, a): 1.0}, terminal_soc, terminal_soc)

    for t in range(n_steps):
        add_constraint(
            {
                idx_grid(t): 1.0,
                **{idx_pd(t, a): 1.0 for a in range(n_agents)},
                **{idx_discharge(t, a): 1.0 for a in range(n_agents)},
                **{idx_charge(t, a): -1.0 for a in range(n_agents)},
                idx_curtailment(t): -1.0,
            },
            load_pv[t],
            load_pv[t],
        )
        for a in range(n_agents):
            add_constraint(
                {
                    idx_soc(t + 1, a): 1.0,
                    idx_soc(t, a): -1.0,
                    idx_charge(t, a): -raw_ch[a] / caps[a],
                    idx_discharge(t, a): raw_dis[a] / caps[a],
                },
                0.0,
                0.0,
            )
            add_constraint(
                {
                    idx_charge(t, a): 1.0,
                    idx_is_charging(t, a): -caps[a],
                },
                -np.inf,
                0.0,
            )
            add_constraint(
                {
                    idx_discharge(t, a): 1.0,
                    idx_is_charging(t, a): caps[a],
                },
                -np.inf,
                caps[a],
            )

    matrix = lil_matrix((len(constraints), n_vars), dtype=float)
    for row, entries in enumerate(constraints):
        for col, value in entries.items():
            matrix[row, col] = value

    options = {"mip_rel_gap": mip_rel_gap}
    if time_limit is not None:
        options["time_limit"] = time_limit

    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(
            matrix.tocsr(), np.asarray(lower), np.asarray(upper)
        ),
        options=options,
    )
    if not res.success:
        return {
            "success": False,
            "message": res.message,
            "status": int(res.status),
        }

    x = res.x
    grid_import = np.array([x[idx_grid(t)] for t in range(n_steps)])
    curtailment = np.array([x[idx_curtailment(t)] for t in range(n_steps)])
    pd = np.array([[x[idx_pd(t, a)] for a in range(n_agents)] for t in range(n_steps)])
    charge = np.array(
        [[x[idx_charge(t, a)] for a in range(n_agents)] for t in range(n_steps)]
    )
    discharge = np.array(
        [[x[idx_discharge(t, a)] for a in range(n_agents)] for t in range(n_steps)]
    )
    soc = np.array(
        [[x[idx_soc(t, a)] for a in range(n_agents)] for t in range(n_steps + 1)]
    )

    generation_cost = float(
        np.sum(
            params.get("costa", 0.0) * pd**2
            + params.get("costb", 0.0) * pd
            + params.get("costc", 0.0)
        )
    )
    env_cost = float(np.sum(params.get("env_param", 0.0) * pd))
    battery_throughput = float(np.sum(charge + discharge))
    grid_purchase_cost = float(np.sum(grid_import * price))
    baseline_grid_purchase_cost = float(np.sum(np.maximum(load_pv, 0.0) * price))
    curtailment_total = float(np.sum(curtailment))
    curtailment_penalty = -resolved_curtailment_penalty_weight * curtailment_total
    objective = float(res.fun)

    return {
        "success": True,
        "message": res.message,
        "objective": objective,
        "grid_import": grid_import,
        "curtailment": curtailment,
        "pd": pd,
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
        "grid_purchase_cost": grid_purchase_cost,
        "baseline_grid_purchase_cost": baseline_grid_purchase_cost,
        "grid_purchase_saving": baseline_grid_purchase_cost - grid_purchase_cost,
        "grid_purchase_saving_ratio": (baseline_grid_purchase_cost - grid_purchase_cost)
        / max(baseline_grid_purchase_cost, 1e-6),
        "generation_cost": generation_cost,
        "env_cost": env_cost,
        "curtailment_penalty": curtailment_penalty,
        "curtailment_penalty_weight": resolved_curtailment_penalty_weight,
        "battery_throughput": battery_throughput,
        "mean_soc": float(np.mean(soc[1:])),
    }


def summarize(results):
    keys = [
        "objective",
        "grid_purchase_cost",
        "baseline_grid_purchase_cost",
        "grid_purchase_saving",
        "grid_purchase_saving_ratio",
        "generation_cost",
        "env_cost",
        "battery_throughput",
        "mean_soc",
    ]
    return {f"{key}_mean": float(np.mean([r[key] for r in results])) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="microgrid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--num-days", type=int, default=20)
    parser.add_argument("--initial-soc", type=float, default=0.35)
    parser.add_argument("--terminal-soc", type=float, default=0.35)
    parser.add_argument("--cycle-cost", type=float, default=0.0)
    parser.add_argument("--curtailment-penalty-weight", type=float, default=None)
    parser.add_argument("--dg-max-override", type=float, default=None)
    parser.add_argument("--battery-cap-override", type=float, default=None)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-8)
    parser.add_argument("--time-limit", type=float, default=None)
    args = parser.parse_args()

    case = load_case(args.env_config, args.seed, args.split, args.num_days)
    case["params"] = apply_parameter_overrides(
        case["params"],
        dg_max_override=args.dg_max_override,
        battery_cap_override=args.battery_cap_override,
    )
    results = []
    schedules = {}
    for idx, row_idx in enumerate(case["row_indices"], start=1):
        print(
            f"[{idx}/{len(case['row_indices'])}] solving MILP row {int(row_idx)} ...",
            flush=True,
        )
        result = solve_dispatch(
            case["load_pv"][row_idx],
            case["price"][row_idx],
            case["params"],
            initial_soc=args.initial_soc,
            terminal_soc=args.terminal_soc,
            cycle_cost=args.cycle_cost,
            curtailment_penalty_weight=args.curtailment_penalty_weight,
            mip_rel_gap=args.mip_rel_gap,
            time_limit=args.time_limit,
        )
        if not result["success"]:
            raise RuntimeError(
                f"MILP failed for row {int(row_idx)}: {result['message']}"
            )

        serializable = {
            key: value
            for key, value in result.items()
            if key
            not in {"grid_import", "pd", "charge", "discharge", "soc", "curtailment"}
        }
        serializable["row_idx"] = int(row_idx)
        results.append(serializable)
        schedules[str(int(row_idx))] = {
            "grid_import": result["grid_import"].tolist(),
            "pd": result["pd"].tolist(),
            "charge": result["charge"].tolist(),
            "discharge": result["discharge"].tolist(),
            "soc": result["soc"].tolist(),
        }
        print(
            f"  objective={result['objective']:.2f} "
            f"grid_cost={result['grid_purchase_cost']:.2f} "
            f"saving_ratio={result['grid_purchase_saving_ratio']:.4f} "
            f"battery_throughput={result['battery_throughput']:.2f}",
            flush=True,
        )

    summary = {
        "seed": args.seed,
        "split": args.split,
        "num_days": len(results),
        "initial_soc": args.initial_soc,
        "terminal_soc": args.terminal_soc,
        "cycle_cost": args.cycle_cost,
        "curtailment_penalty_weight": float(
            case["params"].get("curtailment_penalty_weight", 0.0)
            if args.curtailment_penalty_weight is None
            else args.curtailment_penalty_weight
        ),
        "dg_max_override": args.dg_max_override,
        "battery_cap_override": args.battery_cap_override,
        "model": "MILP economic dispatch with linear generation/environment costs and binary charge/discharge exclusivity",
        **summarize(results),
    }

    out_dir = ROOT / "results" / "oracle_milp"
    out_dir.mkdir(parents=True, exist_ok=True)
    override_parts = []
    if args.dg_max_override is not None:
        override_parts.append(f"dg{args.dg_max_override:g}")
    if args.battery_cap_override is not None:
        override_parts.append(f"batt{args.battery_cap_override:g}")
    override_suffix = "_" + "_".join(override_parts) if override_parts else ""
    stem = f"{args.env_config}_{args.split}_{len(results)}d_seed{args.seed}{override_suffix}"
    with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / f"{stem}_episodes.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / f"{stem}_schedules.json", "w", encoding="utf-8") as f:
        json.dump(schedules, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
