#!/usr/bin/env python3
"""
Objective J = weighted L1 distance between cumulative count curves (real vs simulated)
for buckets: (phase1,d3), (phase1,d4), (phase2,d3), (phase2,d4)

Per experiment:
- Change ONE parameter from its default (75 samples in its range; default removed; add -1 if "off")
- Keep all other parameters at their default
- Update osm.type.xml
- Rerun ALL cycles (Info_Cycle_*)
- Compute J from pooled exit-time samples by phase & direction
- Save experiment_results.csv with parameters + objective

Notes:
- Assumes per-cycle real CSVs exist as: Data/Info_Cycle_X.csv (same as optimization_process.py)
- Assumes phase boundaries live in: Data/TrafficLightCycles.csv (same as optimization_process.py)
- Assumes each cycle directory contains osm.sumocfg and InfoProcess.py works with --work-dir <cycle>
"""

import os
import glob
import math
import copy
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from time import sleep

# -----------------------------
# Settings
# -----------------------------
N_SAMPLES = 75
PEN = 1e3
PHASE_FALLBACK = [(0.0, 60.0), (60.0, 90.0)]  # if phase_df lookup fails
BUCKET_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)  # (p1,d3),(p1,d4),(p2,d3),(p2,d4)

# where each cycle lives
cycle_dirs = sorted(glob.glob("Info_Cycle_*"))
if not cycle_dirs:
    raise RuntimeError("No cycle directories found matching Info_Cycle_*")

# load phase boundaries (same schema as your optimizer: Start, Break, Continuation, Stop)
phase_df = pd.read_csv("Data/TrafficLightCycles.csv", delimiter=";")

# -----------------------------
# Parameter configuration (your SA list)
# -----------------------------
param_config = {
    # Per-vehicle-ish (if you don't provide Car_speedFactor, etc., it will apply to all vTypes)
    "speedFactor": {"default": 1.0, "range": [0.5, 2.0]},
    "speedDev": {"default": 0.1, "range": [0.1, 0.5]},
    "minGap": {"default": 2.5, "range": [1.0, 5.0]},
    "accel": {"default": 2.6, "range": [0.5, 8.0]},
    "decel": {"default": 4.5, "range": [0.5, 8.0]},
    "startupDelay": {"default": 0.0, "range": [0.0, 5.0]},
    "sigma": {"default": 0.5, "range": [0.0, 1.0]},
    "tau": {"default": 1.5, "range": [1.0, 3.0]},

    # Lane-changing model
    "lcStrategic": {"default": 1.0, "range": [0.0, 100.0], "off": True},
    "lcCooperative": {"default": 1.0, "range": [0.0, 1.0], "off": True},
    "lcSpeedGain": {"default": 1.0, "range": [0.0, 100.0]},
    "lcKeepRight": {"default": 1.0, "range": [0.0, 100.0]},
    "lcContRight": {"default": 1.0, "range": [0.0, 1.0]},
    "lcOvertakeRight": {"default": 0.0, "range": [0.0, 100.0]},
    "lcStrategicLookahead": {"default": 3000.0, "range": [1000.0, 5000.0]},
    "lcLookaheadLeft": {"default": 2.0, "range": [0.0, 10.0]},
    "lcSpeedGainRight": {"default": 0.1, "range": [0.0, 10.0]},
    "lcSpeedGainLookahead": {"default": 5.0, "range": [0.0, 10.0]},
    "lcSpeedGainRemainTime": {"default": 20.0, "range": [0.0, 40.0]},
    "lcOvertakeDeltaSpeedFactor": {"default": 0.0, "range": [-1.0, 1.0]},
    "lcKeepRightAcceptanceTime": {"default": -1.0, "range": [0.0, 40.0], "off": True},
    "lcCooperativeSpeed": {"default": 1.0, "range": [0.0, 1.0]},
    "minGapLat": {"default": 0.6, "range": [0.1, 2.5]},
    "lcSublane": {"default": 1.0, "range": [0.0, 10.0]},
    "lcPushy": {"default": 0.0, "range": [0.0, 1.0]},
    "lcPushyGap": {"default": 0.6, "range": [0.0, 0.6]},
    "lcAssertive": {"default": 1.0, "range": [0.0, 10.0]},
    "lcImpatience": {"default": 0.0, "range": [0.0, 2.0]},
    "lcTurnAlignmentDistance": {"default": 0.0, "range": [0.0, 10.0]},
    "lcLaneDiscipline": {"default": 0.0, "range": [0.0, 10.0]},
    "lcSigma": {"default": 0.0, "range": [0.0, 10.0]},

    # Junction model
    "jmIgnoreKeepClearTime": {"default": -1.0, "range": [0.0, 10.0], "off": True},
    "jmDriveAfterRedTime": {"default": -1.0, "range": [0.0, 10.0], "off": True},
    "jmDriveAfterYellowTime": {"default": -1.0, "range": [0.0, 10.0], "off": True},
    "jmIgnoreFoeProb": {"default": 0.0, "range": [0.0, 1.0]},
    "jmIgnoreJunctionFoeProb": {"default": 0.0, "range": [0.0, 1.0]},
    "jmSigmaMinor": {"default": 0.5, "range": [0.0, 1.0]},
    "jmStoplineGap": {"default": 1.0, "range": [0.0, 5.0]},
    "jmTimegapMinor": {"default": 1.0, "range": [1.0, 5.0]},
    "jmExtraGap": {"default": 0.0, "range": [0.0, 5.0]},
    "jmStopSignWait": {"default": 1.0, "range": [1.0, 5.0]},
    "impatience": {"default": 0.0, "range": [0.0, 1.0]},
}

# If your vType ids differ, adjust here.
# This list is used only for per-vehicle overrides like Car_speedFactor etc.
vehicle_type_ids = ["Car", "Bus", "MediumVehicle", "HeavyVehicle", "Motorcycle", "Taxi", "Medium", "Heavy"]


# -----------------------------
# XML update (robust: supports either global keys OR per-vehicle keys like Car_minGap)
# -----------------------------
def update_osm_type_xml(opts: dict, xml_file="osm.type.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for v in root.findall(".//vType"):
        vid = v.get("id", "")

        for k in param_config.keys():
            # Per-vehicle override wins if provided (e.g., Car_minGap)
            per_key = f"{vid}_{k}"
            if per_key in opts:
                v.set(k, str(opts[per_key]))
            elif k in opts:
                # Otherwise, apply the (global) value
                v.set(k, str(opts[k]))

    tree.write(xml_file)


# -----------------------------
# Phase windows (same idea as optimizer)
# -----------------------------
def get_phase_bounds_for_cycle(base_name: str):
    try:
        idx = int(base_name.split("_")[-1]) - 1
        if 0 <= idx < len(phase_df):
            row = phase_df.iloc[idx]
            return [
                (float(row.Start), float(row.Break)),
                (float(row.Continuation), float(row.Stop)),
            ]
    except Exception:
        pass
    return PHASE_FALLBACK


# -----------------------------
# Run one cycle (same robustness as optimizer)
# -----------------------------
def run_cycle(cycle_dir: str):
    sim_csv = os.path.join(cycle_dir, "simulation_info.csv")
    try:
        subprocess.run(["sumo", "-c", "osm.sumocfg", "--no-warnings"], cwd=cycle_dir, check=True)
        sleep(0.2)
        subprocess.run(["python", "InfoProcess.py", "--work-dir", cycle_dir], check=True)

        # wait up to ~10s
        for _ in range(50):
            if os.path.isfile(sim_csv) and os.path.getsize(sim_csv) > 0:
                return (cycle_dir, True)
            sleep(0.2)
        return (cycle_dir, False)
    except Exception:
        return (cycle_dir, False)


# -----------------------------
# Objective J (counts-based) — matches your optimizer’s logic
# -----------------------------
def compute_objective_counts(good_cycles: list[str]) -> float:
    # pooled exit_time samples by [phase][direction]
    real_times = [[[] for _ in range(2)] for _ in range(2)]
    sim_times  = [[[] for _ in range(2)] for _ in range(2)]
    phase_lens_seen = [[], []]

    for cycle in good_cycles:
        base = os.path.basename(cycle)
        phases = get_phase_bounds_for_cycle(base)
        real_csv = os.path.join("Data", f"{base}.csv")
        sim_csv  = os.path.join(cycle, "simulation_info.csv")

        if not os.path.isfile(real_csv):
            # skip missing real
            continue
        if not os.path.isfile(sim_csv):
            continue

        try:
            df_r = pd.read_csv(real_csv, delimiter=";")
            df_s = pd.read_csv(sim_csv,  delimiter=";")
        except Exception:
            continue

        # Make sure exit_time is numeric
        df_r["exit_time"] = df_r["exit_time"].astype(float)
        df_s["exit_time"] = df_s["exit_time"].astype(float)
        df_r["exit_detector"] = df_r["exit_detector"].astype(str)
        df_s["exit_detector"] = df_s["exit_detector"].astype(str)

        for p, (t0, t1) in enumerate(phases):
            phase_len = max(0.0, float(t1) - float(t0))
            phase_lens_seen[p].append(phase_len)

            subr = df_r[(df_r.exit_time >= t0) & (df_r.exit_time < t1)]
            subs = df_s[(df_s.exit_time >= t0) & (df_s.exit_time < t1)]

            for d, prefix in enumerate(("d3", "d4")):
                real_times[p][d].extend(
                    (subr[subr.exit_detector.str.startswith(prefix)].exit_time - t0).tolist()
                )
                sim_times[p][d].extend(
                    (subs[subs.exit_detector.str.startswith(prefix)].exit_time - t0).tolist()
                )

    # if no data at all -> penalty
    if all(len(real_times[p][d]) == 0 and len(sim_times[p][d]) == 0 for p in (0, 1) for d in (0, 1)):
        return float(PEN)

    # phase length used for grid (integer seconds)
    default_lens = [60.0, 30.0]
    phase_len = [
        int(np.max(phase_lens_seen[0])) if phase_lens_seen[0] else int(default_lens[0]),
        int(np.max(phase_lens_seen[1])) if phase_lens_seen[1] else int(default_lens[1]),
    ]

    l1_vals = []
    for (p, d) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        rt = np.sort(np.asarray(real_times[p][d], dtype=float))
        st = np.sort(np.asarray(sim_times[p][d],  dtype=float))

        T = max(1, int(phase_len[p]))
        grid = np.arange(0, T + 1)

        Cr = np.searchsorted(rt, grid, side="right") if rt.size else np.zeros_like(grid)
        Cs = np.searchsorted(st, grid, side="right") if st.size else np.zeros_like(grid)

        mae = float(np.mean(np.abs(Cr - Cs)))
        l1_vals.append(mae)

    J = float(np.dot(BUCKET_WEIGHTS, np.asarray(l1_vals, dtype=float)))
    return J


# -----------------------------
# Experiment generation: one-at-a-time, 75 samples, default removed, add -1 if off
# -----------------------------
def generate_experiments_from_config(cfg: dict, n_samples: int = 75):
    experiments = []
    for param, c in cfg.items():
        default = float(c["default"])
        lo, hi = map(float, c["range"])

        samples = np.linspace(lo, hi, n_samples)
        alt = [float(s) for s in samples if not math.isclose(float(s), default, rel_tol=1e-9, abs_tol=1e-9)]

        if c.get("off", False):
            if not any(math.isclose(a, -1.0, rel_tol=1e-9, abs_tol=1e-9) for a in alt):
                alt.append(-1.0)

        alt = sorted(alt)

        # one-at-a-time: all defaults except this param
        for v in alt:
            exp = {k: float(cfg[k]["default"]) for k in cfg.keys()}
            exp[param] = v
            experiments.append(exp)

        print(f"[gen] {param}: {len(alt)} alternatives")

    print(f"[gen] total experiments: {len(experiments)}")
    return experiments


# -----------------------------
# Main sensitivity loop
# -----------------------------
def main():
    experiments = generate_experiments_from_config(param_config, n_samples=N_SAMPLES)
    total = len(experiments)

    results = []

    for i, opts in enumerate(experiments, start=1):
        print(f"\n=== SA Experiment {i}/{total} ===")

        # 1) Update XML (in root; cycles read their local osm.type.xml if they have copies.
        #    If each cycle has its own osm.type.xml, you should copy it into each cycle dir too.
        update_osm_type_xml(opts, xml_file="osm.type.xml")

        # Optional: if cycles each have their own osm.type.xml, uncomment:
        # for cyc in cycle_dirs:
        #     update_osm_type_xml(opts, xml_file=os.path.join(cyc, "osm.type.xml"))

        # 2) Run all cycles (parallel)
        with ProcessPoolExecutor() as exe:
            run_results = list(exe.map(run_cycle, cycle_dirs))

        good_cycles = [cyc for cyc, ok in run_results if ok]
        if not good_cycles:
            J = float(PEN)
        else:
            J = compute_objective_counts(good_cycles)

        row = copy.deepcopy(opts)
        row["objective"] = float(J)
        results.append(row)

        print(f"objective = {J:.6f}")

    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    print("\nSaved: experiment_results.csv")


if __name__ == "__main__":
    main()
