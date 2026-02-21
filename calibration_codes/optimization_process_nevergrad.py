#!/usr/bin/env python3
"""
optimization_process.py

Nevergrad (NGOpt)-driven calibration using only per-phase, per-direction vehicle counts.
Objective = weighted L1 distance between cumulative count curves (real vs. simulated)
for the four buckets: (phase1,d3), (phase1,d4), (phase2,d3), (phase2,d4).
"""

import os
import glob
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from time import sleep

# --- NEW: Nevergrad ---
import nevergrad as ng

# --------------------------
# 1. Parameter configuration 
# --------------------------
global_param_config = {
    # Lane-changing Model:
    "lcStrategic": {"default": 1.0, "range": [0, 100], "off": True},
    "lcLookaheadLeft": {"default": 2.0, "range": [0, 10]},
    "lcCooperativeSpeed": {"default": 1.0, "range": [0, 1]},
    "minGapLat": {"default": 0.6, "range": [0.1, 2.5]},
    "lcPushy": {"default": 0, "range": [0, 1]},
    "lcAssertive": {"default": 1, "range": [0, 10]},
    "lcImpatience": {"default": 0, "range": [0, 2]},
    "lcSigma": {"default": 0.0, "range": [0, 10]},
    "jmDriveAfterRedTime": {"default": -1, "range": [0, 10], "off": True},
    "jmDriveAfterYellowTime": {"default": -1, "range": [0, 10], "off": True},
}
global_keys   = list(global_param_config.keys())
global_bounds = [tuple(global_param_config[k]["range"]) for k in global_keys]

per_vehicle_param_config = {
    "speedFactor": {"default": 1.0, "range": [0.5, 2.0]},
    "speedDev": {"default": 0.1, "range": [0.1, 0.5]},
    "minGap": {"default": 2.5, "range": [1.0, 5.0]},
    "accel": {"default": 2.6, "range": [0.5, 8.0]},
    "sigma": {"default": 0.5, "range": [0, 1]},
    "tau": {"default": 1.5, "range": [1, 3]},
}
vehicle_types    = ["Car","Bus","Medium","Heavy","Motorcycle","Taxi"]
per_vehicle_keys = list(per_vehicle_param_config.keys())

per_vehicle_names  = []
per_vehicle_bounds = []
for vt in vehicle_types:
    for k in per_vehicle_keys:
        per_vehicle_names.append(f"{vt}_{k}")
        low, high = per_vehicle_param_config[k]["range"]
        if vt == "Motorcycle" and k == "speedFactor":
            low = max(low, 1.65)
        per_vehicle_bounds.append((low, high))

opt_param_names = global_keys + per_vehicle_names
bounds          = global_bounds + per_vehicle_bounds

# where each cycle lives
cycle_dirs = sorted(glob.glob("Info_Cycle_*"))

# load phase boundaries
phase_df = pd.read_csv("Data/TrafficLightCycles.csv", delimiter=';')

# ----------------------------------------
# 2. XML update
# ----------------------------------------
def update_osm_type_xml(opts, xml_file="osm.type.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for v in root.findall(".//vType"):
        for k in global_keys:
            v.set(k, str(opts[k]))
        vid = v.get("id")
        if vid in vehicle_types:
            for k in per_vehicle_keys:
                v.set(k, str(opts[f"{vid}_{k}"]))
    tree.write(xml_file)

# ----------------------------------------
# 3. Simulation helpers & robust phase windows
# ----------------------------------------
PEN = 1e3
PHASE_FALLBACK = [(0.0, 60.0), (60.0, 90.0)]  # if phase_df lookup fails

def get_phase_bounds_for_cycle(base_name):
    try:
        idx = int(base_name.split("_")[-1]) - 1
        if 0 <= idx < len(phase_df):
            row = phase_df.iloc[idx]
            return [(float(row.Start), float(row.Break)),
                    (float(row.Continuation), float(row.Stop))]
    except Exception:
        pass
    return PHASE_FALLBACK

def run_cycle(cycle):
    sim_csv = os.path.join(cycle, "simulation_info.csv")
    try:
        subprocess.run(["sumo", "-c", "osm.sumocfg"], cwd=cycle, check=True)
        sleep(0.2)
        subprocess.run(["python", "InfoProcess.py", "--work-dir", cycle], check=True)
        for _ in range(50):  # wait up to ~10s
            if os.path.isfile(sim_csv) and os.path.getsize(sim_csv) > 0:
                return (cycle, True)
            sleep(0.2)
        return (cycle, False)
    except Exception:
        return (cycle, False)

# ----------------------------------------
# 4. Objective: cumulative-count L1-distance
# ----------------------------------------
def objective(x):
    global iteration
    iteration += 1
    print(f"=== Iter {iteration} ===", end=" ")

    # 1) Update XML
    opts = {opt_param_names[i]: float(x[i]) for i in range(len(x))}
    update_osm_type_xml(opts)

    # 2) Rerun all cycles in parallel and FORCE completion
    with ProcessPoolExecutor() as exe:
        results = list(exe.map(run_cycle, cycle_dirs))

    good_cycles = [cyc for cyc, ok in results if ok]
    if not good_cycles:
        print(" [WARN] No cycles produced simulation_info.csv → penalty")
        return float(PEN)

    # 3) Pool raw exit_time samples by [phase][direction]
    real_times = [[[] for _ in range(2)] for _ in range(2)]
    sim_times  = [[[] for _ in range(2)] for _ in range(2)]
    phase_lens_seen = [[], []]

    for cycle in good_cycles:
        base = os.path.basename(cycle)
        phases = get_phase_bounds_for_cycle(base)
        real_csv = os.path.join("Data", f"{base}.csv")
        sim_csv  = os.path.join(cycle, "simulation_info.csv")
        if not os.path.isfile(real_csv):
            print(f" [WARN] Missing real CSV for {base}, skipping this cycle")
            continue
        try:
            df_r = pd.read_csv(real_csv, delimiter=';')
            df_s = pd.read_csv(sim_csv,  delimiter=';')
        except Exception:
            print(f" [WARN] Failed reading CSVs for {base}, skipping")
            continue

        for p, (t0, t1) in enumerate(phases):
            phase_len = max(0.0, float(t1) - float(t0))
            phase_lens_seen[p].append(phase_len)
            subr = df_r[(df_r.exit_time.astype(float) >= t0) &
                        (df_r.exit_time.astype(float) <  t1)]
            subs = df_s[(df_s.exit_time.astype(float) >= t0) &
                        (df_s.exit_time.astype(float) <  t1)]
            for d, prefix in enumerate(("d3", "d4")):
                real_times[p][d].extend(
                    (subr[subr.exit_detector.astype(str).str.startswith(prefix)]
                        .exit_time.astype(float) - t0).tolist()
                )
                sim_times[p][d].extend(
                    (subs[subs.exit_detector.astype(str).str.startswith(prefix)]
                        .exit_time.astype(float) - t0).tolist()
                )

    if all(len(real_times[p][d]) == 0 and len(sim_times[p][d]) == 0
           for p in (0,1) for d in (0,1)):
        print(" [WARN] No data collected across all buckets → penalty")
        return float(PEN)

    default_lens = [60.0, 30.0]
    phase_len = [
        int(np.max(phase_lens_seen[0])) if phase_lens_seen[0] else int(default_lens[0]),
        int(np.max(phase_lens_seen[1])) if phase_lens_seen[1] else int(default_lens[1]),
    ]

    bucket_weights = [0.4, 0.4, 0.1, 0.1]  # (p1,d3),(p1,d4),(p2,d3),(p2,d4)
    l1_vals = []

    for (p, d) in [(0,0),(0,1),(1,0),(1,1)]:
        rt = np.sort(np.asarray(real_times[p][d], dtype=float))
        st = np.sort(np.asarray(sim_times [p][d], dtype=float))
        T  = max(1, int(phase_len[p]))
        grid = np.arange(0, T + 1)
        Cr = np.searchsorted(rt, grid, side='right') if rt.size else np.zeros_like(grid)
        Cs = np.searchsorted(st, grid, side='right') if st.size else np.zeros_like(grid)
        mae = float(np.mean(np.abs(Cr - Cs)))
        l1_vals.append(mae)

    J = float(np.dot(bucket_weights, l1_vals))
    return J  # plain float for Nevergrad

# ----------------------------------------
# 5. Run Nevergrad (NGOpt)
# ----------------------------------------
if __name__ == "__main__":
    # bounds → arrays
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    x0 = (lb + ub) / 2.0

    # Parametrization as an Array with per-dimension bounds
    param = ng.p.Array(init=x0).set_bounds(lower=lb, upper=ub)

    # Create optimizer
    # - budget = number of objective evaluations
    # - num_workers=1 to avoid concurrent evals (since SUMO writes shared files)
    budget = 700  # adjust as needed
    optimizer = ng.optimizers.NGOpt(parametrization=param, budget=budget, num_workers=4)

    # Optional: reproducibility
    optimizer.parametrization.random_state = np.random.RandomState(42)

    # Ask/Tell loop so we can log history
    history_X = []
    history_F = []

    iteration = 0  # used by objective() for printing

    for _ in range(budget):
        candidate = optimizer.ask()
        x = np.asarray(candidate.value, dtype=float)
        J = objective(x)
        optimizer.tell(candidate, float(J))

        # log
        row = {opt_param_names[i]: x[i] for i in range(len(x))}
        row["objective"] = float(J)
        history_X.append(row)
        history_F.append(float(J))

    # Best found
    recommendation = optimizer.provide_recommendation()
    xbest = np.asarray(recommendation.value, dtype=float)
    fbest = float(min(history_F)) if history_F else float("nan")

    # Save history (same schema as before)
    df = pd.DataFrame(history_X)
    df.to_csv("optimization_history_nevergrad.csv", index=False)

    # Apply best parameters
    best_opts = {opt_param_names[i]: xbest[i] for i in range(len(xbest))}
    update_osm_type_xml(best_opts)

    print("\nDone! Best J =", fbest)
