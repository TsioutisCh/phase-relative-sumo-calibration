import os
import glob
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "lines.linewidth": 2.0,
})

COLORS = {"Observed": "#0A9E72", "Simulated": "#D13812"}

# -------------------------
# Settings
# -------------------------
DATA_DIR   = "Data"
CYCLE_GLOB = "Info_Cycle_*"
cycle_dirs = sorted(glob.glob(CYCLE_GLOB))

PHASES     = ["1", "2"]
DIRECTIONS = ["1", "2"]
BUCKETS    = [(p, d) for p in PHASES for d in DIRECTIONS]

EPS = 1e-9  # safeguard

# -------------------------
# Helpers
# -------------------------
def get_phase(cycle_folder_name):
    cycle_num = int(cycle_folder_name.split("_")[-1])
    return "1" if cycle_num <= 4 else "2"

def get_direction(detector_id):
    detector_id = str(detector_id)
    if detector_id.startswith("d3"):
        return "1"
    elif detector_id.startswith("d4"):
        return "2"
    return None

def cumulative_curve(times, T):
    times = np.sort(np.asarray(times, dtype=float))
    grid = np.arange(0, T + 1)
    return np.searchsorted(times, grid, side="right")

def mape_curve(C_obs, C_sim):
    """
    MAPE between two cumulative curves.
    Only evaluate time steps where C_obs > 0 to avoid division by zero.
    Returns a value in %.
    """
    C_obs = np.asarray(C_obs, dtype=float)
    C_sim = np.asarray(C_sim, dtype=float)

    mask = C_obs > 0
    if not np.any(mask):
        return np.nan

    perc_err = np.abs(C_obs[mask] - C_sim[mask]) / (C_obs[mask] + EPS)
    return 100.0 * float(np.mean(perc_err))


def print_bucket_stats(results_default, results_calib, buckets, label="MAPE (%)"):
    def stats(arr):
        a = np.asarray([x for x in arr if np.isfinite(x)], dtype=float)
        if a.size == 0:
            return None
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        iqr = q3 - q1
        return {
            "n": int(a.size),
            "median": float(med),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if a.size > 1 else float("nan"),
        }

    def fmt(s):
        if s is None:
            return "n=0 (no data)"
        return (f"n={s['n']} | median={s['median']:.2f} | "
                f"IQR=[{s['q1']:.2f}, {s['q3']:.2f}] (iqr={s['iqr']:.2f}) | "
                f"min/max={s['min']:.2f}/{s['max']:.2f} | "
                f"mean±std={s['mean']:.2f}±{s['std']:.2f}")

    print("\n" + "="*78)
    print(f"Summary statistics per bucket ({label})")
    print("="*78)
    for b in buckets:
        s_def = stats(results_default.get(b, []))
        s_cal = stats(results_calib.get(b, []))
        print(f"Bucket {b}:")
        print(f"  Default    : {fmt(s_def)}")
        print(f"  Calibrated : {fmt(s_cal)}")
    print("="*78 + "\n")


# -------------------------
# Ensure outputs exist
# -------------------------
for cycle in cycle_dirs:
    subprocess.run(["python", "InfoProcess.py", "--work-dir", cycle], check=True)

# -------------------------
# Collect per-cycle MAPE values per bucket
# -------------------------
results_default = {b: [] for b in BUCKETS}
results_calib   = {b: [] for b in BUCKETS}

for cycle in cycle_dirs:
    base        = os.path.basename(cycle)
    real_csv    = os.path.join(DATA_DIR, f"{base}.csv")
    default_csv = os.path.join(cycle, "simulation_info_default.csv")
    calib_csv   = os.path.join(cycle, "simulation_info.csv")

    if not (os.path.isfile(real_csv) and os.path.isfile(default_csv) and os.path.isfile(calib_csv)):
        continue

    df_r   = pd.read_csv(real_csv, delimiter=";")
    df_def = pd.read_csv(default_csv, delimiter=";")
    df_cal = pd.read_csv(calib_csv, delimiter=";")

    for df in (df_r, df_def, df_cal):
        df["exit_time"] = df["exit_time"].astype(float)

    phase = get_phase(base)

    dir_r   = df_r["exit_detector"].apply(get_direction)
    dir_def = df_def["exit_detector"].apply(get_direction)
    dir_cal = df_cal["exit_detector"].apply(get_direction)

    for direction in DIRECTIONS:
        bucket = (phase, direction)

        real_times = df_r.loc[dir_r == direction, "exit_time"].values
        def_times  = df_def.loc[dir_def == direction, "exit_time"].values
        cal_times  = df_cal.loc[dir_cal == direction, "exit_time"].values

        if len(real_times) == 0:
            continue

        T = int(np.ceil(max(real_times.max(), 1)))

        C_obs = cumulative_curve(real_times, T)
        C_def = cumulative_curve(def_times,  T)
        C_cal = cumulative_curve(cal_times,  T)

        results_default[bucket].append(mape_curve(C_obs, C_def))
        results_calib[bucket].append(mape_curve(C_obs, C_cal))



print_bucket_stats(results_default, results_calib, BUCKETS, label="MAPE (%)")

# -------------------------
# Boxplot Visualization
# -------------------------
fig, ax = plt.subplots(figsize=(7.8, 3.6))  # compact TRC-ish aspect

positions = []
data = []
labels = []

i = 0
for bucket in BUCKETS:
    positions.extend([i - 0.18, i + 0.18])
    data.extend([results_default[bucket], results_calib[bucket]])
    labels.append(f"({bucket[0]}, {bucket[1]})")
    i += 1

box = ax.boxplot(
    data,
    positions=positions,
    widths=0.30,
    patch_artist=True,
    showfliers=True,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgewidth=0.8)
)

# Map palette 
color_default = COLORS["Simulated"]  # orange
color_calib   = COLORS["Observed"]   # teal

for j, patch in enumerate(box["boxes"]):
    if j % 2 == 0:  # Default
        patch.set_facecolor(color_default)
        patch.set_edgecolor(color_default)
        patch.set_alpha(0.55)
    else:           # Calibrated
        patch.set_facecolor(color_calib)
        patch.set_edgecolor(color_calib)
        patch.set_alpha(0.55)

# Axes labels/ticks
ax.set_xticks(range(len(BUCKETS)))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Phase–Direction bucket index")

# Grid and spines
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
legend_elements = [
    Patch(facecolor=color_default, edgecolor=color_default, alpha=0.75, label="Default"),
    Patch(facecolor=color_calib,   edgecolor=color_calib,   alpha=0.75, label="Calibrated"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=False)

plt.tight_layout()
plt.savefig("box_plot_comparison_mape.pdf", bbox_inches="tight")
plt.show()

