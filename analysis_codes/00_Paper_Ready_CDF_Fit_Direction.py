import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import ks_2samp

# -----------------------
# Settings
# -----------------------
DATA_DIR   = "Data"
CYCLE_GLOB = "Info_Cycle_*"
cycle_dirs = sorted(glob.glob(CYCLE_GLOB))

phase_df = pd.read_csv("Data/TrafficLightCycles.csv", delimiter=';')

PHASE_FALLBACK = [(0.0, 60.0), (60.0, 90.0)]  # same as optimization
DEFAULT_LENS = [60.0, 30.0]                   # same as optimization

# Publication-friendly styling (TRC-ish)
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

# Colorblind-safe palette (Okabe–Ito-like)
COLORS = {"Observed": "#00C489", "Simulated": "#E0370D"}

PHASE_NAMES = {0: r"$ 1 $", 1: r"$ 2 $"}
DIR_NAMES   = {0: r"$ 1 $", 1: r"$ 2 $"}
SRC_NAMES   = {"Real": "Observed", "Sim": "Simulated"}

def beautify_axis(ax):
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.tick_params(direction='out', length=5, width=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

def normalized_l1_percent(Cr: np.ndarray, Cs: np.ndarray) -> float:
    mae = np.mean(np.abs(Cr - Cs))
    N = max(float(Cr[-1]), 1.0)  # total observed vehicles
    return 100.0 * mae / N


def round_significant(x, sig=2):
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

def get_phase_bounds_for_cycle(base_name: str):
    try:
        idx = int(base_name.split("_")[-1]) - 1
        if 0 <= idx < len(phase_df):
            row = phase_df.iloc[idx]
            return [(float(row.Start), float(row.Break)),
                    (float(row.Continuation), float(row.Stop))]
    except Exception:
        pass
    return PHASE_FALLBACK

def cumulative_curve(times: np.ndarray, T: int) -> np.ndarray:
    """C(t) on grid t=0..T (inclusive), using the same searchsorted logic as optimization."""
    grid = np.arange(0, T + 1)
    if times.size == 0:
        return np.zeros_like(grid, dtype=int)
    times = np.sort(times.astype(float))
    return np.searchsorted(times, grid, side='right')

def l1_mae_counts(Cr: np.ndarray, Cs: np.ndarray) -> float:
    """Matches optimization: mae = mean(|Cr - Cs|). Units: vehicles."""
    return float(np.mean(np.abs(Cr - Cs)))

def ecdf_from_cumulative(C: np.ndarray) -> np.ndarray:
    """Convert cumulative counts to an empirical CDF in [0,1]."""
    N = max(float(C[-1]), 1.0)
    return C.astype(float) / N

def max_dev_veh(Cr: np.ndarray, Cs: np.ndarray) -> float:
    """Maximum absolute deviation between cumulative curves. Units: vehicles."""
    return float(np.max(np.abs(Cr - Cs)))

def nABC_percent_area(Cr: np.ndarray, Cs: np.ndarray) -> float:
    """
    Normalized area between cumulative curves, in percent.
    Normalization: divide by area under observed cumulative curve.
    """
    abs_diff = np.abs(Cr - Cs).astype(float)
    denom = float(np.sum(Cr))  # area under observed cumulative curve
    denom = max(denom, 1.0)
    return 100.0 * float(np.sum(abs_diff)) / denom




# -----------------------
# Pool phase-relative times exactly like optimization
# real_times[p][d], sim_times[p][d]
# -----------------------
real_times = [[[] for _ in range(2)] for _ in range(2)]
sim_times  = [[[] for _ in range(2)] for _ in range(2)]
phase_lens_seen = [[], []]

for cycle in cycle_dirs:
    base = os.path.basename(cycle)
    phases = get_phase_bounds_for_cycle(base)

    real_csv = os.path.join(DATA_DIR, f"{base}.csv")
    sim_csv  = os.path.join(cycle, "simulation_info.csv")
    if not (os.path.isfile(real_csv) and os.path.isfile(sim_csv)):
        continue

    try:
        df_r = pd.read_csv(real_csv, delimiter=';')
        df_s = pd.read_csv(sim_csv,  delimiter=';')
    except Exception:
        continue

    # ensure types
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
            rt = subr[subr.exit_detector.str.startswith(prefix)].exit_time.to_numpy() - t0
            st = subs[subs.exit_detector.str.startswith(prefix)].exit_time.to_numpy() - t0
            real_times[p][d].extend(rt.tolist())
            sim_times[p][d].extend(st.tolist())

# phase lengths consistent with optimization (max seen per phase)
T_phase = [
    int(np.max(phase_lens_seen[0])) if phase_lens_seen[0] else int(DEFAULT_LENS[0]),
    int(np.max(phase_lens_seen[1])) if phase_lens_seen[1] else int(DEFAULT_LENS[1]),
]
T_phase = [max(1, T) for T in T_phase]

# -----------------------
# Plot (2x2) : K-S statistic + L1 (MAE) (no p-value), no global title
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), sharey=True)

for p in (0, 1):
    for d in (0, 1):
        ax = axes[p, d]
        rt = np.asarray(real_times[p][d], dtype=float)
        st = np.asarray(sim_times[p][d], dtype=float)

        T = T_phase[p]
        grid = np.arange(0, T + 1)

        Cr = cumulative_curve(rt, T)
        Cs = cumulative_curve(st, T)

        ax.plot(grid, Cr, label="Observed",  color=COLORS["Observed"])
        ax.plot(grid, Cs, label="Simulated", color=COLORS["Simulated"])

        C = len(cycle_dirs)  # or better: count only cycles that actually contributed to this (p,d) pool
        Dveh = max_dev_veh(Cr, Cs)
        Dveh_per_cycle = Dveh / max(C, 1)

        nabc = nABC_percent_area(Cr, Cs)
        nabc = round_significant(nabc, sig=2)

        ax.set_title(
            f"(p,d)  = ({PHASE_NAMES[p]},{DIR_NAMES[d]}) \n"
            rf"$\mathcal{{D}}$ = {Dveh_per_cycle:.0f} veh/cycle" "\n" 
            f"nABC = {nabc}%"
        )

        print(
            f"(p,d)  = ({PHASE_NAMES[p]},{DIR_NAMES[d]}) \n"
            rf"$\mathcal{{D}}$ = {Dveh_per_cycle:.0f} veh/cycle" "\n" 
            f"nABC = {nabc}%"
        )

        ax.set_xlabel("Phase-relative time (s)")
        ax.set_ylabel("Cumulative vehicle count")
        beautify_axis(ax)
        ax.legend(frameon=False, loc="best")

fig.tight_layout()
fig.savefig("cumulative_exit_counts_KS_L1.pdf", bbox_inches="tight")
fig.savefig("cumulative_exit_counts_KS_L1.png", bbox_inches="tight")
plt.show()

print("Saved: cumulative_exit_counts_KS_L1.pdf/png")



