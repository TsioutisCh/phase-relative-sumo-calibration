import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.ticker import MaxNLocator

def apply_best_and_plot(history_csv='optimization_history_nevergrad_20_30_25_25.csv', xml_file='osm.type.xml',
                        objective_threshold=5000, out_prefix='opt_history'):
    """
    Transportation Research Part C-style plotting + XML update.

    - Reads optimization history CSV (must include 'objective')
    - Plots RAW objective values vs iteration (no rolling average), for objectives < threshold
    - Plots sorted objective values (descending) for objectives < threshold
    - Uses larger fonts/line widths and publication-friendly styling
    - Saves figures as vector PDF + high-res PNG
    - Updates xml_file with best parameters (global min objective across ALL rows)
    """
    # -----------------------
    # Load history
    # -----------------------
    df = pd.read_csv(history_csv)
    if 'objective' not in df.columns:
        raise ValueError("CSV must contain an 'objective' column")

    # Filter objectives for plotting (keep raw iterations)
    df_plot = df[df['objective'] < objective_threshold].copy()
    if df_plot.empty:
        print(f"No objective values under threshold ({objective_threshold}) to plot.")
        return

    # Iteration number (1-based)
    df_plot['iteration'] = df_plot.index + 1

    # -----------------------
    # Plot styling (journal-ish)
    # -----------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })

    def _beautify_axis(ax):
        ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.tick_params(direction='out', length=5, width=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    # -----------------------
    # Figure 1: Objective vs Iteration (RAW)
    # -----------------------
    fig1, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(
        df_plot['iteration'],
        df_plot['objective'],
        linestyle='-',
        linewidth=1.8,
        marker='o',
        markersize=4.0
    )
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(r'Objective value $ z(\theta) $')
    #ax1.set_title('Objective evolution (raw iterations)')
    _beautify_axis(ax1)
    fig1.tight_layout()

    fig1.savefig(f"{out_prefix}_objective_vs_iteration.pdf", bbox_inches='tight')
    fig1.savefig(f"{out_prefix}_objective_vs_iteration.png", bbox_inches='tight')

    plt.show()
    # -----------------------
    # Find best row overall (global min objective) and update XML
    # -----------------------
    best_row = df.loc[df['objective'].idxmin()]
    params = best_row.drop('objective').to_dict()

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for v in root.findall(".//vType"):
        for key, val in params.items():
            val_str = str(val)
            if "_" in key:
                vt, param = key.split("_", 1)
                if v.get("id") == vt:
                    v.set(param, val_str)
            else:
                v.set(key, val_str)

    tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"Updated {xml_file} with best parameters (objective={best_row['objective']:.4f}).")
    print(f"Saved figures: {out_prefix}_objective_vs_iteration.(pdf/png) and {out_prefix}_objective_sorted_desc.(pdf/png)")

# Example usage:
if __name__ == '__main__':
    apply_best_and_plot()
