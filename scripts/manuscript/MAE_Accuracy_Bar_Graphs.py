from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("outputs/plots/bar_graphs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Manuscript bar plots are intentionally horizontal so split/case labels remain
# readable in compact RSC/Digital Discovery figure panels.
def horizontal_figsize(n_categories: int, base_width: float = 7.2) -> tuple[float, float]:
    """Return a readable figure size for horizontal category labels."""
    return (base_width, max(3.2, 1.0 + 0.72 * n_categories))


def save_current_figure(filename: str) -> None:
    """Save the active matplotlib figure as PNG and PDF."""
    png_path = OUT_DIR / f"{filename}.png"
    pdf_path = OUT_DIR / f"{filename}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_grouped_bars(
    df: pd.DataFrame,
    x_col: str,
    group_col: str,
    y_col: str,
    yerr_col: str | None,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Create a grouped horizontal bar chart."""
    y_labels = list(df[x_col].drop_duplicates())
    groups = list(df[group_col].drop_duplicates())

    y = np.arange(len(y_labels))
    height = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=horizontal_figsize(len(y_labels)))

    for i, group in enumerate(groups):
        sub = df[df[group_col] == group].set_index(x_col).reindex(y_labels)
        offsets = y - 0.4 + height / 2 + i * height

        values = sub[y_col].to_numpy()
        xerr = sub[yerr_col].to_numpy() if yerr_col else None

        ax.barh(
            offsets,
            values,
            height,
            label=group,
            xerr=xerr,
            capsize=3 if yerr_col else 0,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(ylabel)
    ax.set_ylabel(x_col)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.invert_yaxis()
    ax.margins(x=0.08)

    save_current_figure(filename)


def plot_simple_bars(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    yerr_col: str | None,
    ylabel: str,
    title: str,
    filename: str,
    zero_reference: bool = False,
) -> None:
    """Create a simple one-series horizontal bar chart."""
    fig, ax = plt.subplots(figsize=horizontal_figsize(len(df), base_width=6.2))

    y = np.arange(len(df))
    xerr = df[yerr_col].to_numpy() if yerr_col else None

    ax.barh(
        y,
        df[y_col].to_numpy(),
        xerr=xerr,
        capsize=3 if yerr_col else 0,
    )

    if zero_reference:
        ax.axvline(0, color="0.25", linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(df[x_col].to_list())
    ax.set_xlabel(ylabel)
    ax.set_ylabel(x_col)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.margins(x=0.08)

    save_current_figure(filename)


def plot_paired_bars(
    df: pd.DataFrame,
    x_col: str,
    y1_col: str,
    y2_col: str,
    y1_label: str,
    y2_label: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Create paired horizontal bars for two metrics over the same categories."""
    fig, ax = plt.subplots(figsize=horizontal_figsize(len(df)))

    y = np.arange(len(df))
    height = 0.36

    ax.barh(y - height / 2, df[y1_col].to_numpy(), height, label=y1_label)
    ax.barh(y + height / 2, df[y2_col].to_numpy(), height, label=y2_label)

    ax.set_yticks(y)
    ax.set_yticklabels(df[x_col].to_list())
    ax.set_xlabel(ylabel)
    ax.set_ylabel(x_col)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.invert_yaxis()
    ax.margins(x=0.08)

    save_current_figure(filename)


def plot_classification_metrics() -> None:
    """Create grouped bar chart for PLQY classification metrics."""
    classification = pd.DataFrame(
        {
            "Split": ["Random", "Molecule", "Scaffold"],
            "Accuracy": [0.8770, 0.8015, 0.7316],
            "Balanced accuracy": [0.8775, 0.8004, 0.7268],
            "Macro-F1": [0.8769, 0.7994, 0.7252],
            "Weighted-F1": [0.8769, 0.7997, 0.7269],
        }
    )

    long_df = classification.melt(
        id_vars="Split",
        var_name="Metric",
        value_name="Score",
    )

    plot_grouped_bars(
        df=long_df,
        x_col="Split",
        group_col="Metric",
        y_col="Score",
        yerr_col=None,
        ylabel="Classification score",
        title="Binary PLQY Classification by Split",
        filename="fig4_plqy_classification_metrics",
    )


def main() -> None:
    # ------------------------------------------------------------
    # 1. Conventional benchmark data
    # ------------------------------------------------------------
    conventional = pd.DataFrame(
        [
            ["Absorption", "Random", "RF", 12.50, 0.14, 26.83, 0.9360],
            ["Absorption", "Molecule", "RF", 20.74, 0.45, 37.85, 0.8675],
            ["Absorption", "Scaffold", "RF", 32.14, 2.83, 49.67, 0.7625],
            ["Emission", "Random", "ExtraTrees", 14.62, 0.19, 29.09, 0.9030],
            ["Emission", "Molecule", "RF", 24.85, 0.66, 39.06, 0.8250],
            ["Emission", "Scaffold", "RF", 37.66, 2.13, 52.98, 0.6622],
            ["Quantum yield", "Random", "ExtraTrees", 0.0961, 0.0018, 0.1610, 0.7358],
            ["Quantum yield", "Molecule", "ExtraTrees", 0.1473, 0.0049, 0.2205, 0.5046],
            ["Quantum yield", "Scaffold", "RF", 0.1961, 0.0055, 0.2538, 0.3586],
        ],
        columns=["Target", "Split", "Best model", "MAE", "MAE std", "RMSE", "R2"],
    )

    stokes = pd.DataFrame(
        [
            ["Stokes shift", "Random", "ExtraTrees", 11.05, 0.11, 21.32, 0.8431],
            ["Stokes shift", "Molecule", "RF", 17.85, 0.43, 29.34, 0.7057],
            ["Stokes shift", "Scaffold", "RF", 23.00, 2.15, 34.20, 0.5986],
        ],
        columns=["Target", "Split", "Best model", "MAE", "MAE std", "RMSE", "R2"],
    )

    wavelength_df = pd.concat(
        [
            conventional[conventional["Target"].isin(["Absorption", "Emission"])],
            stokes,
        ],
        ignore_index=True,
    )

    plot_grouped_bars(
        df=wavelength_df,
        x_col="Split",
        group_col="Target",
        y_col="MAE",
        yerr_col="MAE std",
        ylabel="MAE (nm)",
        title="Conventional Wavelength and Stokes-Shift MAE by Split",
        filename="fig1_conventional_wavelength_stokes_mae",
    )

    qy_df = conventional[conventional["Target"] == "Quantum yield"].copy()

    plot_simple_bars(
        df=qy_df,
        x_col="Split",
        y_col="MAE",
        yerr_col="MAE std",
        ylabel="MAE",
        title="Quantum-Yield Regression MAE by Split",
        filename="fig2_quantum_yield_regression_mae",
    )

    # ------------------------------------------------------------
    # 2. Hybrid benchmark data
    # ------------------------------------------------------------
    hybrid = pd.DataFrame(
        [
            ["Absorption", "Molecule", 8470, "RF", 22.6108, 22.1534, 36.2029, 0.8789, 0.4573],
            ["Emission", "Molecule", 8225, "RF", 27.0531, 26.6456, 39.0584, 0.8203, 0.4075],
            ["Quantum yield", "Molecule", 5783, "RF", 0.1650, 0.1574, 0.2166, 0.5137, 0.0076],
            ["Absorption", "Scaffold", 7814, "RF", 31.2901, 35.7370, 51.1127, 0.7373, -4.4468],
            ["Emission", "Scaffold", 7366, "RF", 36.3510, 34.5058, 47.9038, 0.7493, 1.8452],
            ["Quantum yield", "Scaffold", 7138, "RF", 0.2016, 0.1965, 0.2614, 0.3292, 0.0051],
        ],
        columns=[
            "Target",
            "Split",
            "Final-test rows",
            "Best base model",
            "Best base MAE",
            "Hybrid MAE",
            "Hybrid RMSE",
            "Hybrid R2",
            "MAE improvement",
        ],
    )

    hybrid["Case"] = hybrid["Target"] + "\n" + hybrid["Split"]

    hybrid_wavelength = hybrid[hybrid["Target"].isin(["Absorption", "Emission"])].copy()

    plot_paired_bars(
        df=hybrid_wavelength,
        x_col="Case",
        y1_col="Best base MAE",
        y2_col="Hybrid MAE",
        y1_label="Best base model",
        y2_label="Hybrid model",
        ylabel="MAE (nm)",
        title="Hybrid Evaluation for Wavelength Prediction",
        filename="fig3a_hybrid_wavelength_base_vs_hybrid_mae",
    )

    hybrid_qy = hybrid[hybrid["Target"] == "Quantum yield"].copy()

    plot_paired_bars(
        df=hybrid_qy,
        x_col="Case",
        y1_col="Best base MAE",
        y2_col="Hybrid MAE",
        y1_label="Best base model",
        y2_label="Hybrid model",
        ylabel="MAE",
        title="Hybrid Evaluation for Quantum-Yield Prediction",
        filename="fig3b_hybrid_qy_base_vs_hybrid_mae",
    )

    # Optional improvement-only plot for wavelength targets.
    plot_simple_bars(
        df=hybrid_wavelength,
        x_col="Case",
        y_col="MAE improvement",
        yerr_col=None,
        ylabel="MAE improvement (nm)",
        title="Hybrid MAE Improvement for Wavelength Prediction",
        filename="fig3c_hybrid_wavelength_mae_improvement",
        zero_reference=True,
    )

    # Optional improvement-only plot for quantum yield.
    plot_simple_bars(
        df=hybrid_qy,
        x_col="Case",
        y_col="MAE improvement",
        yerr_col=None,
        ylabel="MAE improvement",
        title="Hybrid MAE Improvement for Quantum-Yield Prediction",
        filename="fig3d_hybrid_qy_mae_improvement",
        zero_reference=True,
    )

    # ------------------------------------------------------------
    # 3. PLQY classification metrics
    # ------------------------------------------------------------
    plot_classification_metrics()


if __name__ == "__main__":
    main()
