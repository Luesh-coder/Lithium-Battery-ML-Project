"""Matplotlib figures for the results written by the training scripts.

Color means the data split in every figure: train is gray context, and the primary and
secondary test sets take the first two categorical slots (blue, orange). That pair passes
the dataviz palette validator in light mode on all pairs (CVD delta E 24.7).

Call `use_style()` once before plotting. The functions build `matplotlib.figure.Figure`
objects directly, so they don't depend on pyplot or a GUI backend.
"""

import math

import matplotlib as mpl
from matplotlib import patheffects
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

from battery_ml.data import SPLITS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SPLIT_COLORS = {"train": INK_MUTED, "primary_test": "#2a78d6", "secondary_test": "#eb6834"}
SPLIT_LABELS = {"train": "Train", "primary_test": "Primary test", "secondary_test": "Secondary test"}
MODEL_LABELS = {
    "elastic_net": "Elastic net",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}
METRIC_LABELS = {"rmse": "RMSE (cycles)", "mape": "MAPE (%)"}

# Elastic net at 100 cycles from [1], Table 1, per split (train, primary, secondary).
# The primary test set includes b2c1, the cell that reaches 80% capacity after 148 cycles;
# without it [1] reports primary RMSE 138 / 86 / 100 and MAPE 13.2 / 10.1 / 7.5 %.
SEVERSON_TABLE1 = {
    "variance": {"rmse": (103, 138, 196), "mape": (14.1, 14.7, 11.4)},
    "discharge": {"rmse": (76, 91, 173), "mape": (9.8, 13.0, 8.6)},
    "full": {"rmse": (51, 118, 214), "mape": (5.6, 14.1, 10.7)},
}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "savefig.dpi": 200,
}

BAR_MAX_WIDTH_PT = 18  # about 24 px on screen
BAR_GAP_PT = 1.5  # surface gap between touching bars
BAR_RADIUS_PT = 3  # rounded data end; the baseline end stays square
MARKER_SIZE = 40  # scatter area in pt^2 (about 9 px across)


def use_style():
    """Apply the shared chart style globally (system sans, hairline grid, light surface)."""
    mpl.rcParams.update(STYLE)


def save_figure(fig, stem, formats=("png", "pdf")):
    """Write `fig` to stem.png and stem.pdf (PDF for the LaTeX report)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), bbox_inches="tight")


def _points_to_data(ax, points):
    """(dx, dy) in data units of a length in points, for the axes as currently laid out."""
    bbox = ax.get_window_extent()
    px = points * ax.figure.dpi / 72
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return px * (x1 - x0) / bbox.width, px * (y1 - y0) / bbox.height


def _bar(ax, x0, x1, height, color, radius):
    """A bar from y=0 with rounded top corners. `radius` is (rx, ry) in data units."""
    rx, ry = min(radius[0], (x1 - x0) / 2), min(radius[1], height)
    k = 0.5523  # cubic Bezier control distance for a quarter ellipse
    verts = [
        (x0, 0), (x0, height - ry),
        (x0, height - ry + k * ry), (x0 + rx - k * rx, height), (x0 + rx, height),
        (x1 - rx, height),
        (x1 - rx + k * rx, height), (x1, height - ry + k * ry), (x1, height - ry),
        (x1, 0), (x0, 0),
    ]
    codes = [Path.MOVETO, Path.LINETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3
    codes += [Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=2))


def _split_legend_handles(marker=None):
    if marker:
        return [
            Line2D([], [], ls="", marker=marker, markersize=7, markerfacecolor=SPLIT_COLORS[s],
                   markeredgecolor=SURFACE, label=SPLIT_LABELS[s])
            for s in SPLITS
        ]
    return [Patch(facecolor=SPLIT_COLORS[s], label=SPLIT_LABELS[s]) for s in SPLITS]


def plot_metric_bars(scores, metric, title, reference=None):
    """Grouped bars of one metric: a panel per feature set, models on x, one bar per split.

    `scores` is the long format written by the training scripts (window, model, split,
    rmse, mape) for a single window, plus a `feature_set` column. `reference` maps
    feature set -> metric -> per-split values for the elastic net (e.g. SEVERSON_TABLE1);
    each value is drawn as a black tick across the matching bar.
    """
    feature_sets = list(dict.fromkeys(scores["feature_set"]))
    models = [m for m in MODEL_LABELS if m in set(scores["model"])]
    values = scores.set_index(["feature_set", "model", "split"])[metric]

    top = values.max()
    if reference:
        top = max(top, *(v for fs in feature_sets for v in reference[fs][metric]))

    fig = Figure(figsize=(3.2 * len(feature_sets), 3.7), layout="constrained")
    axes = fig.subplots(1, len(feature_sets), sharey=True, squeeze=False)[0]
    for ax, fs in zip(axes, feature_sets):
        ax.set_title(f"{fs.capitalize()} features")
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_ylim(0, 1.08 * top)
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m].replace(" ", "\n") for m in models])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(METRIC_LABELS[metric])

    handles = _split_legend_handles()
    if reference:
        handles.append(Line2D([], [], color=INK, lw=1.6, label="Severson et al. (2019), elastic net"))
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    fig.suptitle(title, fontsize=11, fontweight="semibold")

    # Bar geometry is set in points, so freeze the layout before converting to data units.
    fig.draw_without_rendering()
    fig.set_layout_engine("none")
    for ax, fs in zip(axes, feature_sets):
        max_width, _ = _points_to_data(ax, BAR_MAX_WIDTH_PT)
        gap, _ = _points_to_data(ax, BAR_GAP_PT)
        width = min(max_width, (0.8 - (len(SPLITS) - 1) * gap) / len(SPLITS))
        group = len(SPLITS) * width + (len(SPLITS) - 1) * gap
        radius = _points_to_data(ax, BAR_RADIUS_PT)
        tick_pad, _ = _points_to_data(ax, 3)

        for i, model in enumerate(models):
            for j, split in enumerate(SPLITS):
                x0 = i - group / 2 + j * (width + gap)
                value = values[fs, model, split]
                _bar(ax, x0, x0 + width, value, SPLIT_COLORS[split], radius)
                if value < 0.03 * top:  # too short to see: print the value instead
                    ax.annotate(f"{value:.1f}", (x0 + width / 2, value), xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
                if reference and model == "elastic_net":
                    ax.plot(
                        [x0 - tick_pad, x0 + width + tick_pad], [reference[fs][metric][j]] * 2,
                        color=INK, lw=1.6, solid_capstyle="butt", zorder=3,
                        path_effects=[patheffects.withStroke(linewidth=3.6, foreground=SURFACE)],
                    )
    return fig


def plot_predicted_vs_observed(predictions, scores, title, outlier_ratio=2.5):
    """Predicted vs observed cycle life per cell: rows are models, columns feature sets.

    `predictions` is the per-cell output of a training script (indexed by cell id) plus a
    `feature_set` column; `scores` supplies the test RMSE printed in each panel. Cells
    predicted more than `outlier_ratio` times too high or too low are labeled by id.
    """
    feature_sets = list(dict.fromkeys(predictions["feature_set"]))
    models = [m for m in MODEL_LABELS if m in set(predictions["model"])]
    rmse = scores.set_index(["feature_set", "model", "split"])["rmse"]
    lim = 500 * math.ceil(max(predictions["cycle_life"].max(), predictions["predicted"].max()) / 500)

    fig = Figure(figsize=(3.0 * len(feature_sets), 3.0 * len(models) + 0.7), layout="constrained")
    axes = fig.subplots(len(models), len(feature_sets), sharex=True, sharey=True, squeeze=False)
    for r, model in enumerate(models):
        for c, fs in enumerate(feature_sets):
            ax = axes[r, c]
            cell = predictions[(predictions["model"] == model) & (predictions["feature_set"] == fs)]
            ax.plot([0, lim], [0, lim], color=INK_MUTED, lw=0.8, zorder=1)
            for z, split in enumerate(SPLITS):
                d = cell[cell["split"] == split]
                # clip_on=False keeps dots sitting on an axis (predicted ~0) whole
                ax.scatter(d["cycle_life"], d["predicted"], s=MARKER_SIZE, color=SPLIT_COLORS[split],
                           edgecolors=SURFACE, linewidths=1.1, zorder=2 + z, clip_on=False)

            ratio = cell["predicted"] / cell["cycle_life"]
            for cell_id, row in cell[(ratio > outlier_ratio) | (ratio < 1 / outlier_ratio)].iterrows():
                ax.annotate(cell_id, (row["cycle_life"], row["predicted"]), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7.5, color=INK_SECONDARY, zorder=6,
                            path_effects=[patheffects.withStroke(linewidth=2.5, foreground=SURFACE)])

            ax.text(0.04, 0.96, f"Test RMSE: primary {rmse[fs, model, 'primary_test']:.0f}, "
                    f"secondary {rmse[fs, model, 'secondary_test']:.0f}",
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK_SECONDARY)
            ax.set(xlim=(0, lim), ylim=(0, lim), aspect="equal")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(MultipleLocator(1000))
                axis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            if r == 0:
                ax.set_title(f"{fs.capitalize()} features")
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\npredicted cycle life")
            if r == len(models) - 1:
                ax.set_xlabel("Observed cycle life")

    handles = _split_legend_handles(marker="o") + [Line2D([], [], color=INK_MUTED, lw=0.8, label="Perfect prediction")]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    fig.suptitle(title, fontsize=11, fontweight="semibold")
    return fig
