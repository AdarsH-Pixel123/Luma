"""Visual comparison utilities for ISP pipeline evaluation.

Creates side-by-side image grids, metric tables, and bar-chart comparisons
so different pipeline variants can be compared at a glance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


# ---------------------------------------------------------------------------
# Comparison Grid
# ---------------------------------------------------------------------------

def create_comparison_grid(
    images: dict[str, np.ndarray],
    titles: Optional[dict[str, str]] = None,
    save_path: Optional[str] = None,
    crop_region: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Create a side-by-side comparison of pipeline outputs.

    Args:
        images: Mapping of pipeline names to output images ``(H, W, 3)``
            in [0, 1] float or [0, 255] uint8.  Example keys:
            ``'Input RAW'``, ``'Classical ISP'``, ``'Neural ISP'``,
            ``'Hybrid ISP'``, ``'Ground Truth'``.
        titles: Optional mapping from the same keys to custom display titles.
            When *None*, the dict keys themselves are used.
        save_path: If given, save the figure to this path.
        crop_region: Optional ``(y, x, h, w)`` rectangle.  When provided a
            second row of zoomed crop patches is appended below the full images.

    Returns:
        Rendered comparison grid as a uint8 numpy array ``(H, W, 3)``.
    """
    names = list(images.keys())
    n = len(names)
    if n == 0:
        raise ValueError("images dict must contain at least one entry")

    show_crops = crop_region is not None
    n_rows = 2 if show_crops else 1

    fig, axes = plt.subplots(
        n_rows, n, figsize=(4.5 * n, 4.5 * n_rows), squeeze=False,
    )

    for idx, name in enumerate(names):
        img = _prepare_for_display(images[name])
        title = (titles or {}).get(name, name)

        ax = axes[0, idx]
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

        if show_crops:
            assert crop_region is not None
            y, x, h, w = crop_region
            crop = img[y : y + h, x : x + w]
            ax_crop = axes[1, idx]
            ax_crop.imshow(crop)
            ax_crop.set_title(f"{title} (crop)", fontsize=9)
            ax_crop.axis("off")

            # Draw rectangle on full image
            rect = plt.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)

    plt.tight_layout()

    # Render to numpy array
    grid_array = _figure_to_array(fig)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return grid_array


# ---------------------------------------------------------------------------
# Metrics Table
# ---------------------------------------------------------------------------

def create_metrics_table(
    results: dict[str, dict[str, Optional[float]]],
    save_path: Optional[str] = None,
) -> str:
    """Create a formatted metrics comparison table.

    Args:
        results: Mapping of pipeline names to metric dicts, e.g.::

            {
                'Classical': {'psnr': 25.3, 'ssim': 0.82, 'delta_e': 4.1, 'lpips': 0.21},
                'Hybrid':    {'psnr': 28.1, 'ssim': 0.90, 'delta_e': 2.8, 'lpips': 0.13},
            }

        save_path: Base path (without extension).  Produces both a ``.txt``
            and a ``.png`` table visualisation.

    Returns:
        The formatted text table as a string.
    """
    pipelines = list(results.keys())
    if not pipelines:
        return ""

    # Collect all metric names preserving order
    metric_names: list[str] = []
    for m_dict in results.values():
        for k in m_dict:
            if k not in metric_names:
                metric_names.append(k)

    # ---------- text table ----------
    header = f"{'Pipeline':<20}" + "".join(f"{m.upper():>12}" for m in metric_names)
    sep = "-" * len(header)
    rows = [header, sep]
    for name in pipelines:
        vals = results[name]
        row = f"{name:<20}"
        for m in metric_names:
            v = vals.get(m)
            if v is None:
                row += f"{'N/A':>12}"
            else:
                row += f"{v:>12.4f}"
        rows.append(row)
    table_str = "\n".join(rows)

    if save_path is not None:
        txt_path = save_path if save_path.endswith(".txt") else save_path + ".txt"
        with open(txt_path, "w") as f:
            f.write(table_str + "\n")

    # ---------- matplotlib figure ----------
    fig, ax = plt.subplots(figsize=(2.0 + 2.0 * len(metric_names), 0.8 + 0.5 * len(pipelines)))
    ax.axis("off")

    col_labels = ["Pipeline"] + [m.upper() for m in metric_names]
    cell_text = []
    for name in pipelines:
        vals = results[name]
        row = [name]
        for m in metric_names:
            v = vals.get(m)
            row.append("N/A" if v is None else f"{v:.4f}")
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Bold header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")

    # Highlight best values (highest PSNR/SSIM, lowest Delta E/LPIPS)
    _highlight_best_cells(table, results, pipelines, metric_names)

    plt.tight_layout()
    if save_path is not None:
        png_path = save_path.rsplit(".", 1)[0] + ".png" if "." in save_path else save_path + ".png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return table_str


def _highlight_best_cells(
    table: plt.table,
    results: dict[str, dict[str, Optional[float]]],
    pipelines: list[str],
    metric_names: list[str],
) -> None:
    """Highlight best metric values in green."""
    higher_is_better = {"psnr", "ssim"}

    for col_idx, m in enumerate(metric_names, start=1):
        values = []
        for name in pipelines:
            v = results[name].get(m)
            values.append(v if v is not None else None)

        numeric = [v for v in values if v is not None]
        if not numeric:
            continue

        if m in higher_is_better:
            best = max(numeric)
        else:
            best = min(numeric)

        for row_idx, v in enumerate(values, start=1):
            if v is not None and abs(v - best) < 1e-8:
                table[row_idx, col_idx].set_facecolor("#c6efce")


# ---------------------------------------------------------------------------
# Bar Charts
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    results: dict[str, dict[str, Optional[float]]],
    save_path: Optional[str] = None,
) -> Figure:
    """Create bar charts comparing metrics across pipelines.

    One subplot per metric (PSNR, SSIM, Delta E, LPIPS).

    Args:
        results: Same format as :func:`create_metrics_table`.
        save_path: Optional path to save the figure.

    Returns:
        The matplotlib :class:`Figure` object.
    """
    pipelines = list(results.keys())
    if not pipelines:
        raise ValueError("results dict must contain at least one entry")

    # Collect metric names
    metric_names: list[str] = []
    for m_dict in results.values():
        for k in m_dict:
            if k not in metric_names:
                metric_names.append(k)

    n_metrics = len(metric_names)
    cols = min(n_metrics, 4)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    colours = plt.cm.Set2(np.linspace(0, 1, len(pipelines)))  # type: ignore[attr-defined]

    higher_is_better = {"psnr", "ssim"}

    for idx, metric in enumerate(metric_names):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        vals = []
        labels = []
        for name in pipelines:
            v = results[name].get(metric)
            if v is not None:
                vals.append(v)
                labels.append(name)

        if not vals:
            ax.set_visible(False)
            continue

        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=colours[: len(labels)], edgecolor="grey", linewidth=0.5)

        # Highlight best bar
        if metric in higher_is_better:
            best_idx = int(np.argmax(vals))
        else:
            best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor("green")
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric.upper(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")

        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Hide unused subplots
    for idx in range(n_metrics, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.suptitle("Pipeline Metric Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_for_display(img: np.ndarray) -> np.ndarray:
    """Normalise an image to uint8 [0, 255] for display."""
    if img.dtype == np.uint8:
        return img
    img = np.clip(img.astype(np.float64), 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _figure_to_array(fig: Figure) -> np.ndarray:
    """Render a matplotlib figure to a numpy RGB array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)
    return arr[:, :, :3].copy()
