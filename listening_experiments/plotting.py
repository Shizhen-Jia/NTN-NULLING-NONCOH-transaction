"""Paper-ready Fig. 10/11 panels from archived estimates, with exact CI endpoints."""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


COLORS = ("#176AA6", "#D47714", "#20846C", "#8958A5")
MARKERS = ("o", "s", "^", "D")


def _series(ax, rows, metric, color, marker, label):
    x = np.array([r["L"] for r in rows])
    y = np.array([r[metric] for r in rows])
    ax.plot(x, y, color=color, marker=marker, markersize=5,
            markerfacecolor="white", linewidth=1.6, label=label)
    # Unlike yerr, these also represent percentile intervals that do not
    # happen to bracket the point estimate, without altering their endpoints.
    valid = [r.get(metric + "_lower") is not None for r in rows]
    if any(valid):
        low = np.array([r.get(metric + "_lower", np.nan) for r in rows], dtype=float)
        high = np.array([r.get(metric + "_upper", np.nan) for r in rows], dtype=float)
        ax.vlines(x[valid], low[valid], high[valid], color=color, linewidth=1)
        ax.plot(x[valid], low[valid], "_", color=color, markersize=6)
        ax.plot(x[valid], high[valid], "_", color=color, markersize=6)


def plot_figures(analysis):
    """Accept analyze_run output or a saved analysis.json path. Return both figures."""
    if isinstance(analysis, (str, Path)):
        analysis = json.loads(Path(analysis).read_text())
    directory = Path(analysis["directory"])
    rows, coverage = analysis["summary"], analysis["coverage"]
    probabilities = sorted({r["p"] for r in rows})
    windows = sorted({r["L"] for r in rows})
    n = np.asarray(analysis["protected_counts"])
    epsilon = analysis["analysis_config"]["coverage_epsilon"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180}):
        fig10, axes10 = plt.subplots(1, 2, figsize=(11.6, 4.6))
        fig11, axes11 = plt.subplots(1, 2, figsize=(11.6, 4.6))
        for k, p in enumerate(probabilities):
            color, marker = COLORS[k % len(COLORS)], MARKERS[k % len(MARKERS)]
            subset = sorted([r for r in rows if r["p"] == p], key=lambda r: r["L"])
            c = sorted([r for r in coverage if r["p"] == p], key=lambda r: r["L"])
            renamed = [dict(L=r["L"], q=r["q_miss_mc"], q_lower=r["lower"], q_upper=r["upper"]) for r in c]
            _series(axes10[0], renamed, "q", color, marker, f"p = {p:g}")
            dense = np.arange(1, max(windows) + 1)
            bound = np.minimum(1, n[:, None] * (1 - p) ** dense[None, :]).mean(axis=0)
            axes10[0].plot(dense, bound, "--", color=color, linewidth=1.1, alpha=0.8)
            _series(axes10[1], subset, "q_out", color, marker, f"p = {p:g}")
            _series(axes11[0], subset, "r_eff", color, marker, f"p = {p:g}")
            _series(axes11[1], subset, "q_data", color, marker, f"p = {p:g}")
        axes10[0].axhline(epsilon, color="0.55", linestyle=":", linewidth=1)
        if 0.1 in probabilities:
            sufficient = math.ceil(math.log(float(n.max()) / epsilon) / -math.log1p(-0.1))
            if min(windows) <= sufficient <= max(windows):
                axes10[0].axvline(sufficient, color="0.65", linewidth=0.8, linestyle=":")
                axes10[0].text(sufficient, 0.98, f" L = {sufficient}", rotation=90, va="top", fontsize=8)
        reference = sorted([r for r in rows if r["p"] == probabilities[0]], key=lambda r: r["L"])
        for ax in (axes10[1], axes11[1]):
            ax.axhline(reference[0]["q_mrt"], color="0.2", linestyle="--", linewidth=1.3, label="MRT")
        axes11[0].axhline(reference[0]["r_mrt"], color="0.2", linestyle="--", linewidth=1.3, label="MRT without sensing")
        axes11[0].plot(windows, [r["r_mrt_scheduled"] for r in reference], color="0.5", linestyle=":",
                       linewidth=1.6, label="MRT, same sensing schedule")
        axes10[0].set_title("(a) Activity coverage")
        axes10[1].set_title("(b) Final-beam protection")
        axes11[0].set_title("(a) Effective TN service")
        axes11[1].set_title("(b) Risk during data transmission")
        axes10[0].set_ylabel("Activity coverage failure probability")
        axes10[1].set_ylabel("Final-phase INR exceedance probability")
        axes11[0].set_ylabel("Mean effective TN spectral efficiency (bit/s/Hz)")
        axes11[1].set_ylabel("INR exceedance fraction during data transmission")
        for ax in (axes10[0], axes10[1], axes11[1]):
            ax.set_ylim(-0.015, 1.03)
        axes11[0].set_ylim(bottom=0)
        for ax in (*axes10, *axes11):
            ax.set_xlabel(r"Number of listening windows $L_{\rm s}$")
            ax.set_xlim(min(windows) - 0.6, max(windows) + 0.8)
            if max(windows) <= 8:
                ax.set_xticks(windows)
            elif max(windows) <= 64:
                ax.set_xticks(sorted({min(windows), max(windows)} |
                                     {x for x in (8, 16, 32, 48, 64) if min(windows) <= x <= max(windows)}))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.grid(axis="y", alpha=0.22)
            ax.legend(fontsize=8, frameon=False)
        handles, labels = axes10[0].get_legend_handles_labels()
        axes10[0].legend(handles + [Line2D([], [], color="0.45", linestyle="--"),
                                   Line2D([], [], color="0.55", linestyle=":")],
                          labels + ["Union bound", f"Coverage target = {epsilon:g}"], fontsize=8, frameon=False)
        M, E = analysis["independent_scenes"], analysis["experiment"]["episodes_per_scene"]
        timing = analysis["analysis_config"]["timing"]
        for number, figure in ((10, fig10), (11, fig11)):
            note = f"{M} independent scenes; {E} sensing trials per scene; pointwise 95% intervals"
            if M < 2:
                note = "Single-scene check: scene confidence intervals unavailable (not a statistical performance result)"
            if number == 11:
                note += f"\nNormalized timing: H={timing['horizon']:g}, switch={timing['switch']:g}, gap={timing['gap']:g}, processing={timing['processing']:g}"
            figure.text(0.5, 0.018, note, ha="center", fontsize=8, color="0.4")
            figure.tight_layout(rect=(0, 0.10, 1, 1))
            stem = "fig10_listening_protection" if number == 10 else "fig11_listening_service_cost"
            for extension in ("pdf", "png"):
                figure.savefig(directory / f"{stem}.{extension}")
    return fig10, fig11
