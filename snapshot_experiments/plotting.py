"""Two compact, single-column paper figures with embedded Times New Roman fonts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, MultipleLocator, NullLocator

from .metrics import to_db
from .storage import atomic_json, read_npz


def plot_results(directory, *, detection_only=False):
    directory = Path(directory)
    config = json.loads((directory / "run_config.json").read_text())["config"]
    data = read_npz(directory / "snapshot_metrics.npz")
    for path in (directory / "sources/fonts").glob("*.TTF"):
        font_manager.fontManager.addfont(str(path))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    rc = {"font.family": "Times New Roman", "font.size": 8, "axes.labelsize": 8,
          "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
          "axes.linewidth": .6, "axes.labelpad": 2, "xtick.major.pad": 2, "ytick.major.pad": 2,
          "xtick.major.size": 2.5, "ytick.major.size": 2.5,
          "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
          "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
          "mathtext.cal": "Times New Roman:italic", "mathtext.sf": "Times New Roman",
          "mathtext.tt": "Times New Roman",
          "mathtext.fallback": None, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None}
    validation_path = directory / "render_validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
        fig.subplots_adjust(left=.11, right=.985, top=.96, bottom=.32, wspace=.30)
        t = data["snapshots"]
        ax = axes[0]
        ax.plot(t, data["matched_ue_mean"], color="#355f8d", marker="o", markersize=2.7, linewidth=1.1)
        ax.fill_between(t, data["matched_ue_lower"], data["matched_ue_upper"], color="#355f8d", alpha=.18, linewidth=0)
        ax.set_ylabel("Matched UEs")
        count_lo, count_hi = data["matched_ue_lower"].min(), data["matched_ue_upper"].max()
        count_pad = .08*max(count_hi-count_lo, 1.)
        ax.set_ylim(max(0, 2*np.floor((count_lo-count_pad)/2)),
                    2*np.ceil((count_hi+count_pad)/2))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, steps=[1, 2, 4, 5, 10]))
        for name, label, color, marker in [("power_recall", "Path power", "#355f8d", "o"),
                                           ("high_inr_ue_recall", "High-INR UEs", "#20908c", "s")]:
            axes[1].plot(t, data[name], color=color, marker=marker, markersize=2.7,
                         linewidth=1.1, label=label)
            axes[1].fill_between(t, data[name+"_lower"], data[name+"_upper"], color=color, alpha=.12, linewidth=0)
        recall_lo = min(data[name+"_lower"].min() for name in ("power_recall", "high_inr_ue_recall"))
        recall_hi = max(data[name+"_upper"].max() for name in ("power_recall", "high_inr_ue_recall"))
        recall_span = max(recall_hi-recall_lo, .05)
        # Keep every confidence interval and reserve space below it for the legend.
        axes[1].set(ylabel="Recall", ylim=(max(0, .05*np.floor((recall_lo-.45*recall_span)/.05)),
                                          min(1, .05*np.ceil((recall_hi+.08*recall_span)/.05))))
        axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        legend = axes[1].legend(loc="lower left", framealpha=.92, handlelength=1.2,
                                fontsize=7, borderpad=.25, labelspacing=.2, handletextpad=.4)
        labels = []
        for ax, label in zip(axes, ["(a) Detection count", "(b) Detection coverage"]):
            ax.set_xscale("log", base=2)
            ticks = [x for x in [64, 256, 1024, 8192] if min(t) <= x <= max(t)]
            ax.set_xticks(ticks, labels=[str(x) for x in ticks])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(min(t)/1.2, max(t)*1.2)
            ax.set_xlabel("Snapshots, $T$")
            ax.grid(True, linewidth=.4, alpha=.25)
            labels.append(ax.text(.5, -.30, label, transform=ax.transAxes,
                                  ha="center", va="top", fontsize=7.5))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, label in zip(axes, labels):
            # Place each panel caption two points below its actual x-axis label.
            top = ax.xaxis.label.get_window_extent(renderer).y0 - 2*fig.dpi/72
            label.set_y(ax.transAxes.inverted().transform((0, top))[1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, label in zip(axes, labels):
            box = label.get_window_extent(renderer)
            assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
            assert not box.overlaps(ax.xaxis.label.get_window_extent(renderer))
        lb = legend.get_window_extent(renderer)
        ab = axes[1].get_window_extent(renderer)
        assert ab.contains(lb.x0, lb.y0) and ab.contains(lb.x1, lb.y1)
        assert axes[0].get_ylim()[0] <= count_lo and axes[0].get_ylim()[1] >= count_hi
        assert axes[1].get_ylim()[0] <= recall_lo and axes[1].get_ylim()[1] >= recall_hi
        assert axes[1].yaxis.label.get_window_extent(renderer).x0 > axes[0].get_window_extent(renderer).x1
        for ext in ("pdf", "png"):
            fig.savefig(directory / f"snapshot_detection_ab.{ext}", dpi=600)
        validation["detection"] = dict(size_inches=[3.5, 1.9], panels=2, snapshots=t.tolist(),
                                       font="Times New Roman", legends_inside=True,
                                       num_ntn_ues_per_scene=config["num_ntn_ues"],
                                       quality_metrics=["power_recall", "high_inr_ue_recall"],
                                       high_inr_threshold_db=config["high_inr_threshold_db"],
                                       subplot_wspace=.30,
                                       panel_label_gap_points=2,
                                       count_ylim=list(axes[0].get_ylim()),
                                       recall_ylim=list(axes[1].get_ylim()),
                                       confidence_intervals_fully_visible=True)
        plt.close(fig)

        if detection_only:
            atomic_json(validation_path, validation)
            return directory / "snapshot_detection_ab.png", directory / "snapshot_inr_cdf.png"

        fig, ax = plt.subplots(figsize=(3.5, 2.05))
        fig.subplots_adjust(left=.15, right=.98, top=.965, bottom=.20)
        curves = [("No nulling", data["raw_inr_linear"], "black", "-")]
        # Separate the snapshot curves by both hue and line pattern for print.
        styles = [("#E69F00", "--"), ("#0072B2", "-."), ("#009E73", ":")]
        curves += [(f"$T={budget}$", data[f"inr_linear_T{budget}"], *styles[i % len(styles)])
                   for i, budget in enumerate(config["cdf_snapshots"])]
        all_x = []
        for label, power, color, linestyle in curves:
            x = np.sort(to_db(power, config["display_floor_db"]))
            y = np.arange(1, len(x)+1)/len(x)
            line, = ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.2, label=label)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            all_x.append(x)
        ax.set(xlabel="NTN-UE INR (dB)", ylabel="CDF", ylim=(0, 1))
        # Zoom the displayed range while retaining every sample in the ECDF.
        display_min = -60.
        ax.set_xlim(left=display_min)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.grid(True, linewidth=.4, alpha=.25)
        legend = ax.legend(loc="upper left", framealpha=.92, handlelength=2.4,
                           borderpad=.3, labelspacing=.25, handletextpad=.5)
        fig.canvas.draw()
        box, inner = legend.get_window_extent(fig.canvas.get_renderer()), ax.get_window_extent()
        assert inner.contains(box.x0, box.y0) and inner.contains(box.x1, box.y1)
        assert len(ax.lines) == 1+len(config["cdf_snapshots"])
        below_view = {label: int(np.sum(x < display_min)) for (label, _, _, _), x in zip(curves, all_x)}
        all_x = np.concatenate(all_x)
        assert ax.get_xlim()[0] == display_min and ax.get_xlim()[1] >= all_x.max()
        for ext in ("pdf", "png"):
            fig.savefig(directory / f"snapshot_inr_cdf.{ext}", dpi=600)
        validation["inr"] = dict(size_inches=[3.5, 2.05], curves=len(curves), xlim=list(ax.get_xlim()),
                                  all_samples_visible=bool(all_x.min() >= display_min),
                                  all_samples_used_in_ecdf=True, samples_below_display_min=below_view,
                                  curve_styles={label: dict(color=color, linestyle=linestyle)
                                                for label, _, color, linestyle in curves},
                                  font="Times New Roman",
                                  baseline_samples=len(data["raw_inr_linear"]), legends_inside=True)
        plt.close(fig)
    atomic_json(directory / "render_validation.json", validation)
    return directory / "snapshot_detection_ab.png", directory / "snapshot_inr_cdf.png"
