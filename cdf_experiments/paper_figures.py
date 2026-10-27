"""Render the paired lambda comparison with one deduplicated no-nulling baseline."""
from pathlib import Path
import argparse
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

OFFSETS = [-20., -10., 0., 10., 20.]
EXPONENTS = [11, 13]
STYLES = ["--", "-.", "-", ":", (0, (5, 1.5, 1, 1.5, 1, 1.5))]
INSET_BOUNDS = [.075, .20, .41, .27]
ZOOM_WINDOWS = {
    "inr": dict(xlim=(-19., -3.), ylim=(.68, .84), xticks=[-15, -10, -5], yticks=[.7, .8]),
    "sinr": dict(xlim=(24., 29.), ylim=(.65, .82), xticks=[24, 26, 28], yticks=[.7, .8]),
}


def add_frequency_inset(ax, metric, colored_lines, colors):
    """Magnify full-record CDFs without filtering samples or renormalizing them."""
    window = ZOOM_WINDOWS[metric]
    inset = ax.inset_axes(INSET_BOUNDS, zorder=6)
    inset.set_facecolor("white")
    for exponent, band, original in colored_lines:
        x, y = original.get_xdata(), original.get_ydata()
        line, = inset.plot(x, y, color=colors[EXPONENTS.index(exponent)],
                          linestyle=STYLES[band], linewidth=.8,
                          zorder=3 if band == 2 else 2)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        visible = ((x >= window['xlim'][0]) & (x <= window['xlim'][1]) &
                   (y >= window['ylim'][0]) & (y <= window['ylim'][1]))
        assert visible.sum() >= 3, (metric, exponent, band)
    inset.set(xlim=window['xlim'], ylim=window['ylim'],
              xticks=window['xticks'], yticks=window['yticks'])
    inset.tick_params(labelsize=5.8, length=1.5, width=.45, pad=1)
    inset.grid(alpha=.22, linewidth=.3)
    for spine in inset.spines.values():
        spine.set_linewidth(.5)
        spine.set_color('.35')
    xmin, xmax = window['xlim']
    ymin, ymax = window['ylim']
    ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                          fill=False, edgecolor='.35', linewidth=.55, zorder=4))
    ax.add_artist(ConnectionPatch(xyA=(1, 1), coordsA=inset.transAxes,
                                  xyB=(xmin, ymin), coordsB=ax.transData,
                                  connectionstyle="angle,angleA=0,angleB=90,rad=0",
                                  color='.4', linewidth=.5, zorder=4))
    return inset


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n")


def quantiles(values):
    return dict(count=len(values), p05_db=float(np.quantile(values, .05)),
                median_db=float(np.median(values)), p90_db=float(np.quantile(values, .9)))


def read_data(directory):
    config = json.loads((directory / "run_config.json").read_text())
    assert config["positions"]["ntn_rx"] == 50 and config["num_macro_sims"] == 20
    assert config["ul_frequency_percentages"] == OFFSETS and config["f_dl"] == 8e9
    baseline, curves = {}, {}
    summary = dict(source_run=str(directory.resolve()), num_ntn_ues_per_scene=50,
                   scenes=20, paired_across_lambdas=True, lambdas={}, baseline={},
                   covariance_mode=config["music"]["detect_covariance_mode"],
                   baseline_definition="Each scene/user counted once, without duplication across lambdas or frequencies")
    with np.load(directory / "nulling_cdf_metrics.npz", allow_pickle=False) as z:
        for band, offset in enumerate(OFFSETS):
            prefix = f"ul_{band:03d}/"
            lambdas = z[prefix+"lambda_ranges_music_est"]
            for metric in ("inr", "sinr"):
                raw = z[prefix+f"raw_{metric}_db"]
                assert len(raw) and np.isfinite(raw).all()
                if not band:
                    baseline[metric] = raw.copy()
                    summary["baseline"][metric] = quantiles(raw)
                else:
                    np.testing.assert_array_equal(raw, baseline[metric])
                for exponent in EXPONENTS:
                    index, = np.flatnonzero(lambdas == 10.**exponent)
                    values = z[prefix+f"est_{metric}_db_index_{index:03d}"]
                    assert values.shape == raw.shape and np.isfinite(values).all()
                    curves[metric, exponent, band] = values.copy()
                    row = quantiles(values)
                    row["fraction_le_minus3"] = float(np.mean(values <= -3)) if metric == "inr" else None
                    row["fraction_loss_gt3"] = float(np.mean(raw-values > 3)) if metric == "sinr" else None
                    summary["lambdas"].setdefault(str(exponent), {}).setdefault(str(offset), {})[metric] = row
        undetected = z["ul_002/raw_undetected_inr_db"].copy()
        ids = z["ul_002/undetected_inr_sample_ids"]
        assert len(ids) == len(undetected) and len(np.unique(ids, axis=0)) == len(ids)
        assert np.isfinite(undetected).all()
    assert len(baseline["sinr"]) == 240
    assert len(baseline["inr"]) <= 1000
    summary["baseline"]["inr"]["fraction_le_minus3"] = float(np.mean(baseline["inr"] <= -3))
    summary["undetected_count"] = len(undetected)
    write_json(directory / "paper_summary.json", summary)
    return baseline, curves, undetected, summary


def render(directory):
    directory = Path(directory)
    baseline, curves, undetected, summary = read_data(directory)
    for p in (directory / "sources/fonts").glob("*.TTF"):
        font_manager.fontManager.addfont(str(p))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    colors = plt.get_cmap("viridis")(np.linspace(.15, .65, 2))
    rc = {"font.family": "Times New Roman", "font.size": 8, "axes.labelsize": 8,
          "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6, "axes.labelpad": 2,
          "xtick.major.pad": 2, "ytick.major.pad": 2, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
          "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
          "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
          "mathtext.cal": "Times New Roman:italic", "mathtext.sf": "Times New Roman",
          "mathtext.tt": "Times New Roman", "mathtext.fallback": None,
          "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None}
    panels = [("inr", "NTN-UE INR (dB)", "nulling_inr_cdf", "(a) NTN-UE INR"),
              ("sinr", "TN-UE SINR (dB)", "nulling_tn_sinr_cdf", "(b) TN-UE SINR")]
    validation = {}
    with plt.rc_context(rc):
        combined = plt.figure(figsize=(3.5, 4.6))
        axes = [combined.add_axes([.15, .615, .83, .35]), combined.add_axes([.15, .13, .83, .35])]
        for mode in ("standalone", "combined"):
            for panel, (metric, xlabel, stem, label) in enumerate(panels):
                if mode == "standalone":
                    fig, ax = plt.subplots(figsize=(3.5, 2.15))
                    fig.subplots_adjust(left=.15, bottom=.20, right=.98, top=.965)
                else:
                    fig, ax = combined, axes[panel]
                plotted, colored_lines = [], []
                def add_cdf(values, **style):
                    x = np.sort(values)
                    y = np.arange(1, len(x)+1)/len(x)
                    line, = ax.plot(x, y, **style)
                    np.testing.assert_array_equal(line.get_xdata(), x)
                    np.testing.assert_array_equal(line.get_ydata(), y)
                    plotted.append(values)
                    return line
                handles = [add_cdf(baseline[metric], color="black", linestyle="-", linewidth=1.1, label="No nulling")]
                if metric == "inr":
                    handles.append(add_cdf(undetected, color="0.5", linestyle="-", linewidth=1.2, label="Undetected"))
                for i, exponent in enumerate(EXPONENTS):
                    for band in range(len(OFFSETS)):
                        line = add_cdf(curves[metric, exponent, band], color=colors[i], linestyle=STYLES[band], linewidth=.95)
                        colored_lines.append((exponent, band, line))
                    handles.append(Line2D([], [], color=colors[i], linewidth=1.1, label=rf"$\lambda=10^{{{exponent}}}$"))
                all_values = np.concatenate(plotted)
                lo, hi = all_values.min(), all_values.max()
                pad = .04*(hi-lo)
                ax.set(xlabel=xlabel, ylabel="CDF", xlim=(lo-pad, hi+pad), ylim=(0, 1))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.grid(True, alpha=.25, linewidth=.4)
                opts = dict(fontsize=6.7, handlelength=2.3, framealpha=.94,
                            borderpad=.3, labelspacing=.23, handletextpad=.5)
                first = ax.legend(handles=handles, loc="upper left", ncol=2,
                                  columnspacing=.75, **{**opts, "handlelength": 1.6})
                ax.add_artist(first)
                styles = [Line2D([], [], color="0.4", linestyle=STYLES[i], linewidth=1.,
                                 label=f"{offset:+g}% ({8*(1+offset/100):g} GHz)") for i, offset in enumerate(OFFSETS)]
                second = ax.legend(handles=styles, loc="lower left", bbox_to_anchor=(0, .50),
                                   ncol=2, columnspacing=.75,
                                   **{**opts, "fontsize": 6.2, "labelspacing": .18})
                inset = add_frequency_inset(ax, metric, colored_lines, colors)
                text = ax.text(.5, -.30, label, transform=ax.transAxes, ha="center", va="center", fontsize=8) if mode == "combined" else None
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                b1, b2, inner = first.get_window_extent(renderer), second.get_window_extent(renderer), ax.get_window_extent()
                assert not b1.overlaps(b2), (mode, metric, b1.bounds, b2.bounds)
                for box in (b1, b2):
                    assert inner.contains(box.x0, box.y0) and inner.contains(box.x1, box.y1)
                zoom_box = inset.get_window_extent(renderer)
                assert inner.contains(zoom_box.x0, zoom_box.y0) and inner.contains(zoom_box.x1, zoom_box.y1)
                assert not zoom_box.overlaps(b1) and not zoom_box.overlaps(b2)
                for line in ax.lines:
                    path = line.get_path().transformed(line.get_transform())
                    assert not path.intersects_bbox(zoom_box, filled=False), (mode, metric, 'Inset hides a main curve')
                    for box in (b1, b2):
                        assert not path.intersects_bbox(box, filled=False), (mode, metric, 'Legend hides a main curve')
                assert len(ax.lines) == (12 if metric == "inr" else 11)
                if text is not None:
                    box = text.get_window_extent(renderer)
                    assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
                    assert not box.overlaps(ax.xaxis.label.get_window_extent(renderer))
                validation[f"{mode}_{metric}"] = dict(curves=len(ax.lines), no_nulling_curves=1,
                    baseline_samples=len(baseline[metric]), legends_inside=True, legends_nonoverlapping=True,
                    all_samples_visible=True, size_inches=fig.get_size_inches().tolist(),
                    inset=dict(bounds_axes=INSET_BOUNDS, **ZOOM_WINDOWS[metric],
                               curves=len(inset.lines), lambda_exponents=EXPONENTS,
                               all_original_cdf_samples_retained=True, cdf_not_renormalized=True,
                               main_curves_unobscured=True))
                if mode == "standalone":
                    for ext in ("pdf", "png"):
                        fig.savefig(directory / f"{stem}.{ext}", dpi=600)
                    plt.close(fig)
        for ext in ("pdf", "png"):
            combined.savefig(directory / f"nulling_cdf_combined.{ext}", dpi=600)
        plt.close(combined)
    write_json(directory / "paper_render_validation.json", validation)
    write_json(directory / "cdf_style_map.json", dict(font_family="Times New Roman", title=False,
        legends="inside", frequency_legend_anchor=[0, .50], lambda_exponents=EXPONENTS,
        ul_offsets=OFFSETS, ul_line_styles=STYLES, same_frequency_solid=True,
        no_nulling=dict(color="black", count=1, deduplicated=True),
        undetected=dict(color="0.5", linestyle="-", count=len(undetected)),
        combined_panels=[p[3] for p in panels], combined_size_inches=[3.5, 4.6],
        inset_bounds_axes=INSET_BOUNDS, zoom_windows=ZOOM_WINDOWS))
    destination = directory / "sources/cdf_experiments"
    destination.mkdir(exist_ok=True, parents=True)
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copy2(p, destination / p.name)
    from sector_drop_reporting import write_artifact_manifest
    write_artifact_manifest(directory)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.directory), indent=2))
