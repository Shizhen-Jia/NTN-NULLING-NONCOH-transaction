"""Replay cached channels without ray tracing or changing existing CDF files.

Usage:
    python replay_music_accuracy.py result/music_improved_<timestamp>
    python replay_music_accuracy.py result/music_improved_<timestamp> --variants improved hemisphere
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

import ntn_music_detection as nmd
import nulling_cdf_utils as ncu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--variants", nargs="+", choices=("improved", "hemisphere", "cone60"),
                        default=["cone60", "hemisphere", "improved"])
    args = parser.parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text())
    channels = sorted((args.run_dir / "channels").glob("channels_*.npz"))
    if not channels:
        parser.error("No cached channels found.")
    report = {"note": "All variants use the corrected engine and identical cached channels; cone60 is not the historical implementation.",
              "variants": {}}
    for variant in args.variants:
        rows = []
        kwargs = dict(config["music"])
        kwargs["peak_local_maxima"] = True
        kwargs["peak_max_noise_projection"] = .2
        kwargs["covariance_refine"] = variant == "improved"
        kwargs["sector_forward_only"] = True
        kwargs["sector_forward_cos_min"] = .5 if variant == "cone60" else 0.
        for path in channels:
            with np.load(path, allow_pickle=False) as cache:
                h = cache["h_ntn"]
                settings = dict(kwargs, array_positions_local=cache["array_positions_local"],
                                tx_orientations_rad=cache["tx_orientations_rad"])
                output = nmd.run_music_standard_pipeline(h, **settings)
            lookup = ncu.build_music_tx_lookup(output, num_ntn_rx=h.shape[0],
                                               num_tx_total=h.shape[2], num_tx_ant=h.shape[3])
            metrics = ncu.summarize_music_noncoh_quality(h, lookup)
            metrics.update(ncu.summarize_music_covariance_quality(h, lookup))
            row = {"macro": path.stem, **metrics}
            for key in ("covariance_fit_before", "covariance_fit_after"):
                values = output[key]
                finite = values[np.isfinite(values)]
                row[key] = float(finite.mean()) if finite.size else None
            rows.append(row)
            print(variant, json.dumps(row), flush=True)
        report["variants"][variant] = rows
    output_path = args.run_dir / datetime.now().strftime("replay_%Y%m%d_%H%M%S_%f.json")
    with output_path.open("x") as stream:
        json.dump(report, stream, indent=2)
    print("Replay saved:", output_path, flush=True)


if __name__ == "__main__":
    main()

