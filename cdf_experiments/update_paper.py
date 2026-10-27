"""Copy validated CDF figures and replace the corresponding paper statistics."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def update(directory, paper):
    directory, paper = Path(directory), Path(paper)
    report = json.loads((directory / "paper_summary.json").read_text())
    validation = json.loads((directory / "paper_render_validation.json").read_text())
    assert report["num_ntn_ues_per_scene"] == 50 and report["paired_across_lambdas"]
    assert all(row["legends_inside"] and row["all_samples_visible"] for row in validation.values())
    target = paper / "results_pic"
    copies = []
    for stem in ("nulling_inr_cdf", "nulling_tn_sinr_cdf", "nulling_cdf_combined"):
        for ext in ("pdf", "png"):
            name = f"{stem}.{ext}"
            shutil.copy2(directory / name, target / name)
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
            copies.append(dict(name=name, sha256=digest))
    path = paper / "results.tex"
    text = path.read_text()
    start = text.index("We evaluate the spatial beamforming component")
    end = text.index("\\begin{table}", start)
    introduction = r"""We evaluate the spatial beamforming component through Monte Carlo simulations over a \SI{10}{\kilo\meter}$\times$\SI{10}{\kilo\meter} rural area near Boulder, Colorado, illustrated in Fig.~\ref{fig:map}, using the Sionna ray-tracing platform~\cite{sionna}. The terrestrial network consists of four outdoor BS sites on a $2\times2$ grid, each with three sectors. Each CDF realization contains 50 NTN-UEs and 12 TN-UEs, with one TN-UE sampled for each serving sector subject to the association and TN-link constraints. Table~\ref{tab:simulation_parameters} summarizes the CDF parameters. The common NTN look direction is resampled in each realization with azimuth uniformly distributed over $[0^\circ,360^\circ]$ and elevation over $[25^\circ,90^\circ]$. Both $\lambda=10^{11}$ and $\lambda=10^{13}$ use the same 20 independently drawn macro realizations, with fixed channels and TN scheduling across the two penalty weights. Each no-nulling sample enters the common baseline once, without repetition across penalty weights or frequencies. The downlink carrier is \SI{8}{\giga\hertz}, and the uplink offsets are $-20\%$, $-10\%$, $0\%$, $+10\%$, and $+20\%$. For these frequency-offset curves, the spatially smoothed MUSIC detector uses the analytic covariance of the RT channels; finite-snapshot covariance estimation error is not included. The source count is selected using the configured eigenvalue threshold. The CDFs report NTN INR and TN SINR at fixed transmit power.

"""
    text = text[:start]+introduction+text[end:]
    text = text.replace("NTN-UE Deployment & 20 UEs;", "NTN-UE Deployment & 50 UEs;")
    ntn, tn = report["baseline"]["inr"], report["baseline"]["sinr"]
    low, high = report["lambdas"]["11"], report["lambdas"]["13"]
    low0, high0 = low["0.0"], high["0.0"]
    low_range = [100*v["inr"]["fraction_le_minus3"] for v in low.values()]
    high_range = [100*v["inr"]["fraction_le_minus3"] for v in high.values()]
    inr = rf"""Fig.~\ref{{fig:nulling_cdf}}(a) includes {ntn['count']} NTN samples with nonzero RT links, including undetected terminals, from the 20 shared drops. The black no-nulling curve contains each sample once; {100*ntn['fraction_le_minus3']:.1f}\% of samples have INR at or below \SI{{-3}}{{dB}}. At the co-frequency \SI{{8}}{{\giga\hertz}} operating point, the post-nulling fractions are {100*low0['inr']['fraction_le_minus3']:.1f}\% for $\lambda=10^{{11}}$ and {100*high0['inr']['fraction_le_minus3']:.1f}\% for $\lambda=10^{{13}}$. Across the five uplink frequencies, these fractions range from {min(low_range):.1f}\% to {max(low_range):.1f}\% and from {min(high_range):.1f}\% to {max(high_range):.1f}\%, respectively. The two penalty weights are compared on identical drops. The gray curve reports the pre-nulling INR of {report['undetected_count']} terminals not detected by any sector at the co-frequency operating point, excluding zero-link and zero-INR samples."""
    sinr = rf"""Fig.~\ref{{fig:nulling_cdf}}(b) contains {tn['count']} TN samples for each penalty weight and the same {tn['count']} samples in the common no-nulling baseline. The baseline median and fifth-percentile SINR are \SI{{{tn['median_db']:.2f}}}{{dB}} and \SI{{{tn['p05_db']:.2f}}}{{dB}}, respectively. At the co-frequency operating point, the median SINRs are \SI{{{low0['sinr']['median_db']:.2f}}}{{dB}} for $\lambda=10^{{11}}$ and \SI{{{high0['sinr']['median_db']:.2f}}}{{dB}} for $\lambda=10^{{13}}$, while the fifth-percentile values are \SI{{{low0['sinr']['p05_db']:.2f}}}{{dB}} and \SI{{{high0['sinr']['p05_db']:.2f}}}{{dB}}. Relative to each user's own no-nulling baseline, {100*low0['sinr']['fraction_loss_gt3']:.1f}\% and {100*high0['sinr']['fraction_loss_gt3']:.1f}\% of TN users lose more than \SI{{3}}{{dB}} under the two penalty weights. These lower-tail statistics quantify the TN cost alongside the NTN protection benefit."""
    for label, paragraph in (("INR", inr), ("SINR", sinr)):
        marker = f"\\paragraph*{{{'NTN INR' if label == 'INR' else 'TN SINR'}}}\\label{{{label}}}\n"
        start = text.index(marker)+len(marker)
        end = text.index("\n\n", start)
        text = text[:start]+paragraph+text[end:]
    start = text.index("\\caption{Empirical CDFs of (a) NTN-UE")
    end = text.index("\n\\label{fig:nulling_cdf}", start)
    caption = rf"""\caption{{Empirical CDFs of (a) NTN-UE downlink INR and (b) TN-UE downlink SINR for 50 deployed NTN-UEs per drop. Colors identify $\lambda$, and line styles identify uplink offsets from the \SI{{8}}{{\giga\hertz}} downlink; solid colored curves use the same uplink frequency. Both penalty weights share 20 independent drops. The black baselines contain {ntn['count']} NTN and {tn['count']} TN samples, each counted once. Gray in (a) shows the pre-nulling INR of {report['undetected_count']} undetected terminals with nonzero RT links. Insets magnify the outlined regions using the same colors and line styles. The curves show NTN interference suppression and the associated TN SINR cost.}}"""
    text = text[:start]+caption+text[end:]
    assert "batch A" not in text and "batch B" not in text
    assert "NTN-UE Deployment & 50 UEs;" in text
    path.write_text(text)
    record = dict(source_run=str(directory.resolve()), paper=str(paper.resolve()), figures=copies,
                  results_tex_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (directory / "overleaf_update.json").write_text(json.dumps(record, indent=2)+"\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("paper", type=Path)
    args = parser.parse_args()
    print(json.dumps(update(args.directory, args.paper), indent=2))
