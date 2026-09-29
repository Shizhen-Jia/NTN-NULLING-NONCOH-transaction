# Appendix D — English presentation

A 16:9 presentation for a 25–30 minute technical explanation of the current Appendix D.

## Deliverables

- `Appendix_D_Explained.pptx`: 30 slides with English speaker notes. Text and diagram shapes are editable; formulas and scientific plots are high-resolution images.
- `Appendix_D_Explained.pdf`: matching presentation PDF with embedded fonts.
- `Speaker_Notes.pdf` and `Speaker_Notes.md`: a page-by-page English talk track.
- `preview/`: rendered slides and five contact sheets.
- `build_slides.py`: reproducible source for the presentation and charts.
- `sources.json`: exact paper/result sources and SHA256 hashes.

## Suggested talk

Slides 1–4 explain the problem and architecture. Slides 5–9 cover sensing timing and channel protection. Slides 10–14 develop service/risk constraints and the robust SOCP. Slides 15–19 explain causal history, the occupation LP, randomization and theorem conditions. Slides 20–25 discuss validation and conclusions. Slides 26–30 are optional technical backups.

All text and notes are English. The mathematical content follows `overleaf_ntn_paper/direction.tex`. Experimental charts use the fixed run `result/appendix_d_20260921_235448_775259`; the builder reads existing results and does not run new experiments or change the paper.

The slides distinguish exact set-wise protection, finite-model optimality and physical-system protection. Fresh Sionna RT results are static spatial validation. The repeated-observation and random-arrival examples are fixed causal diagnostics, which violate the main TN service budget; they are not presented as feasible optimal-J improvements.

## Rebuild

Use a separate Python environment containing `python-pptx`, `reportlab`, `matplotlib`, `numpy`, and `Pillow`, with the DejaVu Sans fonts installed:

```bash
python presentations/appendix_d/build_slides.py
```

The current build used `/tmp/appendix-d-slides-env/bin/python`. PowerPoint uses the DejaVu Sans font; if it is absent on another machine, install it or use the PDF to preserve the exact layout. Equations and plots are created locally from code, not generated artwork.

## Checks

The PDF was rendered and visually reviewed across all 30 slides; title/card wrapping issues were corrected. The PowerPoint archive was reopened and checked for 30 slides, 30 nonempty notes, English text, and shape bounds. It was not rendered through Microsoft PowerPoint itself. The PDF and PPTX share the same layout specification.
