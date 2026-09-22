"""Small, dependency-light exports shared by the dynamic experiments."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def rows_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def json_file(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False))


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(str(path) + '.pdf', bbox_inches='tight')
    fig.savefig(str(path) + '.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def ecdf(ax, data, label, **kwargs):
    data = np.sort(np.asarray(data, dtype=float))
    if len(data):
        ax.step(data, np.arange(1, len(data)+1)/len(data), where='post', label=label, **kwargs)


def latex_table(path, headers, rows, caption):
    def esc(s):
        return str(s).replace('_', r'\_').replace('%', r'\%')
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\caption{' + esc(caption) + '}',
             r'\begin{tabular}{' + 'l'*len(headers) + '}', r'\hline',
             ' & '.join(map(esc, headers)) + r' \\', r'\hline']
    lines += [' & '.join(map(esc, row)) + r' \\' for row in rows]
    lines += [r'\hline', r'\end{tabular}', r'\end{table*}']
    Path(path).write_text('\n'.join(lines) + '\n')
