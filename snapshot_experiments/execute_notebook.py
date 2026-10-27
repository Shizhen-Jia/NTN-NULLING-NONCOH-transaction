"""Execute the root notebook and archive its populated outputs in the result folder."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

from .config import PROJECT_ROOT, SnapshotConfig
from .workflow import finalize_run
from .storage import atomic_json


def main():
    path = PROJECT_ROOT / "MUSIC_Snapshot_Experiment.ipynb"
    notebook = nbformat.read(path, as_version=4)
    # Remove previous outputs before executing a fresh or resumed run.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    def completed(cell, cell_index, **kwargs):
        nbformat.write(notebook, path)
        print(f"Executed notebook cell {cell_index+1}/{len(notebook.cells)}", flush=True)
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                print(output.get("text", "").strip(), flush=True)
    client = NotebookClient(notebook, timeout=None, kernel_name="sionna20",
                            resources={"metadata": {"path": str(PROJECT_ROOT)}},
                            on_cell_executed=completed)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    # Read the directory actually printed by the final executed cell.
    output = "\n".join(o.get("text", "") for c in notebook.cells for o in c.get("outputs", [])
                       if o.get("output_type") == "stream")
    line = [line for line in output.splitlines() if line.startswith("RESULT_DIR=")][-1]
    directory = Path(line.split("=", 1)[1])
    nbformat.write(notebook, directory / "MUSIC_Snapshot_Experiment.executed.ipynb")
    config = SnapshotConfig(**json.loads((directory / "run_config.json").read_text())["config"])
    finalize_run(directory, config)
    print(f"Notebook completed and archived: {directory}", flush=True)


if __name__ == "__main__":
    main()
