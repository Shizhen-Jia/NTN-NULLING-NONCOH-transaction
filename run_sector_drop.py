"""Execute the sector-drop notebook and preserve its outputs in the run archive."""
from datetime import datetime
from pathlib import Path
import os
import tempfile
import traceback

import nbformat
from nbclient import NotebookClient

import sector_drop_reporting as reporting


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    path = root / "Nulling_CDF_SectorDrop.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, kernel_name="sionna20", timeout=None,
                            resources={"metadata": {"path": str(root)}})
    directory = None
    messages = []
    def log(text):
        message = f"{datetime.now().astimezone().isoformat()} {text}"
        print(message, flush=True)
        messages.append(message)
    def locate_run():
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    for line in output.get("text", "").splitlines():
                        if line.startswith("Run outputs: "):
                            return root / line.removeprefix("Run outputs: ").strip()
        return None
    try:
        with client.setup_kernel():
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                log(f"Starting cell {index}")
                client.execute_cell(cell, index)
                directory = locate_run()
                log(f"Finished cell {index}" + (f"; archive={directory}" if directory else ""))
        if directory is None:
            raise RuntimeError("Notebook did not report its run directory")
        nbformat.write(notebook, directory / "Nulling_CDF_SectorDrop.executed.ipynb")
        # Atomic replacement avoids leaving a partially written working notebook.
        staged = path.with_suffix(".executed.tmp")
        nbformat.write(notebook, staged)
        staged.replace(path)
        log(f"Completed: {directory}")
    except BaseException as exc:
        directory = locate_run() or directory
        log(traceback.format_exc())
        if directory is not None:
            reporting.update_status(directory, "failed", error=f"{type(exc).__name__}: {exc}")
            nbformat.write(notebook, directory / "Nulling_CDF_SectorDrop.partial.ipynb")
        else:
            partial = Path(tempfile.gettempdir()) / "Nulling_CDF_SectorDrop.partial.ipynb"
            nbformat.write(notebook, partial)
            log(f"Partial notebook: {partial}")
        raise
    finally:
        if directory is not None:
            (directory / "notebook_execution.log").write_text("\n".join(messages) + "\n")
            reporting.write_artifact_manifest(directory)


if __name__ == "__main__":
    main()
