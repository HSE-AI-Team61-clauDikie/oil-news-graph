"""Execute a notebook through IPython in process, without local kernel sockets.

Usage: .venv/bin/python eda/execute_notebook.py eda/notebooks/08_news_corpus.ipynb
The normal Jupyter kernel also works outside environments that restrict sockets.
"""

import argparse
import os
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def execute(path: Path) -> None:
    """Run cells sequentially and persist actual rich outputs and errors."""
    path = path.resolve()
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    original_cwd = Path.cwd()
    os.chdir(path.parent)
    error = None
    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            cell.execution_count = shell.execution_count - 1
            cell.outputs = [
                nbformat.v4.new_output(
                    "display_data",
                    data=out.data,
                    metadata=out.metadata,
                )
                for out in captured.outputs
            ]
            for name, text in [
                ("stdout", captured.stdout),
                ("stderr", captured.stderr),
            ]:
                if text:
                    cell.outputs.append(
                        nbformat.v4.new_output("stream", name=name, text=text),
                    )
            error = result.error_before_exec or result.error_in_exec
            if error:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "error",
                        ename=type(error).__name__,
                        evalue=str(error),
                        traceback=[str(error)],
                    )
                )
                break
    finally:
        os.chdir(original_cwd)
        notebook.metadata["execution_engine"] = "IPython in process"
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    if error:
        raise error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    execute(parser.parse_args().notebook)
