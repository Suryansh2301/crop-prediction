"""Package marker + notebook import hook.

The modules in this package are Jupyter notebooks (src/*.ipynb). This hook lets the unchanged
scripts (run_all.py, app/streamlit_app.py, tests/) keep using
    from src import config, train_yield, ...
by executing the code cells of the matching .ipynb file on import.
"""
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


class _NotebookLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        nb = json.loads(self.path.read_text(encoding="utf-8"))
        module.__file__ = str(self.path)
        code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        exec(compile(code, str(self.path), "exec"), module.__dict__)


class _NotebookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(__name__ + "."):
            return None
        nb_path = _HERE / f"{fullname.rsplit('.', 1)[-1]}.ipynb"
        if nb_path.exists():
            return importlib.util.spec_from_loader(fullname, _NotebookLoader(nb_path), origin=str(nb_path))
        return None


if not any(isinstance(f, _NotebookFinder) for f in sys.meta_path):
    sys.meta_path.append(_NotebookFinder())
