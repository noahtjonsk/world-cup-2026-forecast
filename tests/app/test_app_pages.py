import py_compile
from pathlib import Path


def test_all_app_pages_compile():
    files = sorted(Path("app").rglob("*.py"))
    assert len(files) == 6                                   # Home + 5 pages
    for f in files:
        py_compile.compile(str(f), doraise=True)             # syntax check; no streamlit run
