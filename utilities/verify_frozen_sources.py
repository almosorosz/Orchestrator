"""Verify a frozen release contains the current project source bytecode."""
import marshal
import json
from pathlib import Path
import sys
from types import CodeType


def fingerprint(code):
    constants = tuple(fingerprint(value) if isinstance(value, CodeType) else value for value in code.co_consts)
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, constants)


def verify(root, workpath=None):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / ".build-deps"))
    from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader
    if workpath is None and (root / "build" / "latest_release.json").exists():
        workpath = json.loads((root / "build" / "latest_release.json").read_text())["workpath"]
    workpath = Path(workpath) if workpath else root / "build" / "pyinstaller"
    archive = ZlibArchiveReader(str(workpath / "CrystaLoop" / "PYZ-00.pyz"))
    mismatches, count = [], 0
    for name in archive.toc:
        path = root.joinpath(*name.split(".")).with_suffix(".py")
        if not path.is_file():
            path = root.joinpath(*name.split("."), "__init__.py")
        if not path.is_file():
            continue
        actual = archive.extract(name)
        if not isinstance(actual, CodeType):
            continue
        expected = compile(path.read_bytes(), str(path), "exec", dont_inherit=True, optimize=0)
        count += 1
        if fingerprint(actual) != fingerprint(expected):
            mismatches.append(name)
    executable = CArchiveReader(str(workpath / "CrystaLoop" / "CrystaLoop.exe"))
    entry = "launcher" if "launcher" in executable.toc else "MAIN_WINDOW"
    actual = marshal.loads(executable.extract(entry))
    expected = compile((root / (entry + ".py")).read_bytes(), entry + ".py", "exec", dont_inherit=True)
    if fingerprint(actual) != fingerprint(expected):
        mismatches.append("MAIN_WINDOW entry point")
    if mismatches:
        raise RuntimeError(f"Rebuild required: frozen modules differ from source: {mismatches}")
    print(f"Verified {count} frozen project modules and entry point against current source.", flush=True)


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent.parent)
