"""Build a portable executable plus per-user installer ZIP (no admin required)."""
import importlib.metadata as metadata
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent


def build():
    if sys.platform != "win32":
        raise RuntimeError("Build the Windows release on Windows")
    from app.version import __version__
    from utilities.build_application_icon import build_icon
    from app.paths import LOGO_PATH, WINDOWS_ICON_PATH
    build_icon(LOGO_PATH, WINDOWS_ICON_PATH)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / ".build-deps") + os.pathsep + env.get("PYTHONPATH", "")
    build_id = uuid4().hex[:10]
    env["PYINSTALLER_CONFIG_DIR"] = str(ROOT / "build" / build_id / "cache")
    # Fresh output avoids recursive replacement of folders locked by cloud sync.
    dist = ROOT / "dist" / build_id
    workpath = ROOT / "build" / build_id / "work"
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(dist),
                    "--workpath", str(workpath), str(ROOT / "packaging" / "CrystaLoop.spec")],
                   cwd=ROOT, env=env, check=True)
    from utilities.verify_frozen_sources import verify
    verify(ROOT, workpath)
    payload = dist / "CrystaLoop"
    optional_modules = {name: find_spec(name) is not None for name in ("nidaqmx", "sqlalchemy", "psycopg")}
    include_bo = all(find_spec(module) is not None for module in ("torch", "sklearn"))
    optional_modules.update({"torch": include_bo, "scikit-learn": include_bo})
    info = {"version": __version__, "python": sys.version, "architecture": "Windows x64",
            "optional_modules": optional_modules}
    (payload / "BUILD_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    notices = payload / "THIRD_PARTY_NOTICES"
    notices.mkdir(exist_ok=True)
    # Include runtime dependency licenses; onedir keeps Qt libraries replaceable.
    from packaging.requirements import Requirement
    queue = [Requirement(line).name for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    queue += [name for name, present in info["optional_modules"].items() if present]
    distributions, seen = [], set()
    while queue:
        name = queue.pop()
        normalized = name.lower().replace("_", "-")
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        distributions.append(distribution)
        for requirement in distribution.requires or []:
            requirement = Requirement(requirement)
            if requirement.marker is None or requirement.marker.evaluate():
                queue.append(requirement.name)
    inventory = []
    for distribution in distributions:
        name = distribution.metadata.get("Name", "unknown")
        inventory.append({"name": name, "version": distribution.version,
                          "license": distribution.metadata.get("License-Expression") or distribution.metadata.get("License", "")})
        for entry in distribution.files or []:
            if entry.name.lower().startswith(("license", "copying", "copyright", "notice")):
                source = Path(distribution.locate_file(entry))
                if source.is_file():
                    target = notices / name / Path(str(entry).replace("..", "_")).name
                    target.parent.mkdir(exist_ok=True)
                    if not target.exists():
                        shutil.copyfile(source, target)
    (notices / "installed-distributions.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.exists():
        shutil.copyfile(python_license, notices / "Python-LICENSE.txt")
    # Stage only packaged binaries and installer files, never user recordings/settings.
    staging = ROOT / "build" / "release" / uuid4().hex[:10] / f"CrystaLoop-{__version__}-windows-x64"
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copytree(payload, staging / "CrystaLoop", dirs_exist_ok=True)
    for name in ("Install.cmd", "Install.ps1", "Uninstall.ps1", "START_HERE.txt"):
        shutil.copyfile(ROOT / "packaging" / name, staging / name)
    release = ROOT / "release"
    release.mkdir(exist_ok=True)
    archive = shutil.make_archive(str(release / staging.name), "zip", staging.parent, staging.name)
    (ROOT / "build" / "latest_release.json").write_text(json.dumps({"executable": str(payload / "CrystaLoop.exe"), "zip": archive, "workpath": str(workpath)}, indent=2), encoding="utf-8")
    print(f"Release ready: {archive}", flush=True)


if __name__ == "__main__":
    build()
