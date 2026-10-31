"""Immutable run bundles and explicit atomic local publication."""

import json
import os
from pathlib import Path
import tempfile
from .workflow import digest


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".calibration-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_run(path, rig, data, result):
    if (
        not result.get("quality_passed")
        or result["rig_sha256"] != digest(rig)
        or result["dataset_sha256"] != digest(data)
    ):
        raise ValueError("Result does not match the validated inputs")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    for name, value in (("rig", rig), ("observations", data), ("calibration", result)):
        atomic_json(path / (name + ".json"), value)
    atomic_json(
        path / "manifest.json",
        dict(
            schema_version=1,
            frozen=True,
            hashes={
                n: digest(v)
                for n, v in (
                    ("rig", rig),
                    ("observations", data),
                    ("calibration", result),
                )
            },
        ),
    )


def publish(run_path, destination):
    path, dest = Path(run_path), Path(destination)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("frozen") is not True:
        raise ValueError("Run is not frozen")
    loaded = {
        n: json.loads((path / (n + ".json")).read_text())
        for n in ("rig", "observations", "calibration")
    }
    for n, v in loaded.items():
        if digest(v) != manifest["hashes"][n]:
            raise ValueError("Run changed after validation")
    result = loaded["calibration"]
    if (
        not result.get("quality_passed")
        or result["rig_sha256"] != digest(loaded["rig"])
        or result["dataset_sha256"] != digest(loaded["observations"])
    ):
        raise ValueError("Calibration did not pass or inputs changed")
    if dest.resolve().parent == path.resolve():
        raise ValueError("Publish outside the frozen run directory")
    if dest.exists():
        old = json.loads(dest.read_text())
        atomic_json(
            dest.with_name(dest.name + ".previous-" + digest(old)[:12] + ".json"), old
        )
    atomic_json(dest, result)
