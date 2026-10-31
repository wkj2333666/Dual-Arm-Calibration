import io
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile

import pytest

from dual_arm_calibration.demo import generate
from dual_arm_calibration.web import Workbench, empty_draft, make_server


def finished(app, jid):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = next(j for j in app.state()["jobs"] if j["id"] == jid)
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("Offline solve did not complete")


def test_empty_draft_is_persistent_but_never_connects_hardware(tmp_path):
    app = Workbench(tmp_path)
    try:
        assert app.state()["draft"] == empty_draft()
        assert all(v is None for v in app.state()["draft"]["board"].values())
        draft = empty_draft()
        draft["arms"]["left"]["adapter"] = "not-installed"
        app.save_draft(draft)
        assert not app.state()["hardware_connected"]
    finally:
        app.close()
    app = Workbench(tmp_path)
    try:
        assert app.state()["draft"] == draft
        assert not app.state()["hardware_connected"]
        with pytest.raises(ValueError):
            app.save_draft({"board": {"squares_x": 7}})
    finally:
        app.close()


def test_demo_download_and_restart(tmp_path):
    app = Workbench(tmp_path)
    try:
        jid = app.start(dict(kind="demo", table=True))["id"]
        assert finished(app, jid)["status"] == "succeeded"
        result = json.loads(app.download(jid))
        assert result["table"]["calibrated"]
        with zipfile.ZipFile(io.BytesIO(app.download(jid, archive=True))) as z:
            assert set(z.namelist()) == {
                "rig.json",
                "observations.json",
                "calibration.json",
                "manifest.json",
            }
            assert json.loads(z.read("manifest.json"))["frozen"]
        assert app.state()["draft"] == empty_draft()
    finally:
        app.close()
    app = Workbench(tmp_path)
    try:
        assert app.state()["jobs"][0]["id"] == jid
        assert app.download(jid)
    finally:
        app.close()


def test_import_good_and_bad_observations(tmp_path):
    app = Workbench(tmp_path)
    try:
        rig, data, _ = generate()
        jid = app.start(dict(kind="import", rig=rig, data=data))["id"]
        job = finished(app, jid)
        assert job["status"] == "succeeded"
        assert not job["summary"]["table_calibrated"]
        data["rig_sha256"] = "wrong-binding"
        bad = app.start(dict(kind="import", rig=rig, data=data))["id"]
        job = finished(app, bad)
        assert job["status"] == "failed" and "bound" in job["error"]
        with pytest.raises(ValueError, match="通过"):
            app.download(bad)
        with pytest.raises(ValueError, match="请选择"):
            app.start(dict(kind="connect"))
    finally:
        app.close()


def test_interrupted_jobs_are_marked_failed(tmp_path):
    folder = tmp_path / ("a" * 32)
    folder.mkdir()
    (folder / "job.json").write_text(
        json.dumps(
            dict(id="a" * 32, status="running", created="2026-10-31T00:00:00+00:00")
        )
    )
    app = Workbench(tmp_path)
    try:
        assert app.state()["jobs"][0]["status"] == "failed"
    finally:
        app.close()


@pytest.fixture
def server(tmp_path):
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server.app
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        server.app.close()


def test_http_assets_draft_and_download(server):
    base, app = server
    with urllib.request.urlopen(base) as r:
        assert r.headers["Content-Type"].startswith("text/html")
    for path in ("/app.js", "/style.css"):
        with urllib.request.urlopen(base + path) as r:
            assert r.status == 200 and r.read()
    request = urllib.request.Request(
        base + "/api/draft",
        data=json.dumps(empty_draft()).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as r:
        assert json.load(r)["saved"]
    request = urllib.request.Request(
        base + "/api/jobs",
        data=json.dumps(dict(kind="demo")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as r:
        assert r.status == 202
        jid = json.load(r)["id"]
    assert finished(app, jid)["status"] == "succeeded"
    with urllib.request.urlopen(base + f"/api/jobs/{jid}/bundle.zip") as r:
        assert r.headers["Content-Type"] == "application/zip"
        assert zipfile.is_zipfile(io.BytesIO(r.read()))


@pytest.mark.parametrize(
    "path,headers,payload,status",
    [
        (
            "/api/draft",
            {
                "Content-Type": "application/json",
                "Origin": "https://another-site.example",
            },
            b"{}",
            403,
        ),
        ("/api/draft", {"Content-Type": "text/plain"}, b"{}", 415),
        ("/api/jobs", {"Content-Type": "application/json"}, b"not json", 400),
        ("/api/connect", {"Content-Type": "application/json"}, b"{}", 404),
    ],
)
def test_http_rejects_unconfigured_and_invalid_requests(
    server, path, headers, payload, status
):
    request = urllib.request.Request(server[0] + path, data=payload, headers=headers)
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request)
    assert caught.value.code == status
