"""Local web workbench. Hardware integration is deliberately unconfigured."""

import copy
import io
import json
import re
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from .demo import generate
from .storage import atomic_json, save_run
from .workflow import solve

MAX_BODY = 16 * 1024 * 1024
JOB_ID = re.compile(r"^[a-f0-9]{32}$")


def empty_draft():
    return dict(
        arms={
            a: dict(adapter=None, model_id=None, connection=None)
            for a in ("left", "right")
        },
        cameras={
            r: dict(adapter=None, serial=None, profile=None)
            for r in ("left", "right", "top")
        },
        board=dict(
            squares_x=None,
            squares_y=None,
            square_length_mm=None,
            marker_length_mm=None,
            dictionary=None,
            thickness_mm=None,
        ),
        notes=None,
    )


def validate_draft(value):
    """Keep the draft separate from an executable rig; accept only known fields."""

    def normalize(data, template):
        if isinstance(template, dict):
            if not isinstance(data, dict) or set(data) != set(template):
                raise ValueError("接入草稿字段不完整或包含未知字段")
            return {k: normalize(data[k], v) for k, v in template.items()}
        if data is None:
            return None
        if not isinstance(data, str) or len(data) > 2000:
            raise ValueError("草稿字段应为不超过 2000 字的文本")
        return data.strip() or None

    return normalize(value, empty_draft())


class Workbench:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="calibration")
        self.jobs = {}
        self.draft_path = self.directory / "integration-draft.json"
        self.draft = (
            validate_draft(json.loads(self.draft_path.read_text()))
            if self.draft_path.exists()
            else empty_draft()
        )
        for path in self.directory.glob("*/job.json"):
            if JOB_ID.fullmatch(path.parent.name):
                job = json.loads(path.read_text())
                if job["status"] in ("queued", "running"):
                    job.update(
                        status="failed",
                        error="上次服务退出时任务尚未完成，请重新运行。",
                    )
                    atomic_json(path, job)
                self.jobs[path.parent.name] = job

    def state(self):
        with self.lock:
            return copy.deepcopy(
                dict(
                    draft=self.draft,
                    hardware_connected=False,
                    jobs=sorted(
                        self.jobs.values(), key=lambda j: j["created"], reverse=True
                    ),
                )
            )

    def save_draft(self, draft):
        draft = validate_draft(draft)
        with self.lock:
            atomic_json(self.draft_path, draft)
            self.draft = draft
        return dict(saved=True, hardware_connected=False)

    def _update(self, jid, **changes):
        with self.lock:
            self.jobs[jid].update(changes)
            atomic_json(self.directory / jid / "job.json", self.jobs[jid])

    def start(self, request):
        kind = request.get("kind")
        if kind not in ("demo", "import"):
            raise ValueError("请选择离线示例或导入数据")
        if kind == "demo" and not isinstance(request.get("table", False), bool):
            raise ValueError("桌面选项必须是布尔值")
        if kind == "import" and (
            not isinstance(request.get("rig"), dict)
            or not isinstance(request.get("data"), dict)
        ):
            raise ValueError("需要 rig.json 和 observations.json 两份数据")
        with self.lock:
            if any(j["status"] in ("queued", "running") for j in self.jobs.values()):
                raise ValueError("已有求解正在运行，请等待完成")
            jid = uuid.uuid4().hex
            job = dict(
                id=jid,
                kind=kind,
                status="queued",
                created=datetime.now(timezone.utc).isoformat(),
                error=None,
                summary=None,
            )
            atomic_json(self.directory / jid / "job.json", job)
            self.jobs[jid] = job
            self.pool.submit(self._run, jid, copy.deepcopy(request))
        return dict(id=jid)

    def _run(self, jid, request):
        try:
            self._update(jid, status="running")
            if request["kind"] == "demo":
                rig, data, _ = generate(table=request.get("table", False))
            else:
                rig, data = request["rig"], request["data"]
            result = solve(rig, data)
            save_run(self.directory / jid / "run", rig, data, result)
            summary = dict(
                world_frame=result["world_frame"],
                table_calibrated=result["table"]["calibrated"],
                max_translation_mm=result["validation"]["max_translation_mm"],
                max_rotation_deg=result["validation"]["max_rotation_deg"],
                wrists={
                    a: dict(
                        fit_px=result["wrists"][a]["reprojection_rms_px"],
                        heldout_px=result["wrists"][a]["validation"][
                            "heldout_reprojection_rms_px"
                        ],
                    )
                    for a in ("left", "right")
                },
            )
            self._update(jid, status="succeeded", summary=summary)
        except Exception as exc:
            self._update(jid, status="failed", error=str(exc) or type(exc).__name__)

    def download(self, jid, archive=False):
        with self.lock:
            if jid not in self.jobs or self.jobs[jid]["status"] != "succeeded":
                raise ValueError("任务尚无通过验证的结果")
        run = self.directory / jid / "run"
        if not archive:
            return (run / "calibration.json").read_bytes()
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name in (
                "rig.json",
                "observations.json",
                "calibration.json",
                "manifest.json",
            ):
                z.write(run / name, name)
        return out.getvalue()

    def close(self):
        self.pool.shutdown(wait=True)


def make_server(directory, host="127.0.0.1", port=8765):
    app = Workbench(directory)

    class Handler(BaseHTTPRequestHandler):
        def respond(
            self,
            status,
            payload,
            content_type="application/json; charset=utf-8",
            download=None,
        ):
            if not isinstance(payload, bytes):
                payload = json.dumps(
                    payload, ensure_ascii=False, allow_nan=False
                ).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            if download:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{download}"'
                )
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if path in assets:
                name, mime = assets[path]
                return self.respond(
                    200,
                    files("dual_arm_calibration").joinpath("static", name).read_bytes(),
                    mime,
                )
            if path == "/api/state":
                return self.respond(200, app.state())
            match = re.fullmatch(
                r"/api/jobs/([a-f0-9]{32})/(calibration\.json|bundle\.zip)", path
            )
            if match:
                jid, name = match.groups()
                try:
                    data = app.download(jid, archive=name.endswith("zip"))
                    return self.respond(
                        200,
                        data,
                        "application/zip"
                        if name.endswith("zip")
                        else "application/json",
                        f"{jid}-{name}",
                    )
                except (ValueError, OSError) as exc:
                    return self.respond(404, dict(error=str(exc)))
            self.respond(404, dict(error="未找到页面"))

        def do_POST(self):
            # JSON-only, same-origin writes: no arbitrary file paths or adapter execution.
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.respond(403, dict(error="不接受跨站请求"))
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.respond(415, dict(error="请求必须为 JSON"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    return self.respond(413, dict(error="请求应在 16 MiB 以内"))
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("请求应为 JSON 对象")
                path = urlsplit(self.path).path
                if path == "/api/draft":
                    return self.respond(200, app.save_draft(request))
                if path == "/api/jobs":
                    return self.respond(202, app.start(request))
                return self.respond(404, dict(error="接口待接入或不存在"))
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(400, dict(error=str(exc)))

        def log_message(self, format, *args):
            pass

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except Exception:
        app.close()
        raise
    server.app = app
    return server


def serve(directory, host="127.0.0.1", port=8765):
    server = make_server(directory, host, port)
    print(f"标定工作台：http://{host}:{server.server_port} （硬件待接入）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.app.close()
