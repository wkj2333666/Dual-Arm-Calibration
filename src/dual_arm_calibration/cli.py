import argparse
import json
from pathlib import Path
from .workflow import solve, check_rig, digest
from .storage import save_run, publish, atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline dual-arm / three-camera calibration; no robot motion"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    web = sub.add_parser(
        "web", help="Open a local web workbench; hardware remains unconfigured"
    )
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--data-dir", default="runs/web")
    web.add_argument(
        "--hardware-config",
        help="Explicit site-specific hardware JSON; no default hardware is loaded",
    )
    demo = sub.add_parser(
        "demo", help="Generate and solve a synthetic six/seven-joint rig"
    )
    demo.add_argument("--output", required=True)
    demo.add_argument("--table", action="store_true")
    run = sub.add_parser("solve", help="Solve a bound standard observation dataset")
    run.add_argument("--rig", required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    check = sub.add_parser("check-rig")
    check.add_argument("rig")
    pub = sub.add_parser(
        "publish",
        help="Publish a validated run to a local JSON file, preserving the previous version",
    )
    pub.add_argument("run")
    pub.add_argument("destination")
    video = sub.add_parser(
        "select-video", help="Select ChArUco observations from an existing video"
    )
    video.add_argument("video")
    video.add_argument("--rig", required=True)
    video.add_argument("--camera", choices=("left", "right", "top"), default="top")
    video.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "web":
            from .web import serve

            serve(
                args.data_dir,
                args.host,
                args.port,
                read(args.hardware_config) if args.hardware_config else None,
            )
            return
        if args.command == "check-rig":
            rig = read(args.rig)
            check_rig(rig)
            print(json.dumps(dict(valid=True, rig_sha256=digest(rig))))
            return
        if args.command == "publish":
            publish(args.run, args.destination)
            print(str(Path(args.destination).resolve()))
            return
        if args.command == "select-video":
            from .video import select_frames

            rig = read(args.rig)
            check_rig(rig)
            if Path(args.output).exists():
                raise ValueError("Output already exists")
            result = select_frames(
                args.video, rig["board"], rig["cameras"][args.camera]["model"]
            )
            result.update(rig_sha256=digest(rig), camera=args.camera)
            atomic_json(args.output, result)
            print(
                json.dumps(
                    dict(selected=len(result["samples"]), coverage=result["coverage"])
                )
            )
            return
        if Path(args.output).exists():
            raise ValueError("Output run already exists; choose a new run directory")
        if args.command == "demo":
            from .demo import generate

            rig, data, _ = generate(table=args.table)
        else:
            rig, data = read(args.rig), read(args.data)
        result = solve(rig, data)
        save_run(args.output, rig, data, result)
        print(
            json.dumps(
                dict(
                    run=str(Path(args.output).resolve()),
                    quality_passed=True,
                    world_frame=result["world_frame"],
                    table_calibrated=result["table"]["calibrated"],
                )
            )
        )
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Calibration failed: {exc}\n")


if __name__ == "__main__":
    main()
