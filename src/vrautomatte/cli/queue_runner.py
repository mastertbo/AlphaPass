"""Headless queue runner for VRAutoMatte.

Commands
--------
vrautomatte-queue build --dir D:\\Videos
    Scans a directory recursively for .mp4 files and writes a queue.json.
    Skips files that already have a matching _xalpha output.
    Edit queue.json to reorder or remove entries before running.

vrautomatte-queue run [--queue PATH] [--now]
    Processes queue.json one file at a time.
    Only runs between 02:00–07:00 unless --now is passed.
    Saves progress after each file so it can resume across nights.

vrautomatte-queue status [--queue PATH]
    Shows pending / done counts and lists remaining files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_QUEUE = Path.home() / ".config" / "vrautomatte" / "queue.json"
RUN_START_HOUR = 2   # 02:00
RUN_END_HOUR = 12    # 12:00 (exclusive)
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


# ---------------------------------------------------------------------------
# Queue file helpers
# ---------------------------------------------------------------------------

def _load_queue(queue_path: Path) -> dict:
    with queue_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_queue(queue: dict, queue_path: Path) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = queue_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)
    tmp.replace(queue_path)


def _xalpha_output(input_path: str) -> str:
    p = Path(input_path)
    return str(p.parent / f"{p.stem}_xalpha{p.suffix}")


def _already_done(input_path: str) -> bool:
    """True if the _xalpha output file already exists on disk."""
    return Path(_xalpha_output(input_path)).exists()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> None:
    source_dir = Path(args.dir).resolve()
    if not source_dir.is_dir():
        print(f"ERROR: directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    queue_path = Path(args.queue)

    # Merge with existing queue to preserve manual reordering
    existing: dict[str, dict] = {}
    if queue_path.exists() and not args.reset:
        try:
            old = _load_queue(queue_path)
            for item in old.get("items", []):
                existing[item["input"]] = item
        except Exception:
            pass

    # Scan directory
    found = sorted(
        p for p in source_dir.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
        and "_xalpha" not in p.stem
    )

    items: list[dict] = []
    new_count = 0
    skip_count = 0

    for p in found:
        inp = str(p)
        if inp in existing:
            # Keep existing entry (preserves status + any manual reordering later)
            items.append(existing[inp])
            continue
        if _already_done(inp):
            skip_count += 1
            continue
        items.append({
            "input": inp,
            "output": _xalpha_output(inp),
            "status": "pending",
        })
        new_count += 1

    # Append any previously-queued items from other directories
    for inp, item in existing.items():
        if not any(i["input"] == inp for i in items):
            items.append(item)

    queue = {
        "source_dir": str(source_dir),
        "built": datetime.now().isoformat(timespec="seconds"),
        "items": items,
    }
    _save_queue(queue, queue_path)

    pending = sum(1 for i in items if i["status"] == "pending")
    done = sum(1 for i in items if i["status"] == "done")
    print(f"Queue written to: {queue_path}")
    print(f"  {new_count} new files added")
    print(f"  {skip_count} already processed (skipped)")
    print(f"  {pending} pending  |  {done} done  |  {len(items)} total")
    print()
    print("Tip: open queue.json to reorder or remove entries before running.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _in_window() -> bool:
    h = datetime.now().hour
    return RUN_START_HOUR <= h < RUN_END_HOUR


def _build_pipeline_config(item: dict, settings: dict):
    from vrautomatte.pipeline.runner import (
        OutputFormat,
        PipelineConfig,
        ProjectionType,
    )
    from vrautomatte.utils.settings import load_settings

    if settings is None:
        settings = load_settings()

    model_map = {0: "mobilenetv3", 1: "resnet50", 2: "matanyone2"}
    ds_map = {0: 0.125, 1: 0.25, 2: 0.5, 3: 1.0}
    chunk_map = {0: 100, 1: 250, 2: 500, 3: 1000}

    return PipelineConfig(
        input_path=item["input"],
        output_path=item["output"],
        output_format=OutputFormat.MATTE_ONLY,
        model_variant=model_map.get(settings.get("model_variant", 0), "mobilenetv3"),
        downsample_ratio=ds_map.get(settings.get("downsample_ratio", 0), 0.125),
        crf=settings.get("crf", 18),
        is_sbs=settings.get("is_sbs", False),
        pov_mode=settings.get("pov_mode", False),
        temporal_smoothing=1.0,
        chunk_size=chunk_map.get(settings.get("chunk_size", 2), 500),
        auto_resume=settings.get("auto_resume", True),
        temp_dir=settings.get("temp_dir", ""),
        projection=ProjectionType.EQUIRECTANGULAR,
    )


def cmd_run(args: argparse.Namespace) -> None:
    queue_path = Path(args.queue)

    if not queue_path.exists():
        print(f"No queue file found at: {queue_path}", file=sys.stderr)
        print("Run:  vrautomatte-queue build --dir <folder>  first.", file=sys.stderr)
        sys.exit(1)

    if not args.now and not _in_window():
        now = datetime.now().strftime("%H:%M")
        print(
            f"Outside processing window ({RUN_START_HOUR:02d}:00–{RUN_END_HOUR:02d}:00). "
            f"Current time: {now}. Use --now to override."
        )
        sys.exit(0)

    from vrautomatte.pipeline.runner import Pipeline, PipelineProgress
    from vrautomatte.utils.settings import load_settings

    settings = load_settings()
    queue = _load_queue(queue_path)
    items = queue.get("items", [])

    pending = [i for i in items if i["status"] == "pending"]
    if not pending:
        print("Queue is empty — nothing to process.")
        sys.exit(0)

    total = len(pending)
    done_count = sum(1 for i in items if i["status"] == "done")
    print(f"Processing {total} pending files ({done_count} already done).")
    print(f"Time window: {RUN_START_HOUR:02d}:00–{RUN_END_HOUR:02d}:00"
          + ("  [bypassed with --now]" if args.now else ""))
    print()

    processed = 0

    for item in items:
        if item["status"] != "pending":
            continue

        # Re-check time window before each file (unless --now)
        if not args.now and not _in_window():
            print(
                f"\nReached {RUN_END_HOUR:02d}:00 — stopping for tonight. "
                f"{processed}/{total} files processed this session."
            )
            break

        inp = Path(item["input"])
        print(f"[{processed + 1}/{total}] {inp.name}")

        # Skip if output already exists (processed outside the queue)
        if _already_done(item["input"]):
            print("  Output already exists — marking done.")
            item["status"] = "done"
            _save_queue(queue, queue_path)
            processed += 1
            continue

        config = _build_pipeline_config(item, settings)

        last_line: list[str] = [""]

        def on_progress(p: PipelineProgress, _ll=last_line) -> None:
            eta = f"  ETA {p.eta_sec:.0f}s" if p.eta_sec else ""
            fps = f"  {p.fps:.1f} fps" if p.fps else ""
            line = f"  {p.stage}  {p.frame_num}/{p.total_frames}{fps}{eta}"
            # Overwrite the same line
            print(f"\r{line:<72}", end="", flush=True)
            _ll[0] = line

        try:
            pipeline = Pipeline(config, on_progress=on_progress)
            output = pipeline.run()
            print(f"\r  Done → {Path(output).name:<68}")
            item["status"] = "done"
        except KeyboardInterrupt:
            print("\nInterrupted — progress saved.")
            _save_queue(queue, queue_path)
            sys.exit(0)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            item["status"] = "error"
            item["error"] = str(e)

        _save_queue(queue, queue_path)
        processed += 1

    remaining = sum(1 for i in items if i["status"] == "pending")
    if remaining == 0:
        print("\nAll files processed!")
    else:
        print(f"\n{remaining} files still pending.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"No queue file at: {queue_path}")
        sys.exit(0)

    queue = _load_queue(queue_path)
    items = queue.get("items", [])

    pending = [i for i in items if i["status"] == "pending"]
    done = [i for i in items if i["status"] == "done"]
    errors = [i for i in items if i["status"] == "error"]

    print(f"Queue: {queue_path}")
    print(f"Built: {queue.get('built', 'unknown')}")
    print(f"Source dir: {queue.get('source_dir', 'unknown')}")
    print()
    print(f"  {len(pending)} pending")
    print(f"  {len(done)} done")
    print(f"  {len(errors)} errors")
    print(f"  {len(items)} total")

    if pending:
        print(f"\nNext up ({min(5, len(pending))} of {len(pending)}):")
        for item in pending[:5]:
            print(f"  {item['input']}")

    if errors:
        print(f"\nErrors:")
        for item in errors:
            print(f"  {item['input']}")
            if "error" in item:
                print(f"    {item['error']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vrautomatte-queue",
        description="Headless batch queue manager for VRAutoMatte.",
    )
    parser.add_argument(
        "--queue",
        default=str(DEFAULT_QUEUE),
        metavar="PATH",
        help=f"Path to queue.json (default: {DEFAULT_QUEUE})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Scan a folder and populate the queue.")
    p_build.add_argument(
        "--dir",
        required=True,
        metavar="DIR",
        help="Directory to scan recursively for video files.",
    )
    p_build.add_argument(
        "--reset",
        action="store_true",
        help="Discard existing queue entries and rebuild from scratch.",
    )

    # run
    p_run = sub.add_parser("run", help="Process the queue (2am–7am window by default).")
    p_run.add_argument(
        "--now",
        action="store_true",
        help="Bypass the time window and process immediately.",
    )

    # status
    sub.add_parser("status", help="Show queue progress.")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
