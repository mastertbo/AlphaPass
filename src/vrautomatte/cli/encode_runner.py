"""Headless AV1 re-encoder queue for VRAutoMatte.

Encodes video files in-place to AV1 using av1_nvenc (NVIDIA GPU).
Encodes to a temp file first, then atomically replaces the original on success.
Skips files already encoded in AV1.

Commands
--------
vrautomatte-encode build --dir D:\\Videos
    Scans a directory recursively for .mp4 files (originals + _xalpha).
    Checks each file's codec via ffprobe — skips files already in AV1.
    Writes an encode queue to ~/.config/vrautomatte/encode_queue.json.
    Edit the file to reorder or remove entries before running.

vrautomatte-encode run [--now] [--crf N]
    Processes the encode queue one file at a time.
    Only runs between 02:00–07:00 unless --now is passed.
    Encodes to a sibling .tmp file, then replaces the original on success.

vrautomatte-encode status
    Shows pending / done / error counts and lists remaining files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_QUEUE = Path.home() / ".config" / "vrautomatte" / "encode_queue.json"
DEFAULT_CRF = 35
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


def _temp_path(video_path: str) -> Path:
    """Sibling temp file used during encoding."""
    p = Path(video_path)
    return p.parent / f".{p.stem}_av1enc_tmp{p.suffix}"


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def _get_codec(video_path: str) -> str | None:
    """Return the primary video codec name, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        codec = result.stdout.strip()
        return codec if codec else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _is_av1(video_path: str) -> bool:
    codec = _get_codec(video_path)
    return codec == "av1"


def _ffprobe_available() -> bool:
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            timeout=10,
        )
        return True
    except FileNotFoundError:
        return False


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=10,
        )
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> None:
    source_dir = Path(args.dir).resolve()
    if not source_dir.is_dir():
        print(f"ERROR: directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    if not _ffprobe_available():
        print("ERROR: ffprobe not found — install ffmpeg and ensure it is on PATH.",
              file=sys.stderr)
        sys.exit(1)

    queue_path = Path(args.queue)

    # Preserve existing entries (status, ordering) if not resetting
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
        and not p.name.startswith(".")          # skip our own temp files
    )

    items: list[dict] = []
    new_count = 0
    skip_av1 = 0
    skip_missing = 0

    print(f"Scanning {source_dir} ...")
    for p in found:
        inp = str(p)

        if inp in existing:
            items.append(existing[inp])
            continue

        if not p.exists():
            skip_missing += 1
            continue

        print(f"  checking {p.name} ...", end="\r", flush=True)
        if _is_av1(inp):
            skip_av1 += 1
            continue

        items.append({
            "input": inp,
            "status": "pending",
        })
        new_count += 1

    print(" " * 72, end="\r")  # clear progress line

    # Append previously-queued items from other directories
    for inp, item in existing.items():
        if not any(i["input"] == inp for i in items):
            items.append(item)

    queue = {
        "source_dir": str(source_dir),
        "built": datetime.now().isoformat(timespec="seconds"),
        "crf": args.crf,
        "items": items,
    }
    _save_queue(queue, queue_path)

    pending = sum(1 for i in items if i["status"] == "pending")
    done = sum(1 for i in items if i["status"] == "done")
    print(f"Queue written to: {queue_path}")
    print(f"  {new_count} new files added")
    print(f"  {skip_av1} already AV1 (skipped)")
    print(f"  {skip_missing} missing files (skipped)")
    print(f"  {pending} pending  |  {done} done  |  {len(items)} total")
    print()
    print("Tip: open encode_queue.json to reorder or remove entries before running.")


# ---------------------------------------------------------------------------
# encode one file
# ---------------------------------------------------------------------------

def _encode_file(input_path: str, crf: int) -> None:
    """Encode input_path to AV1 in-place using av1_nvenc.

    Encodes to a sibling temp file, then atomically replaces the original.
    Raises RuntimeError on failure and cleans up the temp file.
    """
    tmp = _temp_path(input_path)

    # Clean up any leftover temp from a previous failed run
    if tmp.exists():
        tmp.unlink()

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "av1_nvenc",
        "-cq", str(crf),        # AV1 NVENC quality (0=best, 51=worst)
        "-c:a", "copy",          # copy audio stream unchanged
        "-movflags", "+faststart",
        "-y",
        str(tmp),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"ffmpeg exited {result.returncode}:\n{result.stderr[-800:]}"
        )

    # Atomic replace: temp → original
    os.replace(str(tmp), input_path)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _in_window() -> bool:
    h = datetime.now().hour
    return RUN_START_HOUR <= h < RUN_END_HOUR


def cmd_run(args: argparse.Namespace) -> None:
    queue_path = Path(args.queue)

    if not queue_path.exists():
        print(f"No queue file found at: {queue_path}", file=sys.stderr)
        print("Run:  vrautomatte-encode build --dir <folder>  first.", file=sys.stderr)
        sys.exit(1)

    if not _ffmpeg_available():
        print("ERROR: ffmpeg not found — install ffmpeg and ensure it is on PATH.",
              file=sys.stderr)
        sys.exit(1)

    if not args.now and not _in_window():
        now = datetime.now().strftime("%H:%M")
        print(
            f"Outside processing window ({RUN_START_HOUR:02d}:00–{RUN_END_HOUR:02d}:00). "
            f"Current time: {now}. Use --now to override."
        )
        sys.exit(0)

    queue = _load_queue(queue_path)
    items = queue.get("items", [])
    crf = args.crf if args.crf is not None else queue.get("crf", DEFAULT_CRF)

    pending = [i for i in items if i["status"] == "pending"]
    if not pending:
        print("Encode queue is empty — nothing to process.")
        sys.exit(0)

    total = len(pending)
    done_count = sum(1 for i in items if i["status"] == "done")
    print(f"Encoding {total} pending files ({done_count} already done).")
    print(f"Encoder: av1_nvenc  CRF: {crf}")
    print(f"Time window: {RUN_START_HOUR:02d}:00–{RUN_END_HOUR:02d}:00"
          + ("  [bypassed with --now]" if args.now else ""))
    print()

    processed = 0

    for item in items:
        if item["status"] != "pending":
            continue

        if not args.now and not _in_window():
            print(
                f"\nReached {RUN_END_HOUR:02d}:00 — stopping for tonight. "
                f"{processed}/{total} files encoded this session."
            )
            break

        inp = Path(item["input"])
        print(f"[{processed + 1}/{total}] {inp.name}")

        # Skip if file no longer exists
        if not inp.exists():
            print("  File not found — skipping.")
            item["status"] = "error"
            item["error"] = "File not found"
            _save_queue(queue, queue_path)
            processed += 1
            continue

        # Skip if already AV1 (e.g. encoded outside the queue)
        if _is_av1(item["input"]):
            print("  Already AV1 — marking done.")
            item["status"] = "done"
            _save_queue(queue, queue_path)
            processed += 1
            continue

        start = datetime.now()
        try:
            _encode_file(item["input"], crf)
            elapsed = (datetime.now() - start).seconds
            print(f"  Done  ({elapsed}s)  →  {inp.name}")
            item["status"] = "done"
        except KeyboardInterrupt:
            # Clean up temp file if interrupted
            tmp = _temp_path(item["input"])
            if tmp.exists():
                tmp.unlink()
            print("\nInterrupted — progress saved. Temp file removed.")
            _save_queue(queue, queue_path)
            sys.exit(0)
        except Exception as e:
            print(f"  ERROR: {e}")
            item["status"] = "error"
            item["error"] = str(e)

        _save_queue(queue, queue_path)
        processed += 1

    remaining = sum(1 for i in items if i["status"] == "pending")
    if remaining == 0:
        print("\nAll files encoded!")
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

    print(f"Queue:      {queue_path}")
    print(f"Built:      {queue.get('built', 'unknown')}")
    print(f"Source dir: {queue.get('source_dir', 'unknown')}")
    print(f"CRF:        {queue.get('crf', DEFAULT_CRF)}")
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
        prog="vrautomatte-encode",
        description="Headless AV1 re-encoder queue for VRAutoMatte (av1_nvenc).",
    )
    parser.add_argument(
        "--queue",
        default=str(DEFAULT_QUEUE),
        metavar="PATH",
        help=f"Path to encode_queue.json (default: {DEFAULT_QUEUE})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Scan a folder and populate the encode queue.")
    p_build.add_argument(
        "--dir",
        required=True,
        metavar="DIR",
        help="Directory to scan recursively for video files.",
    )
    p_build.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        metavar="N",
        help=f"AV1 quality (0=best, 51=worst, default: {DEFAULT_CRF}).",
    )
    p_build.add_argument(
        "--reset",
        action="store_true",
        help="Discard existing queue entries and rebuild from scratch.",
    )

    # run
    p_run = sub.add_parser("run", help="Encode files in the queue (2am–7am by default).")
    p_run.add_argument(
        "--now",
        action="store_true",
        help="Bypass the time window and encode immediately.",
    )
    p_run.add_argument(
        "--crf",
        type=int,
        default=None,
        metavar="N",
        help="Override the CRF stored in the queue file.",
    )

    # status
    sub.add_parser("status", help="Show encode queue progress.")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
