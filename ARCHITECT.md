# AlphaPass — Architect's Handbook

> This document is the onboarding handbook for any Claude session working on AlphaPass.
> Read the **Foundation** section first, then read the chapter(s) relevant to your task.
> If your task spans multiple domains, read each relevant domain chapter before starting work.


---

## Chapter 1: Foundation

### What AlphaPass Is

AlphaPass is a Qt-based desktop application that generates alpha channel (transparency) mattes for VR passthrough video. It takes a VR video as input (equirectangular or fisheye, mono or side-by-side stereo), runs AI-based video matting to separate people from the background, and packs the resulting alpha channel into the video following the DeoVR specification. The output is a companion `_alpha.mp4` file that VR players use to render real-world passthrough behind the subject.

### The Team and Their Roles

AlphaPass is developed by a single human developer who uses multiple Claude sessions as "agent teams." Each session focuses on one domain of the codebase. The **architect** (this document's audience) is responsible for understanding how the domains connect, maintaining interface contracts between them, and ensuring that work done in one domain doesn't break another.

### The Five Domains

The codebase is organized into five ownership zones. Each domain has its own chapter in this handbook, its own interface contract, and its own test suite. A Claude session working on a task should identify which domain(s) the task belongs to and read only those chapters.

**Domain 1 — Pipeline Core** owns everything that transforms video frames into mattes. This includes the `MatteProcessor` protocol, all matting model implementations (RVM, MatAnyone2), the frame scaler, scene change detection, alpha smoothing, and checkpoint/resume logic. The orchestrator (`runner.py`) lives here too, since it coordinates the chunked matting pipeline.

**Domain 2 — Encoding & FFmpeg** owns frame extraction, video assembly, alpha packing (the DeoVR spec), codec selection, hardware-accelerated encoding (NVENC, SVT-AV1), and the CLI encode queue. Everything that talks to ffmpeg lives here.

**Domain 3 — User Interface** owns the PySide6/Qt GUI — the main window, preview pane, theme system, drag-and-drop, batch queue UI, and the background worker threading that connects the UI to the pipeline.

**Domain 4 — CLI & Batch** owns the headless runners (`queue_runner.py`, `encode_runner.py`) and any future command-line workflows. These are alternative entry points that use the pipeline without the GUI.

**Domain 5 — Infrastructure** owns cross-cutting concerns: GPU detection and auto-configuration, PyTorch bootstrap, settings persistence, SBS stereo utilities, and DeoVR mask management.

### How Domains Communicate

Domains interact through well-defined interfaces — Python protocols, dataclasses, and function signatures that are documented in each domain's chapter. The critical rule is: **if you need to change an interface that another domain depends on, you must update the interface contract in this handbook and flag it in the work log.** Internal implementation details within a domain can change freely.

### Code Conventions (All Domains)

These conventions apply everywhere and are non-negotiable:

- Logging uses `from loguru import logger` — never `print()`.
- Strings use double quotes.
- Line length caps at 100 characters.
- Imports are ordered: stdlib, then third-party, then local, alphabetical within each group.
- Type hints are required for all new or modified functions.
- Modules target 150 lines or fewer, with a hard cap at 200 (CSS/theme files exempt).
- Docstrings are required for modules, classes, and public functions.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `style:`).

### When to Escalate

Some decisions affect multiple domains and should not be made unilaterally by a single session. These include: changing the `MatteProcessor` protocol signature, modifying the `PipelineProgress` dataclass, altering the settings schema, changing the chunked pipeline's segment format, or modifying the DeoVR alpha packing spec compliance. If your task requires any of these, document the proposed change in the work log and flag it for the human developer's review before implementing.


---

## Chapter 2: Pipeline Core

### Ownership

This domain owns: `pipeline/matte.py`, `pipeline/rvm.py`, `pipeline/matanyone2.py`, `pipeline/sam2_masks.py`, `pipeline/scene_detect.py`, `pipeline/scaler.py`, `pipeline/checkpoint.py`, `pipeline/runner.py`.

### Interface Contract — MatteProcessor Protocol

The `MatteProcessor` protocol defined in `matte.py` is the most important interface in the codebase. Every matting model must implement it. The protocol requires:

- `process_frame(frame: np.ndarray) -> np.ndarray` — accepts a BGR frame (H, W, 3), returns a single-channel matte (H, W) with values 0.0–1.0.
- `reset()` — clears any recurrent state (important for scene changes and new videos).
- The processor must be constructable via `create_processor(model_name, device, **kwargs)` factory in `matte.py`.

**What other domains depend on:** The UI domain calls `create_processor()` to instantiate processors based on user selection. The runner calls `process_frame()` and `reset()`. The CLI domain uses the same factory. Any change to this protocol ripples everywhere.

### Interface Contract — PipelineRunner

`runner.py` exposes `PipelineRunner` which the UI's `PipelineWorker` and the CLI's `queue_runner` both instantiate. Key interfaces:

- `run(input_path, output_path, config) -> Path` — runs the full pipeline, returns the output path.
- Progress is emitted via a callback (`progress_callback(PipelineProgress)`) passed at construction.
- `PipelineProgress` is a dataclass with fields: `stage: str`, `stage_num: int`, `total_stages: int`, `frame: int | None`, `total_frames: int | None`, `fps: float | None`.

### Interface Contract — Checkpoint

- `PipelineCheckpoint.save(tmp_dir, input_path, config_hash, completed_segments, completed_frames)` — persists resume state.
- `PipelineCheckpoint.load(tmp_dir) -> PipelineCheckpoint | None` — loads checkpoint if it exists.
- `PipelineCheckpoint.validate(input_path, config_hash) -> bool` — confirms the checkpoint matches the current run.

### Internal Details (Free to Change)

Everything inside a processor's implementation (RVM's recurrent hidden state management, MatAnyone2's InferenceCore usage, SAM2 mask generation heuristics, the alpha smoother's EMA parameters, the scene detector's histogram correlation threshold and cooldown logic, the scaler's LANCZOS resize strategy) — all of this is internal to the pipeline core domain. Change it freely without updating this handbook, as long as the protocol interfaces above remain stable.

### Tests

Pipeline core tests live in `tests/test_matte_protocol.py`, `tests/test_scene_detect.py`, `tests/test_pov_mask.py`, and `tests/test_integration_matanyone2.py`. Run these after any pipeline change. All tests mock GPU operations — no GPU or FFmpeg required.


---

## Chapter 3: Encoding & FFmpeg

### Ownership

This domain owns: `utils/ffmpeg.py`, `cli/encode_runner.py`, and the alpha packing stages within `runner.py` (stages 4–5 specifically — the fisheye conversion and alpha compositing calls).

### Interface Contract — FFmpeg Utilities

`utils/ffmpeg.py` exposes these functions that other domains call:

- `extract_frames(input_path, output_dir, start_frame, end_frame, ...) -> int` — extracts a range of frames as PNGs using keyframe-seeking (`-ss` before `-i`). Returns frame count extracted.
- `assemble_video(frames_dir, output_path, fps, codec, crf) -> Path` — assembles PNGs into a video segment.
- `concat_segments(segment_paths, output_path) -> Path` — concatenates segment videos using ffmpeg's concat demuxer.
- `pack_alpha(source_video, matte_video, output_path, codec, crf)` — performs the full DeoVR alpha packing pipeline (scale, red-channel, segment, overlay, encode).
- `probe_video(path) -> dict` — returns video metadata (width, height, fps, frame_count, codec, duration).
- `_hwaccel_args() -> list[str]` — returns platform-appropriate hardware acceleration flags.

**What other domains depend on:** The pipeline runner calls all of these. The CLI encode_runner calls `probe_video` and raw ffmpeg for re-encoding. The UI displays metadata from `probe_video`.

### DeoVR Alpha Packing Specification

The alpha packing follows DeoVR's official spec precisely. The steps are: scale the matte to 40% of the source video resolution, convert to red-channel-only via `colorchannelmixer`, apply `colorkey` to make black transparent, split into 6 segments, overlay each segment into the corresponding fisheye corner area, and encode as AV1 (with HEVC and H.264 fallbacks). This spec is not something we invented — it must match what DeoVR expects. Do not modify the packing logic without verifying against DeoVR's documentation.

### Codec Fallback Chain

Encoding attempts codecs in this order: `av1_nvenc` (NVIDIA hardware) → `libsvtav1` (CPU SVT-AV1) → `hevc_nvenc` → `libx265` → `h264_nvenc` → `libx264`. The fallback is handled inside `pack_alpha` and `assemble_video`.

### Tests

Encoding tests are currently distributed across the integration tests. FFmpeg calls are mocked in tests — no ffmpeg binary required.


---

## Chapter 4: User Interface

### Ownership

This domain owns: `ui/main_window.py`, `ui/preview.py`, `ui/themes.py`, `ui/worker.py`.

### Interface Contract — PipelineWorker

`worker.py` defines `PipelineWorker(QThread)` which bridges the UI and the pipeline:

- Constructed with: `PipelineWorker(input_path, output_path, config)`.
- Emits Qt signals: `progress(PipelineProgress)`, `finished(Path)`, `error(str)`.
- Internally creates a `PipelineRunner` and calls `run()` on a background thread.

The UI's `MainWindow` connects to these signals to update the progress bar, preview pane, and status text. The worker is a thin wrapper — it should contain no pipeline logic, only threading and signal plumbing.

### Interface Contract — InstallWorker

`InstallWorker(QThread)` handles in-app dependency installation (e.g., MatAnyone2):

- Constructed with the package specifier and install method.
- Emits: `finished(bool)`, `error(str)`, `log(str)`.
- Uses `uv pip install` (not `uv sync`) to avoid DLL lock issues on Windows.

### Theme System

`themes.py` provides `light_stylesheet()` and `dark_stylesheet()` returning Qt CSS strings. Theme preference is persisted via the settings system (Infrastructure domain). The theme toggle button and the `_apply_theme()` method live in `main_window.py`.

### DeoVR Lens Detection

`main_window.py` contains filename-based lens profile detection (parsing tags like `_FISHEYE190`, `_MKX200`, etc.). This is UI logic because it determines what the UI displays in the projection dropdown, but the actual projection handling happens in the pipeline/ffmpeg domains.

### Internal Details (Free to Change)

Widget layout, button placement, preview rendering, drag-and-drop handling, keyboard shortcuts, status bar formatting — all internal to the UI domain. Change freely as long as the worker interface and settings keys remain stable.

### Tests

The UI currently has no dedicated test suite (Qt widget testing is complex). UI changes should be verified manually. If adding testable logic to the UI, consider extracting it into a pure function that can be tested without Qt.


---

## Chapter 5: CLI & Batch

### Ownership

This domain owns: `cli/queue_runner.py`, `cli/encode_runner.py`, and the entry points defined in `pyproject.toml` (`AlphaPass-queue`, `AlphaPass-encode`).

### Interface Contract — Queue Runner

`queue_runner.py` provides headless batch processing. It reads a queue file (JSON), processes each video through the pipeline, and writes results. It uses the same `PipelineRunner` and `create_processor` as the GUI — it's an alternative frontend, not a separate pipeline.

### Interface Contract — Encode Runner

`encode_runner.py` provides headless AV1 re-encoding. Commands:

- `build --dir <path>` — scans for video files, checks codecs via ffprobe, writes an encode queue JSON.
- `run [--now] [--crf N]` — processes the queue, encoding to AV1. Respects a time window (02:00–12:00) unless `--now` is passed.
- `status` — reports queue progress.

The encoder writes to a temp file alongside the original, then atomically replaces on success. This is a safety-critical pattern — do not change it to in-place encoding.

### Internal Details (Free to Change)

Queue file format, scheduling logic, progress display, file scanning patterns — all internal. The only hard dependency is on `utils/ffmpeg.py` for probing and encoding.


---

## Chapter 6: Infrastructure

### Ownership

This domain owns: `utils/gpu.py`, `utils/bootstrap.py`, `utils/settings.py`, `utils/sbs.py`, `utils/masks.py`.

### Interface Contract — GPU Detection

`gpu.py` exposes:

- `get_device() -> torch.device` — returns the best available device (cuda, mps, xpu, or cpu).
- `get_device_info() -> dict` — returns device name, VRAM, compute capability for display.
- `auto_config(vram_mb: int) -> dict` — returns recommended pipeline settings (max_matting_pixels, mem_frames, downsample_ratio) based on VRAM tier.

**VRAM tiers:** 24 GB+ → full res, 16 GB → 1080p, 12 GB → 810p, 8 GB → 720p, <8 GB → 540p.

### Interface Contract — Settings

`settings.py` exposes:

- `load_settings() -> dict` — loads from the platform-appropriate config path.
- `save_settings(settings: dict)` — persists to the same path.
- Settings path: `~/.config/AlphaPass/settings.json` (Linux/macOS), `%APPDATA%/AlphaPass/settings.json` (Windows).

The settings schema is implicitly defined by what the UI and pipeline write into it. Adding new keys is safe. Removing or renaming existing keys requires coordination with the UI domain (which reads them on startup).

### Interface Contract — SBS Utilities

`sbs.py` exposes:

- `is_sbs(frame: np.ndarray) -> bool` — detects side-by-side stereo.
- `split_sbs(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — splits into left/right eyes.
- `merge_sbs(left: np.ndarray, right: np.ndarray) -> np.ndarray` — merges back.
- `split_matte_sbs(matte, left_w, right_w) -> tuple` — splits a matte for independent eye processing.

### Interface Contract — Bootstrap

`bootstrap.py` exposes `ensure_correct_torch()` which is called once at application startup, before any torch imports. It detects the GPU via `nvidia-smi`, checks if the installed PyTorch matches the required CUDA version, and reinstalls if necessary. This function must run before anything else imports torch — it's called in `main.py` before the UI or pipeline modules are loaded.

### Interface Contract — Masks

`masks.py` handles automatic download of the DeoVR `mask8k.png` file on first use. Exposes `get_mask_path() -> Path` which returns the cached mask location, downloading if needed.

### Tests

SBS utilities are tested in `tests/test_sbs.py` (14 tests). GPU and bootstrap logic is tested implicitly through the matte protocol tests (which mock device detection).


---

## Appendix A: Adding a New Matting Model

This is a cross-domain task that touches Pipeline Core (new processor), UI (combo box entry), and potentially Infrastructure (GPU memory tuning). Follow this sequence:

1. Create `src/AlphaPass/pipeline/your_model.py` — implement the `MatteProcessor` protocol.
2. Add the variant to `create_processor()` factory in `matte.py`.
3. Add the model name to the UI combo box in `main_window.py`.
4. Write tests in `tests/test_your_model.py`.
5. If the model has unusual VRAM requirements, update `auto_config()` tiers in `gpu.py`.
6. Update this handbook's Pipeline Core chapter if the new model introduces new interface concepts.


---

## Appendix B: File Map

For quick reference, here's which domain owns which file:

| File | Domain |
|------|--------|
| `pipeline/matte.py` | Pipeline Core |
| `pipeline/rvm.py` | Pipeline Core |
| `pipeline/matanyone2.py` | Pipeline Core |
| `pipeline/sam2_masks.py` | Pipeline Core |
| `pipeline/scene_detect.py` | Pipeline Core |
| `pipeline/scaler.py` | Pipeline Core |
| `pipeline/checkpoint.py` | Pipeline Core |
| `pipeline/runner.py` | Pipeline Core (stages 1–3), Encoding (stages 4–5) |
| `utils/ffmpeg.py` | Encoding & FFmpeg |
| `cli/queue_runner.py` | CLI & Batch |
| `cli/encode_runner.py` | CLI & Batch |
| `ui/main_window.py` | User Interface |
| `ui/preview.py` | User Interface |
| `ui/themes.py` | User Interface |
| `ui/worker.py` | User Interface |
| `utils/gpu.py` | Infrastructure |
| `utils/bootstrap.py` | Infrastructure |
| `utils/settings.py` | Infrastructure |
| `utils/sbs.py` | Infrastructure |
| `utils/masks.py` | Infrastructure |
| `main.py` | Infrastructure (entry point) |
