# AlphaPass — Architecture

This document explains *why* the code is structured the way it is. It is aimed at contributors and maintainers. For user-facing feature documentation see the [README](README.md); for environment setup and common tasks see [docs/onboarding.md](docs/onboarding.md).

---

## Overview

AlphaPass has three execution paths that share the same core pipeline:

| Entry point | Launched by | Use case |
|---|---|---|
| `AlphaPass` | `uv run AlphaPass` | Interactive GUI |
| `AlphaPass-queue` | crontab / Task Scheduler | Overnight batch matting |
| `AlphaPass-encode` | crontab / Task Scheduler | Overnight AV1 re-encode |

All three paths use the same `Pipeline` class (`pipeline/runner.py`) and `PipelineConfig` dataclass. The CLI tools read their settings from the same `~/.config/AlphaPass/settings.json` that the GUI writes, so users configure via the GUI and run unattended from the CLI.

---

## Module Map

```
src/vrautomatte/
├── main.py                # bootstrap → Qt app
├── cli/
│   ├── queue_runner.py    # AlphaPass-queue subcommands (build/run/status)
│   └── encode_runner.py   # AlphaPass-encode subcommands (build/run/status)
├── pipeline/
│   ├── runner.py          # Pipeline, PipelineConfig, PipelineProgress
│   ├── matte.py           # MatteProcessor protocol + factory + wrappers
│   ├── rvm.py             # RVM processor (mobilenetv3 / resnet50)
│   ├── matanyone2.py      # MatAnyone 2 processor (experimental)
│   ├── sam2_masks.py      # SAM2 one-shot body mask + POV heuristics
│   ├── scene_detect.py    # Histogram scene change detector
│   ├── scaler.py          # LANCZOS frame downscaler for VRAM budgets
│   └── checkpoint.py      # Resumable pipeline checkpoints
├── ui/
│   ├── main_window.py     # Main window; DeoVR lens tag detection
│   ├── preview.py         # Dual-pane preview widget + scrubber
│   ├── themes.py          # Light / dark stylesheets
│   └── worker.py          # PipelineWorker + InstallWorker (QThread)
└── utils/
    ├── bootstrap.py       # GPU detect + CUDA wheel install (pre-torch)
    ├── ffmpeg.py          # FFmpeg wrappers (extract, fisheye, alpha pack)
    ├── gpu.py             # Device detection + VRAM-tier auto-config
    ├── masks.py           # DeoVR mask8k.png download + cache
    ├── sbs.py             # SBS stereo split / merge / detect
    └── settings.py        # JSON settings persistence
```

---

## Bootstrap and PyTorch Installation

**File:** `utils/bootstrap.py` — `ensure_correct_torch()`

### The problem

`uv sync` resolves PyTorch from PyPI, which only hosts the CPU-only wheel. CUDA wheels live on a separate index (`download.pytorch.org/whl/cu128`). Even with the index configured in `pyproject.toml`, a first-run or environment migration can end up with the wrong wheel.

On RTX 50xx (Blackwell) hardware this is especially critical: cu126 wheels recognise the GPU but cannot execute kernels on sm_120, causing silent CPU fallback or crashes.

### The mechanism

1. Run `nvidia-smi` to determine driver CUDA version
2. Check the installed torch local-version tag (e.g. `+cu128`, `+cpu`)
3. If mismatch: run `uv pip install torch torchvision --index pytorch-cu128`
4. **Re-exec the entire process** via `os.execv` — Python cannot unload already-imported C extensions, so a clean process is the only reliable way to switch the active wheel
5. Set env var `AlphaPass_TORCH_OK=1` before re-exec so the restarted process skips the check

Bootstrap runs at the very top of `main.py`, before any `import torch`.

---

## The MatteProcessor Protocol

**File:** `pipeline/matte.py`

### Why a Protocol, not an ABC

Structural subtyping (PEP 544) means any object that implements `process_frame()`, `reset()`, and `cleanup()` is a valid `MatteProcessor`, without inheriting from a base class. This matters in two ways:

- **Tests** can use plain `MagicMock` without subclassing — no `spec=` gymnastics required
- **Third-party backends** can conform to the interface without importing from this package at all

### Compositor pattern

Both `AlphaSmoother` and `POVExclusionProcessor` are wrappers that implement the same `MatteProcessor` protocol. They can be stacked transparently:

```
POVExclusionProcessor(
    AlphaSmoother(
        RVMProcessor(...)
    )
)
```

The pipeline runner never calls a specific processor class directly — it receives a `MatteProcessor` and calls `process_frame()`. The compositor structure is assembled once in `create_processor()`.

### Adding a new backend

Add a module, add one branch to `create_processor()`. No other code changes needed. See [docs/onboarding.md](docs/onboarding.md) for a step-by-step walkthrough.

---

## Chunked Pipeline Design

**File:** `pipeline/runner.py`

### The problem

A 161,000-frame 8K video produces approximately 240 GB of PNGs if extracted all at once before any matting begins. That is larger than most consumer disks.

### The solution

Process in chunks of N frames (default 500):

```
for each chunk:
    extract N frames (ffmpeg keyframe seek)
    matte each frame (delete source PNG immediately after)
    flush segment video (delete matte PNGs after encoding)
    save checkpoint
concat all segments → final matte
```

Peak disk usage is bounded to roughly `chunk_size × one_PNG_size + accumulated_segment_videos`.

### The keyframe seek trade-off

ffmpeg's `-ss before -i` (fast seek) lands on the nearest keyframe, which introduces ~1–2 frame boundary imprecision between chunks. The alternative, `-ss after -i` (exact seek), decodes from the previous keyframe and is significantly slower for large chunks.

For this application the imprecision is acceptable: temporal coherence across chunk boundaries comes from the matting model's recurrent hidden state, not from perfect frame alignment.

---

## Checkpoint and Resume

**File:** `pipeline/checkpoint.py`

### Deterministic temp directory

The temp directory is named `AlphaPass_{stem}_{config_hash[:8]}` — derived from the input filename and a hash of the processing configuration (model variant, chunk size, output format, etc.). This name is fully deterministic, so a restarted process can locate the correct temp dir without storing any path anywhere.

### Validation

Before resuming, the checkpoint validates two hashes:

- **`input_hash`**: SHA-256 of the first 64 KB of the input video. Detects if the user swapped the input file to a different video with the same filename.
- **`config_hash`**: SHA-256 of processing-relevant `PipelineConfig` fields (model, codec, format, etc.), excluding paths. Detects if the user changed settings between runs.

A hash mismatch discards the checkpoint and starts fresh, preventing silently wrong output.

### Stale cleanup

Temp dirs older than 7 days are cleaned when the pipeline starts — not on successful completion. This keeps crashed jobs resumable for a week without requiring manual cleanup.

---

## GPU Auto-Configuration

**File:** `utils/gpu.py` — `auto_configure_gpu()`

### Why it exists

This app targets 8K VR content on consumer GPUs ranging from 6 GB to 24 GB. Without auto-config, users would need to manually tune `downsample_ratio` and `max_matting_pixels` by trial and error to avoid OOM, and the correct values differ wildly across hardware.

### VRAM tiers

| VRAM | Max matting resolution | Notes |
|---|---|---|
| 24 GB+ | No limit | Full resolution |
| 16 GB | 1920 × 1080 | MatAnyone 2 internal 480px |
| 12 GB | 1440 × 810 | MatAnyone 2 internal 360px |
| 8 GB | 1280 × 720 | |
| ≤ 6 GB | 960 × 540 | |

For SBS content, thresholds apply per-eye (the frame is already halved before matting).

### "Only overrides defaults" rule

`auto_configure_gpu()` only writes fields that still hold the `PipelineConfig` default value. If a user has manually adjusted a setting in the GUI, auto-config does not override it — user intent takes precedence.

---

## CUDA Performance Flags

**File:** `utils/gpu.py` — `configure_cuda_performance()`

Three flags are set at startup on CUDA devices:

**TF32** (`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`): Enables TensorFloat-32 for matrix multiplications. TF32 has the same numeric range as FP32 but a reduced mantissa, giving ~10% throughput improvement on Ampere, Ada, and Blackwell hardware. It is safe for inference — the quality difference in alpha mattes is imperceptible.

**cuDNN benchmark** (`torch.backends.cudnn.benchmark = True`): Runs a short kernel-selection benchmark on the first forward pass and then locks in the fastest algorithm. This pays off quickly because all video frames are the same shape.

**Expandable segments** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`): The PyTorch CUDA allocator's fixed-segment strategy can produce OOM errors even when enough total VRAM is available but it is fragmented across chunks. Expandable segments reduce this fragmentation.

---

## Frame Scaling

**File:** `pipeline/scaler.py` — `FrameScaler`

The matting model operates on a downscaled copy of the frame. VRAM consumption scales with `target_pixels`, not `original_pixels`, so an 8K frame can be matted on a 6 GB GPU by scaling it to 540p first.

After matting, the grayscale alpha mask is upscaled back to the original frame resolution using LANCZOS, which is the highest-quality standard resampling filter available in Pillow. The upscale is smooth because alpha masks contain soft gradients, not hard edges.

For VR passthrough content at 90+ degree FOV, the perceptual quality impact of the downscale/upscale round-trip is negligible — the blur introduced by upscaling falls below the noise floor of the matte itself.

`FrameScaler.active` is `False` when the frame already fits within the pixel budget; in that case both `downscale()` and `upscale_matte()` are no-ops.

---

## SBS Stereo Processing

**Files:** `utils/sbs.py`, `pipeline/runner.py`

RVM is trained on conventional video. An SBS frame looks like two people standing side-by-side, and the model's recurrent hidden state would be confused by the horizontal seam between the eyes.

When SBS is active, the pipeline:

1. Splits each frame into left and right eye images (`sbs.split_frame`)
2. Runs two independent `MatteProcessor` instances, each with its own recurrent state
3. Merges the resulting mattes back to SBS layout (`sbs.merge_mattes`)

Auto-detection uses the heuristic: aspect ratio ≥ 1.9:1 (e.g. 3840 × 1920). The checkbox in the UI allows manual override for files that don't match the heuristic.

---

## POV Mode and Scene Change Detection

**Files:** `pipeline/sam2_masks.py`, `pipeline/scene_detect.py`

### Two-pass design

SAM2 mask generation is expensive — it runs a full segmentation pass on the first frame. The scene change detector is cheap — it compares frame histograms every frame. Combining them gives a practical system:

1. On frame 1: SAM2 generates all candidate masks; heuristics score them by POV likelihood (center position, area, bottom-heaviness); the highest-scoring mask becomes the POV body exclusion mask
2. Every subsequent frame: `SceneChangeDetector.check()` computes a 64-bin grayscale histogram and measures Pearson correlation against the reference histogram
3. On scene change: SAM2 regenerates the mask; detector resets with the new reference

### Why Pearson correlation

Pearson correlation on normalised histograms is illumination-invariant. A slow fade-to-white changes per-bin counts proportionally, keeping the correlation near 1.0 (no false positive). A hard cut or jump cut drops the correlation sharply (true positive). Per-pixel difference metrics would false-positive on any camera movement.

The 30-frame cooldown prevents re-triggering during multi-frame transition sequences (cross-fades, wipes) that would otherwise fire on every frame.

SAM2 is fully unloaded after mask generation via `gc.collect()` to return its VRAM for the matting model.

---

## DeoVR Alpha Packing

**File:** `utils/ffmpeg.py` — `pack_alpha()`

The DeoVR specification requires the alpha matte to be packed into the corners of the fisheye video as a red-channel overlay. The exact steps:

1. Scale the matte to 40% of the video resolution
2. Apply `colorchannelmixer` to isolate the red channel (G=0, B=0)
3. Use `colorkey` to make black pixels transparent
4. Split into 6 segments and overlay each into a specific corner area of the fisheye frame

All six steps execute in a single `ffmpeg` filter graph, avoiding an intermediate encode step.

**Why AV1:** The final alpha-packed output is always encoded as AV1 (`libsvtav1` / `av1_nvenc`). Meta Quest 3 and 3S hardware-decode AV1 natively, giving better quality per bit than HEVC at the same file size. The Codec setting in the GUI applies to intermediate fisheye conversion steps, not the final output.

---

## UI Threading Model

**File:** `ui/worker.py`

The Qt event loop is single-threaded. Running the pipeline on the main thread would freeze the UI for the entire duration of processing.

`PipelineWorker(QThread)` runs `Pipeline.run()` on a background thread and bridges results back to the main thread via Qt Signals:

- `progress(PipelineProgress)` — emitted per frame; drives the progress bar and live preview
- `finished(str)` — output path on success
- `error(str)` — error message on failure

`InstallWorker(QThread)` uses the same pattern for in-app MatAnyone 2 dependency installation: runs `uv pip install` as a subprocess, emits each output line as a `log(str)` signal, and emits `finished` or `error` on completion.

Neither worker shares mutable state with the main thread. All communication travels through signals, which PySide6 delivers safely across thread boundaries.

---

## CLI Tool Design

**Files:** `cli/queue_runner.py`, `cli/encode_runner.py`

### Why separate binaries

The GUI is unsuitable for scheduled overnight automation — it requires a display and user interaction. The CLI tools are plain Python scripts with no Qt dependency that can be called from crontab or Windows Task Scheduler.

### Time-window guard

Both tools default to running only during 02:00–12:00. The check happens at the top of `run` — if called outside the window the process exits immediately. `--now` bypasses the check for interactive or testing use.

### Fault-tolerant queue format

Each item in `queue.json` has an independent `status` field (`pending`, `done`, `error`). After each file completes, its status is updated and the file is written to disk before moving to the next. If the process is killed mid-queue, a subsequent `run` skips already-done entries and retries errored ones.

### AlphaPass-encode is NVIDIA-only

`av1_nvenc` requires a recent NVIDIA GPU. AV1 software encoding (`libsvtav1`) is fast enough for single files but impractically slow for a large library queue — a 10-minute 8K video would take hours to encode in software. The tool exits with an error if `av1_nvenc` is not available.
