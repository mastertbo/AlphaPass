# AlphaPass

Automated AI video matting for VR passthrough content. Separate people from backgrounds in VR videos and generate alpha channel mattes for DeoVR passthrough playback on Meta Quest headsets.

## Features

- **Two AI matting models** — RVM MobileNetV3 (fast), RVM ResNet50 (quality). MatAnyone 2 available as experimental option for flat (non-VR) video.
- **All-people detection** — RVM models detect every person in the frame (crowds, groups, any count)
- **Zero manual input** — no masks, trimaps, or prompts required
- **Temporal alpha smoothing** — EMA-based frame blending (off / light / medium / heavy) to reduce matte jitter
- **Chunked extraction** — processes video in chunks; source PNGs deleted as matted, matte PNGs deleted after segment encoding. Peak disk = one chunk of PNGs + accumulated segment videos.
- **Resumable pipeline** — checkpoint after each chunk; interrupted jobs resume from the last completed segment
- **GPU auto-config** — automatically adapts matting resolution and memory settings to your GPU's VRAM (24 GB → full res, 16 GB → 1080p, 12 GB → 810p, etc.)
- **Frame downscaling** — LANCZOS downscale before matting, upscale matte after; prevents OOM on 8K content with consumer GPUs
- **POV body removal** — automatically detect and exclude the camera operator's body from the matte
- **Scene change detection** — refreshes masks automatically when cuts or position changes occur
- **SBS stereo support** — auto-detects side-by-side VR videos and processes each eye independently
- **DeoVR alpha packing** — follows the official DeoVR spec: 40% scale matte, red-channel-only, 6-segment corner packing, AV1 encoding
- **Auto projection detection** — detects lens profile from filename (`_FISHEYE190`, `_MKX200`, `_VRCA220`, `_RF52`) and sets projection + FOV automatically
- **Live preview** — source frame and matte side-by-side with FPS counter, ETA, and frame scrubber (toggle on/off)
- **Batch processing** — queue multiple files with per-file projection detection; mixed equirectangular and fisheye batches work automatically
- **Drag & drop** — drop one video to set input, drop multiple to batch queue
- **NVENC + AV1 encoding** — hardware-accelerated encoding via NVENC (HEVC, H.264, AV1) with automatic CPU fallback
- **Light / dark theme** — toggle with the 🌙/☀️ button; preference is saved
- **Settings persistence** — remembers all settings across sessions
- **Cross-platform** — Windows, macOS, Linux (PySide6 GUI)
- **GPU accelerated** — NVIDIA CUDA (FP16), AMD ROCm, Apple MPS (FP16), Intel XPU, or CPU fallback

## Quick Start

### Prerequisites

- **Python 3.10+**
- **FFmpeg** on your PATH
- **GPU recommended** (any CUDA/ROCm/MPS GPU; CPU works but is much slower)

```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux (Debian/Ubuntu)
sudo apt install ffmpeg
```

### Install & Run

```bash
git clone https://github.com/SifuInTheShell/AlphaPass.git
cd AlphaPass

# Install with uv (recommended)
uv sync
uv run AlphaPass

# Or with pip
pip install -e .
AlphaPass
```

### Optional: MatAnyone 2 (experimental, non-VR only)

MatAnyone 2 requires additional dependencies. Install them with:

```bash
uv sync --extra matanyone2
```

This pulls in [MatAnyone 2](https://github.com/pq-yang/MatAnyone2) and [SAM2](https://github.com/facebookresearch/sam2). Models are downloaded automatically on first use.

**Note:** MatAnyone 2 depends on SAM2 for initial masks and does not work with fisheye or equirectangular VR content. Use for standard flat video only.

## Usage Guide

### Basic Workflow

1. **Launch** the app: ``uv run AlphaPass
2. **Load a video** — click Browse or drag a file onto the window
3. **Choose a model**:
   - `mobilenetv3` — fastest, detects all people (crowds, groups)
   - `resnet50` — better quality, detects all people
   - `MatAnyone 2 (experimental)` — sharpest edges, flat video only
4. **Choose output format**:
   - `Matte Only` — just the alpha matte video
   - `DeoVR Alpha Pack` — full passthrough pipeline for Quest headsets
5. **Click Start** — watch the live preview as it processes

### Auto-Detection

When you load a video, the app automatically detects:

| Property | How It's Detected | Fallback |
|----------|------------------|----------|
| **SBS stereo** | Aspect ratio ≥ 1.9:1 | Manual checkbox |
| **Projection** | Filename tags: `_FISHEYE`, `_MKX200`, `_VRCA220`, `_RF52` | Defaults to Equirectangular → Fisheye |
| **FOV** | Extracted from tag (e.g. `_FISHEYE190` → 190°, `_MKX200` → 200°) | FOV slider value |
| **GPU settings** | VRAM tier auto-config (resolution, downsample ratio, memory frames) | Manual override |

In batch mode, projection and FOV are detected **per file** from each filename — mixed batches of equirectangular and fisheye content process correctly.

### Output Formats

| Format | What You Get | Use Case |
|--------|-------------|----------|
| **Matte Only** | Grayscale matte video (white = person, black = background) | Compositing in editors, custom pipelines |
| **DeoVR Alpha Pack** | `*_alpha.mp4` with red-channel alpha packed into fisheye corners | Direct playback in DeoVR with passthrough |

### Model Comparison

| Model | People | Edges | Speed | GPU Memory | Best For |
|-------|--------|-------|-------|------------|----------|
| RVM MobileNetV3 | All | Good | ~50 fps | ~1 GB | Previewing, crowds, lower-end GPUs |
| RVM ResNet50 | All | Better | ~30 fps | ~2 GB | Crowds, groups, general use |
| MatAnyone 2 | Close-up | Best | ~8 fps | ~6 GB | Flat video only, 1-3 subjects |

*FPS measured at 1080p on RTX 4070. Actual performance varies by resolution and hardware.*

**Which model should I use?**
- Scenes with many people (street, event, crowd) → **resnet50** (detects every person automatically)
- Quick preview or limited GPU → **mobilenetv3**
- Flat (non-VR) video with 1-3 subjects → **MatAnyone 2** (experimental)

### POV Mode

Enable **POV mode** for first-person VR content where the camera operator's body is visible. The app uses SAM2 to detect the operator's body on the first frame and excludes it from the matte — only other people are kept.

- With **RVM**: uses static mask subtraction (fast)
- Automatically refreshes when a scene change is detected

### SBS Stereo Videos

Side-by-side stereo VR videos (aspect ratio ≥ 1.9:1) are **auto-detected**. When the SBS checkbox is active:

- Each eye is matted independently with its own processor instance
- Results are merged back to SBS format
- Progress bar shows both eye passes

You can also manually toggle SBS for files that don't match the auto-detection heuristic.

### Temporal Smoothing

The **Temporal Smoothing** control reduces frame-to-frame alpha jitter using exponential moving average (EMA):

| Setting | Weight | Effect |
|---------|--------|--------|
| Off | 1.0 | Raw matte output |
| Light | 0.85 | Subtle stabilisation |
| Medium | 0.7 | Moderate smoothing |
| Heavy | 0.5 | Strong smoothing |

Useful for RVM on VR content where edges can flicker between frames.

### Batch Processing

Process multiple videos unattended:

1. Set up your first file (input, output, settings)
2. Click **+ Queue** to add it to the batch
3. Repeat for more files, or drag multiple files onto the window
4. Click **Start** — files process sequentially

Each file gets its own projection/FOV detection from its filename. Output filenames are auto-generated with the correct DeoVR tags.

### Drag & Drop

- **Single file** → sets it as the current input
- **Multiple files** → adds all to the batch queue

Supported formats: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.wmv`

### DeoVR Alpha Pipeline

When using the **DeoVR Alpha Pack** output format, the pipeline follows the official DeoVR alpha packing spec:

**For equirectangular sources** (no fisheye tag in filename):
```
Input Video (equirectangular SBS)
  → Extract Frames → AI Matte Generation → Assemble Matte Video
  → Convert Video to Fisheye (with DeoVR mask)
  → Convert Matte to Fisheye (without mask, preserves edge data)
  → Pack Alpha: scale matte to 40%, red-channel-only,
    split into 6 segments, overlay into fisheye corner areas
  → Encode as AV1 (NVENC)
  → Output: *_FISHEYE{fov}_alpha.mp4
```

**For already-fisheye sources** (`_FISHEYE190`, `_MKX200`, etc.):
```
Input Video (fisheye SBS)
  → Extract Frames → AI Matte Generation → Assemble Matte Video
  → Pack Alpha (same 6-segment packing, no projection conversion)
  → Encode as AV1 (NVENC)
  → Output: *_{lens_tag}_alpha.mp4
```

**Supported DeoVR lens profiles:**

| Filename Tag | Lens | FOV |
|-------------|------|-----|
| `_FISHEYE` | Generic fisheye | 180° |
| `_FISHEYE190` | Canon VR lens | 190° |
| `_MKX200` | Metalenz MKX | 200° |
| `_MKX220` | Metalenz MKX | 220° |
| `_VRCA220` | VRCA lens | 220° |
| `_RF52` | Canon RF 5.2mm | 190° |

**DeoVR settings:**
- **Final output codec**: always AV1 (`libsvtav1` / `av1_nvenc`) — Meta Quest 3/3S decodes AV1 natively
- **Intermediate codec** (Codec setting): HEVC or H.264 — used for the fisheye conversion steps; does not affect the final alpha-packed file
- **CRF**: 10–30 (default 18; lower = better quality, larger file)
- **Fisheye mask**: `mask8k.png` downloaded automatically on first use

### Settings

All settings are saved automatically to `~/.config/AlphaPass/settings.json` (Linux/macOS) or `%APPDATA%/AlphaPass/settings.json` (Windows) and restored on next launch.

**Chunk size** (100 / 250 / 500 / 1000 frames) controls the peak disk footprint during processing: smaller chunks use less peak disk space and save checkpoints more frequently, but add a little per-chunk overhead. 500 is a good default for most systems.

**Auto-resume**: when enabled, each run uses a deterministic temp directory so that if the job is interrupted (crash, cancel, power loss), restarting with the same input and settings automatically picks up from the last completed segment. Uncheck to always start fresh.

## CLI Tools

In addition to the GUI, AlphaPass ships two headless command-line tools designed for scheduled overnight processing — useful when you have a large library of VR videos to process unattended.

### AlphaPass-queue — overnight batch matting

Scan a directory and build a queue of videos to matte, then process them on a schedule.

```bash
# 1. Scan a directory and write queue.json
AlphaPass-queue build --dir /path/to/videos

# 2. Process the queue (runs only during 02:00–12:00 by default)
AlphaPass-queue run

# 3. Check status
AlphaPass-queue status

# Process immediately, bypassing the time window
AlphaPass-queue run --now

# Rebuild the queue from scratch (discards existing entries)
AlphaPass-queue build --dir /path/to/videos --reset

# Use a custom queue file location
AlphaPass-queue build --dir /videos --queue /data/my-queue.json
AlphaPass-queue run --queue /data/my-queue.json
```

**How it works:**
- `build` recursively scans for `.mp4 .mkv .mov .avi .webm` files and writes a `queue.json` with one entry per video (status: `pending`)
- `run` processes each pending file using the current settings from `~/.config/AlphaPass/settings.json` — configure the GUI first, then run the queue
- Output files are written as `{stem}_xalpha{ext}` alongside the input
- Each file's status is updated atomically after completion (`done` or `error`) so a partial run can be safely resumed

**Default queue location:** `~/.config/AlphaPass/queue.json`

### AlphaPass-encode — AV1 re-encode queue

Scan a directory and re-encode videos to AV1. Useful for re-encoding a library of matte files or any other videos.

```bash
# 1. Scan a directory and write encode_queue.json
AlphaPass-encode build --dir /path/to/mattes

# 2. Process the queue (same 02:00–12:00 window)
AlphaPass-encode run

# 3. Check status
AlphaPass-encode status

# Build with a specific quality level (default CRF is 35)
AlphaPass-encode build --dir /path/to/mattes --crf 28

# Override CRF at run time
AlphaPass-encode run --crf 28 --now
```

**How it works:**
- `build` recursively scans for video files and writes an `encode_queue.json`
- `run` re-encodes each file using `av1_nvenc` (NVIDIA GPU required — AV1 software encoding is too slow for large libraries)
- Encoded files replace the originals in-place

**Default queue location:** `~/.config/AlphaPass/encode_queue.json`

### Scheduling (crontab / Task Scheduler)

Both tools are designed to be called from a nightly job. They self-limit to the 02:00–12:00 window and exit immediately if called outside it (use `--now` to bypass).

```bash
# crontab example: run at 02:00 every night
0 2 * * * /path/to/venv/bin/AlphaPass-queue run
0 3 * * * /path/to/venv/bin/AlphaPass-encode run
```

## Architecture

```
src/vrautomatte/
├── main.py                    # Entry point
├── cli/
│   ├── queue_runner.py        # A3phaPass-queue: overnight batch matting
│   └── encode_runner.py       # AlphaPass-encode: AV1 re-encode queue
├── pipeline/
│   ├── matte.py               # MatteProcessor protocol, factory, AlphaSmoother
│   ├── rvm.py                 # RVM processor (MobileNetV3 / ResNet50)
│   ├── matanyone2.py          # MatAnyone 2 processor (experimental)
│   ├── sam2_masks.py          # SAM2 mask generation + POV heuristics
│   ├── scene_detect.py        # Scene change detector (histogram correlation)
│   ├── scaler.py              # Frame downscaler for VRAM-constrained GPUs
│   ├── checkpoint.py          # Resumable pipeline checkpoints
│   └── runner.py              # Pipeline orchestrator (chunked extract → matte → pack)
├── ui/
│   ├── main_window.py         # Main GUI window + DeoVR lens detection
│   ├── preview.py             # Dual-pane preview + scrubber
│   ├── themes.py              # Light / dark theme stylesheets
│   └── worker.py              # Background processing thread
└── utils/
    ├── ffmpeg.py              # FFmpeg wrappers (extract, fisheye, alpha pack)
    ├── gpu.py                 # Device detection + GPU auto-configuration
    ├── masks.py               # DeoVR mask auto-download
    ├── sbs.py                 # SBS stereo split/merge/detection
    └── settings.py            # Settings persistence
```

### Processing Pipeline

```
Chunked Pipeline (runner.py)
  for each chunk of N frames:
    1. ffmpeg -ss <timestamp> → extract chunk (keyframe seek)
    2. FrameScaler.downscale() if frame exceeds GPU budget
    3. MatteProcessor.process_frame() per frame
       (source PNG deleted immediately after matting)
    4. FrameScaler.upscale_matte() back to original resolution
    5. Flush segment video, delete matte PNGs, save checkpoint
  concat all segments → final matte

Alpha Packing (DeoVR spec)
  1. Scale matte to 40% of video resolution
  2. Red-channel-only via colorchannelmixer
  3. colorkey black → transparent
  4. Split into 6 segments → overlay into fisheye corner areas
  5. Encode as AV1 (NVENC → CPU fallback)

MatteProcessor Protocol
├── RVMProcessor          — recurrent forward pass, detects ALL people
├── AlphaSmoother         — EMA wrapper, reduces frame-to-frame jitter
├── MatAnyone2Processor   — SAM2 masks → InferenceCore (experimental)
└── POVExclusionProcessor — wraps any processor, subtracts POV body mask

GPU Auto-Config (gpu.py)
└── VRAM tier → max_matting_pixels, mem_frames, downsample_ratio
```

### Temp Directory Layout

Each run uses a deterministic temp directory named after the input file and current settings:

```
{temp_dir}/AlphaPass_{stem}_{config_hash[:8]}/
├── frames/           # extracted source PNGs  (deleted immediately after matting)
├── mattes/           # generated alpha PNGs   (deleted after segment flush)
├── segments/         # intermediate segment videos (accumulated until concat)
├── AlphaPass.log     # detailed per-run processing log
└── checkpoint.json   # resume state (segments completed, input + config hashes)
```

Temp dirs older than 7 days are cleaned automatically when the pipeline starts. To find a log after a crash, look for `AlphaPass_*` directories in your system temp folder (`%TEMP%` on Windows, `/tmp` on Linux/macOS).

## Development

```bash
# Run tests
uv run python -m unittest discover -s tests -p "test_*.py"

# Run a specific test file
uv run python -m unittest tests/test_sbs.py

# Check syntax
uv run python -c "import ast; ast.parse(open('src/vrautomatte/ui/main_window.py').read()); print('OK')"
```

### Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_matte_protocol.py` | 17 | Processor protocol, factory, settings, CPU fallback |
| `test_sbs.py` | 14 | SBS detection, frame split/merge, matte split/merge |
| `test_scene_detect.py` | 8 | Scene change detector, cooldown, threshold, reset |
| `test_pov_mask.py` | 6 | POV body mask selection, scoring, dilation |
| `test_integration_matanyone2.py` | 5 | MatAnyone 2 processor, SAM2 masks, re-exports |

### Code Conventions

- **Logging**: `from loguru import logger` — no `print()` in committed code
- **Strings**: double quotes preferred
- **Line length**: max 100 characters
- **Imports**: stdlib → third-party → local, alphabetical within groups
- **Type hints**: required for new/modified functions
- **Tests**: `test_*.py` or `*_test.py`, using `unittest`

## Requirements

### System

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.12 |
| FFmpeg | 4.x | 7.x |
| GPU VRAM | — (CPU ok) | 6 GB+ |
| RAM | 4 GB | 16 GB |
| Disk | 500 MB (app) | + space for video processing |

### GPU Support

| Platform | Framework | Status |
|----------|-----------|--------|
| NVIDIA | CUDA | Full support |
| AMD | ROCm | Full support |
| Apple | MPS | Full support |
| Intel | XPU (oneAPI) | Full support |
| CPU | — | Fallback (slower) |

## Model Downloads

Models are downloaded automatically on first use and cached locally:

| Model | Size | Cache Location |
|-------|------|----------------|
| RVM MobileNetV3 | ~15 MB | `~/.cache/AlphaPass/models/` |
| RVM ResNet50 | ~55 MB | `~/.cache/AlphaPass/models/` |
| MatAnyone 2 | ~2 GB | Managed by HuggingFace Hub |
| SAM2 | ~400 MB | Managed by HuggingFace Hub |
| DeoVR mask8k.png | ~2 MB | `~/.cache/AlphaPass/masks/` |

## License

[MIT](LICENSE)

## Acknowledgements

- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) — recurrent matting architecture
- [MatAnyone 2](https://github.com/pq-yang/MatAnyone2) — CVPR 2026 SOTA video matting
- [SAM2](https://github.com/facebookresearch/sam2) — Segment Anything Model 2 for automatic mask generation
- [DeoVR](https://deovr.com/) — VR video player with alpha passthrough support
