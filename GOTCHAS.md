# AlphaPass — Gotchas & Hard-Won Lessons

These are insights from past debugging sessions. Read these before working on the relevant areas — they'll save you hours.

---

## Windows subprocess pipes

**Problem:** Long-running ffmpeg processes can deadlock when reading stderr.

**Root causes:**
- **Avoid `subprocess.PIPE`** for long-running ffmpeg processes — use `DEVNULL` + directory polling instead
- `BufferedReader.read(n)` blocks until exactly n bytes arrive (not partial reads like Unix)
- ffmpeg stderr uses `\r` not `\n` for progress — line-based drain threads never yield
- Cross-pipe deadlocks happen around ~64KB of stderr (~977 frames × 67 bytes)

**Solution:** Use `subprocess.Popen(..., stdout=DEVNULL, stderr=DEVNULL)` and poll the output directory with `os.listdir()` instead of reading pipes.

**Files involved:** `utils/ffmpeg.py`, `pipeline/runner.py`

---

## ffmpeg select filter

**Problem:** The `select='between(n,start,end)'` filter decodes more frames than expected, causing silent delays.

**Root cause:** ffmpeg's select filter only outputs the selected range, but it still decodes ALL frames in the video to determine which ones to select. For a 161k-frame video with 1000 selected frames, ffmpeg will decode the entire video (~89 minutes) even though it only outputs 1000 PNGs.

**Solution:** Always pair `select='between(...)'` with `-frames:v <count>` to stop ffmpeg after outputting the desired number of frames.

```bash
# ❌ WRONG: will decode entire video
ffmpeg -i input.mp4 -vf "select='between(n,start,end)'" frame_%04d.png

# ✅ CORRECT: stops after N frames
ffmpeg -i input.mp4 -vf "select='between(n,start,end)'" -frames:v 1000 frame_%04d.png
```

**Files involved:** `utils/ffmpeg.py`, `pipeline/runner.py`

---

## PyTorch CUDA on Windows

**Problem:** PyTorch wheel selection doesn't automatically match newer GPU architectures.

**Context:**
- RTX 50xx (Blackwell) requires cu128 minimum — cu126 detects GPU but can't run on sm_120
- `uv sync` overwrites manually installed torch — must configure index in `pyproject.toml`
- Bootstrap module runs before torch imports, uses env var `AlphaPass_TORCH_OK` to skip check on restart

**Solutions:**
1. **Configure CUDA index in `pyproject.toml`:**
   ```toml
   [[tool.uv.index]]
   name = "torch"
   url = "https://download.pytorch.org/whl/cu128"
   priority = "primary"
   ```

2. **Use bootstrap module** (`utils/bootstrap.py`) as safety net:
   - Detects GPU via nvidia-smi before torch imports
   - Installs correct CUDA wheel if mismatch detected
   - Sets `AlphaPass_TORCH_OK` env var to skip check on restart

**Files involved:** `utils/bootstrap.py`, `utils/gpu.py`, `pyproject.toml`

---

## GPU Memory Management

**Problem:** Large 8K VR videos can exhaust VRAM during matting.

**Solution:** Use frame downscaling before matting, LANCZOS upscale after.

**Implementation:** `pipeline/scaler.py` — FrameScaler handles scaling decisions:
- No-op if frame fits within max_pixels budget
- Tiers matched to GPU VRAM: 24GB (no limit), 16GB (1080p), 12GB (810p), 8GB (720p), <=6GB (540p)
- For VR passthrough at 90+ FOV, quality impact is imperceptible

**Files involved:** `pipeline/scaler.py`, `utils/gpu.py`

---

## Checkpoint/Resumption State

**Problem:** If a run crashes partway through, restarting can lead to inconsistent state or wasted re-processing.

**Solution:** Deterministic checkpoint system with validation.

**How it works:**
- Checkpoint saved as JSON after each segment flush
- Temp dir: `AlphaPass_{stem}_{config_hash[:8]}/` (deterministic)
- Validates input_hash (first 64KB) + config_hash — stale checkpoint = fresh start
- Stale temp dirs (>7 days old) cleaned on pipeline start

**Files involved:** `pipeline/checkpoint.py`, `pipeline/runner.py`

---

## Tips by Domain

### Pipeline Core
- Always use MatteProcessor protocol — don't import matte engines directly
- Chunked pipeline saves disk; use `checkpoint.py` for resumability

### Encoding & FFmpeg
- Never use `subprocess.PIPE` for ffmpeg (see Windows subprocess pipes above)
- Always pair select filters with `-frames:v`
- Test codec fallback chain on target hardware

### User Interface
- Worker threads (QThread) must not block the event loop
- Settings persist to `~/.config/AlphaPass/settings.json` — validate before use
- Preview updates use directory polling, not file watchers

### Infrastructure
- GPU detection runs at startup via `utils/gpu.py:get_device()`
- Bootstrap runs BEFORE torch imports — don't import torch in bootstrap.py
- SBS format halves horizontal resolution — adjust GPU tiers accordingly

### CLI & Batch
- Queue runner is headless — log everything, no UI feedback
- Encode runner inherits pipeline and ffmpeg logic — test against chunked pipeline
