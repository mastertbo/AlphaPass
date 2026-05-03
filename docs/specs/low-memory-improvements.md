# Feature Spec: Low-Memory Improvements

**Status:** Proposed
**Owner:** Pipeline Core + Infrastructure (joint)
**Branch:** `claude/cloud-gpu-evaluation-66L1t`
**Related:** `ARCHITECT.md` Chapter 1 (Foundation), `pipeline/runner.py`, `utils/gpu.py`

---

## 1. Background

AlphaPass already implements a comprehensive multi-layered VRAM strategy:
chunked processing, FP16 throughout, expandable CUDA segments, dynamic frame
downscaling (`pipeline/scaler.py`), per-VRAM-tier auto-configuration in
`utils/gpu.py:76-143`, and `torch.cuda.empty_cache()` between chunks
(`pipeline/runner.py:981`).

Despite this, three classes of users still hit OOM or near-OOM conditions:

1. **8K stereo content on 16 GB GPUs** (e.g. RTX 5080) where the MatAnyone2
   memory bank can grow unbounded on long shots.
2. **Sub-8 GB GPUs** (laptop 4060, 3060, 2070) where even the smallest auto
   tier leaves little headroom for other applications.
3. **Long single-shot videos** (>10 minutes without scene cuts) where
   long-term memory consolidation alone is insufficient.

This spec proposes three additive features and a set of configuration
refinements to push AlphaPass's minimum supported VRAM down and improve
robustness on long shots.

## 2. Goals & Non-Goals

### Goals

- G1. Allow 8 GB GPUs to process 1080p content without manual tuning.
- G2. Allow 16 GB GPUs to process 8K stereo without OOM on shots >10 min.
- G3. Detect OOM at runtime and recover automatically by stepping the user
  down one VRAM tier mid-job, without losing checkpoint progress.
- G4. Surface clear, actionable feedback in the UI when VRAM is the bottleneck.

### Non-Goals

- INT4 / sub-8-bit weight quantization. Quality regressions for matting are
  not yet characterised and require a separate research task.
- Multi-GPU sharding. Out of scope for the desktop product.
- Cloud GPU offload. Evaluated separately and rejected for typical AlphaPass
  workloads (file sizes dominate transfer time).
- Changes to the `MatteProcessor` protocol. All changes stay behind existing
  domain interfaces.

## 3. Proposed Features

### 3.1 Feature A — Model CPU Offload Between Chunks

**Domain:** Pipeline Core
**Files:** `pipeline/runner.py`, `pipeline/rvm.py`, `pipeline/matanyone2.py`,
`pipeline/matte.py`

**Problem.** Today the matting model lives on the GPU for the full job. On
sub-8 GB GPUs this leaves no VRAM for ffmpeg's NVENC / NVDEC, browser GPU
compositors, or the user's Qt preview pane.

**Design.**

1. Extend the `MatteProcessor` protocol with two optional methods:
   - `offload_to_cpu() -> None`
   - `reload_to_device() -> None`

   Default implementations on the abstract base move `self.model` between
   `cpu` and `self.device` and call `torch.cuda.empty_cache()` after offload.

2. Add `PipelineConfig.cpu_offload_between_chunks: bool = False`.

3. In `runner.py` chunk boundary (around line 981, after the existing
   `empty_cache` block), if offload is enabled:
   - Call `processor.offload_to_cpu()` before the next chunk's frame
     extraction completes.
   - Call `processor.reload_to_device()` immediately before the next
     `_run_matte_loop` invocation.

4. Pin offload buffers via `pin_memory=True` so the round-trip is
   bandwidth-bound, not allocator-bound.

**Cost.** ~0.5–1.5 s per chunk transition on PCIe 4.0 x16 for the RVM model
(~120 MB FP16) and ~3–5 s for MatAnyone2 (~700 MB FP16). At the default 500
frame chunk size this is <1% overhead.

**Auto-enable rule.** `auto_configure_gpu()` flips this flag on automatically
when `vram_gb < 7` or when the user explicitly selects "Conservative VRAM"
mode (see §3.4).

### 3.2 Feature B — INT8 Dynamic Quantization (RVM only, opt-in)

**Domain:** Pipeline Core
**Files:** `pipeline/rvm.py`

**Problem.** RVM is the smallest matting model but is the only one that runs
on CPU-only fallback paths. INT8 dynamic quantization halves its memory
footprint and roughly doubles CPU throughput, with no measurable quality
loss on alpha mattes per the upstream RVM paper (§4.3).

**Design.**

1. Add `RVMProcessor.__init__(..., quantize_int8: bool = False)`.
2. When enabled and `device.type == "cpu"`, wrap the loaded model in
   `torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear,
   torch.nn.Conv2d}, dtype=torch.qint8)` after `torch.jit.freeze`.
3. Mutually exclusive with `use_fp16` (raises `ValueError` if both true);
   FP16 already wins on CUDA.
4. Surface as a setting only when no CUDA device is detected at startup.

**Out of scope for this spec.** INT8 *static* quantization (requires
calibration data) and INT8 on CUDA (requires TensorRT — separate effort).

### 3.3 Feature C — Adaptive OOM Recovery

**Domain:** Pipeline Core + Infrastructure
**Files:** `pipeline/runner.py`, `utils/gpu.py`, `ui/worker.py`

**Problem.** Today an OOM is fatal: the chunk dies, the worker thread
surfaces a `RuntimeError`, and the user manually lowers settings and
restarts. The checkpoint system saves the *last completed* chunk, so the
user does not lose all progress, but they do lose the partial chunk.

**Design.**

1. Wrap `_run_matte_loop` in a `try/except torch.cuda.OutOfMemoryError`.
2. On OOM:
   - Discard the partial chunk's outputs (do not write a checkpoint).
   - Call `torch.cuda.empty_cache()` and `gc.collect()`.
   - Step the active config down one tier in `auto_configure_gpu()`'s
     ladder (introduce a `step_down_tier(current_cfg) -> cfg | None` helper).
   - Reinstantiate the processor with the new tier's `ma2_internal_size`,
     `ma2_mem_frames`, and `downsample_ratio`.
   - Retry the same chunk once. If it OOMs again, step down another tier.
   - If the lowest tier OOMs, fail with a clear message naming the tier and
     the chunk index.
3. Emit a `pipeline_warning` signal to `ui/worker.py` so the UI can show
   "VRAM tight — stepped down to 720p matting" inline.
4. Persist the stepped-down tier in the checkpoint so resumes start at the
   reduced tier rather than re-discovering the OOM.

**Why this is safe.** Stepping the tier only changes matting-side
parameters. Frame extraction (`utils/ffmpeg.py`), encoding
(`cli/encode_runner.py`), and stereo handling (`utils/sbs.py`) are
untouched. The MatteProcessor protocol guarantees that any tier produces
a valid alpha matte.

### 3.4 Feature D — UI: Conservative VRAM Mode + Diagnostics

**Domain:** User Interface
**Files:** `ui/main_window.py`, `ui/worker.py`, `utils/settings.py`

1. Add a "Conservative VRAM" checkbox to the advanced settings group. When
   enabled it forces:
   - One tier below the auto-detected tier.
   - `cpu_offload_between_chunks=True`.
   - `max_mem_frames` clamped to `min(2, auto_value)`.
2. Add a peak-VRAM read-out to the progress pane that polls
   `torch.cuda.max_memory_allocated()` once per chunk and displays the
   high-water mark alongside total VRAM.
3. When OOM recovery (§3.3) kicks in, surface the tier change as an inline
   warning and log it to the existing CHANGELOG-style session log.

## 4. Configuration Changes

`utils/gpu.py:76-143` is extended with an additional `<7 GB` tier
breakdown so very-low-VRAM laptops (4 GB GTX 1650, 6 GB 3050) get
sensible defaults:

| VRAM | max_matting_pixels | internal | mem_frames | downsample | offload |
|------|--------------------|----------|------------|------------|---------|
| ≥23 GB | unlimited | 480 | 5 | 0.25 | off |
| ≥15 GB | 1920×1080 | 480 | 3 | 0.25 | off |
| ≥11 GB | 1440×810 | 360 | 2 | 0.125 | off |
| ≥7 GB | 1280×720 | 320 | 2 | 0.10 | off |
| ≥5 GB | 1280×720 | 240 | 1 | 0.10 | **on** |
| <5 GB | 960×540 | 192 | 1 | 0.10 | **on** |

The 16 GB row (covers RTX 5080) is also tightened: when the input is 8K
SBS stereo, `mem_frames` drops from 3 to 2 to avoid the long-shot growth
case from §1.

## 5. Settings Schema Additions

`~/.config/AlphaPass/settings.json` gains:

```json
{
  "memory": {
    "cpu_offload_between_chunks": false,
    "conservative_vram_mode": false,
    "rvm_int8_cpu": false,
    "oom_recovery_enabled": true
  }
}
```

`utils/settings.py` reads with defaults; missing keys are not an error.

## 6. Testing Plan

### Unit
- `tests/pipeline/test_offload.py` — assert model parameters are on `cpu`
  after `offload_to_cpu()` and back on `device` after `reload_to_device()`,
  no allocator leak across 50 cycles.
- `tests/utils/test_gpu_tiers.py` — assert `step_down_tier()` returns
  monotonically smaller configs and `None` at the floor.
- `tests/pipeline/test_oom_recovery.py` — inject a simulated
  `OutOfMemoryError` from a fake processor on the first chunk and assert
  the runner retries at a lower tier.

### Integration
- Run the existing 1080p smoke clip on a CUDA_VISIBLE_DEVICES-restricted
  context that caps VRAM to 6 GB (use `PYTORCH_CUDA_ALLOC_CONF` with a
  hard limit) and assert completion.
- Run an 8K SBS clip on a 16 GB GPU through a 12-minute single-shot scene
  (no scene cuts) and assert no OOM and no quality regression vs. baseline.

### Manual
- Toggle "Conservative VRAM" mid-queue, confirm the next job picks up the
  setting.
- Force OOM by setting `mem_frames=10` on a 6 GB GPU; verify the recovery
  message appears in the UI within 30 seconds.

## 7. Rollout

1. Land Feature A (CPU offload) and Feature D (UI) together — these are the
   highest-impact, lowest-risk changes.
2. Land Feature C (OOM recovery) once Feature A is stable; recovery depends
   on processor reload working reliably.
3. Land Feature B (INT8) last, gated to CPU-only environments.

Each feature is independently revertable. No migration is required for
existing checkpoints — the new tier field defaults to the auto-detected
tier on read.

## 8. Open Questions

- **Q1.** Should the OOM recovery tier-step also reduce `chunk_size`, or
  only the matting-side parameters? Smaller chunks reduce peak intermediate
  memory but increase the number of model reloads. Recommend: leave
  `chunk_size` unchanged in v1, revisit if recovery still fails on the
  lowest tier.
- **Q2.** SAM2 (`pipeline/sam2_masks.py`) is one-shot per video and is not
  affected by chunk-boundary offload. Should it get its own pre/post-call
  CPU offload? Probably yes for <8 GB GPUs, but defer to a follow-up.
- **Q3.** The peak-VRAM read-out in §3.4 may interact with NVML on Windows
  laptops with hybrid graphics. Validate on at least one Optimus laptop
  before shipping.

## 9. Out of Scope (Future Work)

- INT4 weight quantization (bitsandbytes / GPTQ-style)
- TensorRT INT8 path for RVM and MatAnyone2 on CUDA
- Multi-GPU sharding for ≥4K stereo
- Streaming inference (frame-by-frame without checkpointing) for live
  preview at lower latency
