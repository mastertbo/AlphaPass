# Pipeline Core Domain — Technical Analysis & Architecture

**Domain Owner:** Video Matting Pipeline  
**Key Files:** `pipeline/*.py`  
**Status:** Production-ready with advanced features  
**Last Updated:** 2026-04-11

---

## Executive Summary

The **Pipeline Core** domain orchestrates the entire video-to-alpha-matte transformation. It uses a chunked, resumable architecture that balances VRAM efficiency with processing speed, supports multiple matting backends (RVM + MatAnyone2), and handles advanced VR-specific features (POV mode, SBS stereo, fisheye conversion).

**Core Strengths:**
- Pluggable MatteProcessor protocol enables seamless backend switching
- Chunked pipeline with checkpoint-based resumption for long jobs
- Adaptive frame scaling prevents GPU OOM on high-resolution content
- Scene change detection for dynamic POV mask refresh
- Temporal smoothing reduces frame-to-frame alpha jitter

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator                        │
│                    (runner.py: Pipeline class)                   │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Frame Extraction (chunked)                            │
│  ├─ Keyframe seek (ffmpeg -ss) for fast chunk boundaries       │
│  └─ Directory polling for progress tracking                     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Matte Generation (per-chunk loop)                    │
│  ├─ MatteProcessor (RVM, MatAnyone2, or wrapped variants)      │
│  ├─ Optional: Alpha Temporal Smoothing                          │
│  ├─ Optional: POV Exclusion (SAM2-based mask refresh)           │
│  ├─ Optional: Frame Scaling for VRAM constraints                │
│  └─ Checkpoint save after chunk flush                           │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Video Assembly & Encoding                            │
│  └─ Re-encode matte frames with codec/CRF settings             │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Post-Processing (optional)                           │
│  ├─ Equirectangular → Fisheye conversion                        │
│  └─ DeoVR alpha packing                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input Video
    ↓
[Config] → GPU Auto-Config → Temp Dir Setup
    ↓
CHUNK LOOP (resume-aware):
  ├─ Extract N frames (ffmpeg with keyframe seek)
  ├─ Load MatteProcessor (backend-specific)
  ├─ Per-frame processing:
  │   ├─ Frame Scaler: downscale if VRAM budget exceeded
  │   ├─ MatteProcessor.process_frame(rgb) → matte
  │   ├─ Scene Detector: check for POV mask refresh
  │   ├─ Alpha Smoother: temporal blending
  │   └─ Save matte PNG to temp dir
  ├─ Checkpoint: save progress
  └─ Repeat until all frames processed
    ↓
Matte Video Assembly (ffmpeg encode)
    ↓
[Optional] Fisheye Conversion / DeoVR Packing
    ↓
Output Video
```

---

## Core Components

### 1. **Pipeline Class** (`runner.py`)

**Role:** Central orchestrator — manages all stages, checkpointing, and progress reporting.

**Key Attributes:**
- `config: PipelineConfig` — all pipeline settings (codec, model, VRAM limits, etc.)
- `on_progress: Callable` — callback for UI updates
- `_cancelled: bool` — supports mid-run cancellation

**Core Methods:**

| Method | Purpose |
|--------|---------|
| `run()` | Main entry point; executes all stages sequentially |
| `_extract_chunk()` | Extract N frames using ffmpeg keyframe seek + polling |
| `_apply_gpu_config()` | Auto-tune model params based on VRAM (tier 1–5) |
| `_setup_temp_dir()` | Create/locate resumable temp with deterministic naming |
| `_process_chunk()` | Load processor, per-frame loop, checkpoint save |

**Chunked Processing Loop:**
1. Extract frames to temp PNG directory (no intermediate disk buffering)
2. Load matte processor (weights cached after first load)
3. Process frames one-by-one, yielding to check cancellation
4. Save matte PNGs alongside source frames
5. After chunk completes: flush temp directory, save checkpoint
6. Repeat until `total_frames` reached

**Checkpoint Integration:**
- After each segment flush, `PipelineCheckpoint` saved to temp dir
- On restart, checkpoint validates against input hash + config hash
- If valid, resumes from `completed_frames`; if stale, starts fresh
- Ensures recovery from crashes without re-processing

---

### 2. **MatteProcessor Protocol** (`matte.py`)

**Design Pattern:** Runtime-checkable Protocol (PEP 544) for duck typing.

```python
@runtime_checkable
class MatteProcessor(Protocol):
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """RGB (H,W,3) uint8 → alpha matte (H,W) uint8"""
    def reset(self) -> None:
        """Reset state between videos"""
    def cleanup(self) -> None:
        """Release GPU memory"""
```

**Benefits:**
- New matting backends require only these 3 methods
- No inheritance overhead; structural typing via duck typing
- Pipeline treats all processors identically

**Factory:** `create_processor(variant, device, ...)` handles instantiation and optional wrapping.

---

### 3. **RVM Processor** (`rvm.py`)

**Backend:** Robust Video Matting (PeterL1n/RobustVideoMatting)

**Model Types:**
- `mobilenetv3` (light, ~144 MB)
- `resnet50` (heavy, ~180 MB)

**Precision:**
- **FP16 TorchScript** (default): 50% VRAM reduction, ~172 FPS on RTX 3090
- **FP32 TorchScript** fallback for non-CUDA devices

**Recurrent State:**
- `self.rec = [None] * 4` stores hidden states for temporal consistency
- Per-frame forward pass: `fgr, pha, *self.rec = model(src, *self.rec, downsample_ratio)`
- Resets between videos to prevent state leakage

**Performance Optimizations:**
- `torch.jit.freeze()` after load for BatchNorm fusion + dead-code elimination
- `configure_cuda_performance()` enables TF32 + cuDNN benchmark
- Direct `torch.from_numpy()` bypass (avoids PIL round-trip)
- Downsampling ratio defaults to `0.125` for 4K content

**Key Parameters:**
- `downsample_ratio`: 0.25 (HD), 0.125 (4K), adjust lower for OOM
- `use_fp16`: Enabled on CUDA/MPS, disabled on CPU

---

### 4. **MatAnyone2 Processor** (`matanyone2.py`)

**Backend:** MatAnyone 2 memory-based matting (D003 architecture)

**Distinguishing Feature:** Requires first-frame mask (generated by SAM2).

**Architecture:**
- **InferenceCore**: Manages XMem long-term memory + working-memory slots
- **First-frame init**: SAM2 mask → InferenceCore memory → model.step()
- **Per-frame**: Only RGB input; model uses learned memory for temporal context

**Memory Management:**
- `max_mem_frames`: Working-memory capacity (default 3); older frames moved to long-term
- `use_long_term`: XMem potentiation prevents unbounded VRAM growth on long videos
- `max_internal_size`: Resolution (px) at which frames are encoded (default 480 for 4K+)

**Precision & Compilation:**
- **FP16**: Halves activation memory (~50% VRAM), enabled on CUDA/MPS
- **torch.compile**: CUDA-graph-replay mode; ~30s startup, ~15-30% per-frame speedup
- **Channels-last**: Memory format optimization on CUDA (`.to(memory_format=torch.channels_last)`)

**Automatic Mask Regeneration (POV Mode):**
- `_scene_detector: SceneChangeDetector` monitors histogram correlation
- On drastic scene change: invoke SAM2, call `_reinit_with_mask()` to reset InferenceCore memory
- Ensures POV body mask stays accurate through position changes

---

### 5. **SAM2 Mask Generation** (`sam2_masks.py`)

**Role:** Auto-generate first-frame segmentation masks for MatAnyone2, or POV-specific masks.

**Variants by Device:**
- CUDA: `sam2-hiera-small`
- MPS: `sam2-hiera-small`
- CPU: `sam2-hiera-tiny`

**Mask Selection Heuristics:**

#### Non-POV Subject (MatAnyone2 standard mode)
Scores masks by person-likeness:
- **Aspect ratio** (~tall): +0.15
- **Area** (3%–40% of frame): +0.2
- **Centering**: -0.25 (vertical distance) / -0.2 (horizontal)
- Union of all masks scoring ≥ 0.4 (handles multi-person closeups)

#### POV Subject (MatAnyone2 POV mode)
Selects non-POV person:
- **Centering**: +0.0 (centered faces score higher)
- **Bottom-heavy penalty**: -0.4 (POV body extends below frame center)
- **Area bias**: Small masks favored (POV hands/feet, not body)

#### POV Body (RVM + POV mode)
Inverse of POV Subject; dilated by 2% of frame height for movement tolerance:
- Lowest-scoring mask (bottom-heavy, edge-touching)
- Applied as exclusion mask to RVM matte

**GPU Memory Strategy:**
```python
# Load SAM2, generate masks, unload immediately
masks = SAM2AutomaticMaskGenerator(...).generate(first_frame)
del mask_gen, predictor
gc.collect()
torch.cuda.empty_cache()
# Now safe to load MatAnyone2 (D003)
```

---

### 6. **Alpha Smoothing Wrapper** (`matte.py`)

**Class:** `AlphaSmoother`

**Purpose:** Reduce frame-to-frame alpha jitter via exponential moving average (EMA).

```python
alpha_out = weight * current + (1 - weight) * previous
```

**Parameters:**
- `weight` (default 0.85): 0 = full smoothing, 1.0 = no smoothing
- Blending happens after matting (post-processor)

**Use Case:** RVM outputs can flicker at occlusion boundaries; smoothing provides temporal stability.

---

### 7. **POV Exclusion Wrapper** (`matte.py`)

**Class:** `POVExclusionProcessor`

**Purpose:** Subtract POV body mask from RVM matte for first-person passthrough content.

**Workflow:**
1. Generate POV body mask from frame 1 (SAM2)
2. Wrap RVM processor to apply exclusion per-frame
3. Monitor for scene changes (histogram correlation)
4. Regenerate mask on drastic changes (laying → standing)

**Exclusion Algorithm:**
```python
exclusion_factor = 1.0 - (pov_mask / 255.0)
result = matte * exclusion_factor
```

Handles dynamic frame resizing (e.g., SBS eye downscaling) by resizing mask via NEAREST interpolation.

---

### 8. **Frame Scaler** (`scaler.py`)

**Purpose:** Pre-matte downscaling for VRAM-constrained matting, post-matte upscaling.

**Design:**
- **No-op** if frame pixels ≤ `max_pixels`
- **Downscale** to target resolution before matting (saves VRAM)
- **Upscale** matte to original resolution with LANCZOS

**Algorithm:**
```python
scale = sqrt(max_pixels / original_pixels)
target_w = int(original_w * scale) & ~1  # even dimensions for codecs
target_h = int(original_h * scale) & ~1
```

**Quality Impact:** For VR at 90+ FOV, downscaling ~20% has imperceptible quality loss.

**Example:** 8K 5800×2900 at 12GB VRAM:
- Tier recommendation: 810p (~1564×784)
- Scale: ~0.27; pixel reduction: ~93%
- Post-upscale retains full-res matte via learned features

---

### 9. **Scene Change Detector** (`scene_detect.py`)

**Purpose:** Monitor for drastic scene changes to trigger POV mask refresh.

**Detection Method:**
- Compute normalized luminance histogram (64 bins)
- Compute Pearson correlation with reference histogram
- If `corr < threshold` (default 0.4) and cooldown elapsed: scene change detected

**Cooldown:**
- `cooldown_frames` (default 30) prevents re-trigger during transition sequences
- Typical transition takes 5–10 frames; 30-frame cooldown avoids jitter

**Use Cases:**
- POV sits → stands (histogram shift: floor → sky)
- POV looks left → right (histogram doesn't change much, but SAM2 mask should)
- Cut to different room (sharp histogram drop)

**Correlation Interpretation:**
- `corr ≈ 1.0`: identical scene (no change)
- `corr ≈ 0.5`: partial change (lighting shift)
- `corr < 0.4`: drastic change (cut, major move)

---

### 10. **Pipeline Checkpoint** (`checkpoint.py`)

**Purpose:** Enable resume-from-crash for long-running jobs.

**Checkpoint Data:**
```python
@dataclass
class PipelineCheckpoint:
    input_path: str          # Video path (for validation)
    input_hash: str          # SHA-256 of first 64 KB
    config_hash: str         # SHA-256 of non-path config
    total_frames: int        # Total frames to process
    chunk_size: int          # Chunk size used
    completed_segments: int  # Segments successfully flushed
    completed_frames: int    # Total frames processed so far
    timestamp: str           # ISO timestamp
```

**Validation:**
- Recompute `input_hash` from file; must match saved
- Recompute `config_hash` from processing-relevant fields; must match saved
- If either differs: **checkpoint stale**, start from frame 1

**Deterministic Temp Naming:**
```python
temp_dir_name = f"AlphaPass_{safe_stem}_{config_hash[:8]}"
```
- Survives process restart (deterministic name)
- Config change → different hash → new temp dir
- Stale dirs (>7 days) auto-cleaned on pipeline start

---

### 11. **Pipeline Configuration** (`runner.py`)

**Class:** `PipelineConfig`

**Categories:**

| Category | Fields |
|----------|--------|
| **I/O** | `input_path`, `output_path`, `temp_dir` |
| **Matting** | `model_variant`, `downsample_ratio` |
| **Output** | `output_format`, `codec`, `crf` |
| **VR** | `projection`, `fisheye_fov`, `fisheye_mask_path`, `is_sbs`, `pov_mode` |
| **Frame Range** | `start_frame`, `end_frame` (1-based, 0 = all) |
| **MatAnyone2** | `use_fp16`, `ma2_internal_size`, `ma2_mem_frames`, `ma2_use_long_term`, `ma2_compile_model` |
| **Smoothing** | `temporal_smoothing` (1.0 = off) |
| **Performance** | `chunk_size`, `max_matting_pixels` (0 = auto) |
| **Resumption** | `auto_resume` |

**Hashing (for checkpoints):**
- Excludes paths (`input_path`, `output_path`, `temp_dir`)
- Excludes temp settings (not relevant to output)
- Includes all processing parameters (model, quality, VR settings, etc.)

---

## Design Patterns & Key Insights

### 1. **Protocol-Based Polymorphism**

Instead of class inheritance, MatteProcessor uses structural typing (duck typing enforced at runtime via `@runtime_checkable`). This allows:
- Minimal coupling between pipeline and backends
- Easy addition of new matting models (no base class modification)
- Wrapping (decorators) without factory method changes

### 2. **Chunked Processing for Long Videos**

**Problem:** 161k-frame videos at 8K would require `161k × PNG_size` disk space if extracted all at once.

**Solution:**
- Extract N frames (chunk_size, default 500)
- Process N frames to mattes
- Flush temp directory
- Repeat

**Peak disk usage:** `chunk_size × PNG_size` instead of `total_frames × PNG_size`

### 3. **Keyframe-Seeking with Imprecision Tolerance**

**ffmpeg `-ss` before `-i`:** Fast seek to nearest keyframe (~1–2 frame imprecision)

**Why acceptable for VR:**
- Keyframe imprecision occurs at chunk boundaries (every 500+ frames)
- Boundary overlaps (<5 frames per 161k) negligible for 90 FPS VR
- 2-frame error at 90 FPS = 22 ms; imperceptible motion

### 4. **GPU Memory Tiers**

Auto-config maps VRAM to performance tiers:
- **≥24GB**: No downscaling (full resolution matting)
- **16GB**: Max 1080p (2M pixels)
- **12GB**: Max 810p (1.5M pixels)
- **8GB**: Max 720p (1M pixels)
- **≤6GB**: Max 540p (0.5M pixels)

Tiers apply **per-eye** for SBS content (halved).

### 5. **SAM2 Load/Unload Sequencing**

MatAnyone2 requires SAM2 for mask generation but then must unload SAM2 before loading the ~2GB MatAnyone2 model.

```python
# Load SAM2 (1–2 GB)
mask = generate_first_frame_mask(first_frame, device)
# Cleanup: del SAM2, gc.collect(), torch.cuda.empty_cache()

# Now load MatAnyone2 (2+ GB)
processor = MatAnyone2Processor(mask, device)
```

---

## Error Handling & Resilience

### Cancellation
- `Pipeline.cancel()` sets `_cancelled = True`
- Per-frame loop checks `_cancelled`; raises `InterruptedError` if set
- Temp directory preserved for later resumption

### Input Validation
- `check_ffmpeg()` ensures ffmpeg available (early fail)
- `get_video_info()` validates video format + frame count
- Frame range validation (1-based, inclusive)

### Checkpoint Staleness Detection
- Input hash change: user edited source video → start fresh
- Config hash change: user adjusted settings → start fresh
- If valid: resume seamlessly from `completed_frames`

### VRAM Monitoring
- GPU auto-config runs at pipeline start
- Frame Scaler applies downscaling if needed
- Per-chunk disk check: halt if free space < 1 GB

### Device Fallback
- FP16 auto-disabled on CPU (FP32 only)
- MPS support added for macOS
- Multi-device test: `torch.cuda.get_device_name(0)` before assumptions

---

## Performance Characteristics

### Throughput (Benchmarks on RTX 5080, 5800×2900 8K SBS)

| Model | Mode | FPS | VRAM |
|-------|------|-----|------|
| RVM (mobilenetv3) | FP16, downsample 0.125 | ~12–15 | 8–10 GB |
| RVM (resnet50) | FP16, downsample 0.125 | ~8–10 | 12–14 GB |
| MatAnyone2 | FP16, internal 480, LT+WM | ~4–6 | 14–16 GB |
| MatAnyone2 | FP16 + torch.compile | ~6–8 | 14–16 GB |

*Note: SBS content processed per-eye (halved resolution per processor).*

### Memory Profile (per-eye 2900×1450)

| Component | FP32 | FP16 |
|-----------|------|------|
| RVM MobileNetV3 | ~3 GB | ~1.5 GB |
| MatAnyone2 Base | ~4 GB | ~2 GB |
| MatAnyone2 (internal 480) | ~6 GB | ~3 GB |
| SAM2 Small | ~2 GB | ~1 GB |

### Extraction Speed

- **ffmpeg keyframe seek**: ~100–200 FPS (fast, CPU-bound)
- **PNG write**: ~50 FPS (I/O-bound)
- **Directory polling**: 0.5s interval (2 polls/second)

---

## Integration Points with Other Domains

### Encoding & FFmpeg (`utils/ffmpeg.py`)
- **Imports:** `get_video_info()`, `check_ffmpeg()`, frame extraction helpers
- **Exports:** Matte video via custom temp directory setup
- **Responsibility Boundary:** Pipeline handles GPU matting; ffmpeg domain handles codec/quality

### UI & Worker (`ui/worker.py`)
- **Pipeline.on_progress** callback feeds `PipelineProgress` to UI
- **Pipeline.cancel()** called on "Stop" button
- **Worker thread** runs Pipeline.run() asynchronously

### Infrastructure (`utils/gpu.py`, `utils/bootstrap.py`)
- **get_device()** determines CUDA/MPS/CPU
- **auto_configure_gpu()** returns tier-based recommendations
- **configure_cuda_performance()** sets TF32 + cuDNN flags

### Settings & Persistence (`utils/settings.py`)
- **PipelineConfig** fields serialized to `settings.json`
- **Resume** state (checkpoint path) stored for auto-resume UI option

---

## Known Limitations & Constraints

### 1. **POV Heuristics**
SAM2 mask selection is heuristic-based (center-of-mass, aspect ratio, area percentiles). Edge cases:
- **Multiple close subjects**: Heuristic may fail; union method is fallback
- **Monochrome backgrounds**: SAM2 may underdetect people
- **Partial occlusions**: Mask may include/exclude incidental areas

### 2. **Scene Change Threshold**
Histogram correlation at 0.4 threshold can falsely trigger on:
- **Gradual lighting changes** (10–15 frames of ramp)
- **Motion blur** on fast pans (temporary histogram shift)
- **Cooldown (30 frames)** mitigates but doesn't eliminate

### 3. **Chunk Boundary Imprecision**
Keyframe-based seeking has ~1–2 frame error at boundaries:
- Acceptable for 60 FPS content (16–33 ms = imperceptible)
- For slow-motion content (24 FPS), error is ~40–80 ms (occasionally visible)

### 4. **MatAnyone2 Compile on Windows**
`torch.compile()` requires a compatible Triton build; fallback to no-compile gracefully if unavailable.

---

## Future Enhancement Opportunities

1. **Adaptive Chunk Sizing**: Adjust chunk size based on available VRAM (smaller chunks = less peak memory)
2. **Multi-frame Lookahead**: MatAnyone2 could benefit from future-frame context for faster motion
3. **Quantization**: INT8 MatAnyone2 weights for 50%+ further VRAM reduction (quality trade-off)
4. **Hybrid Models**: Switch between RVM (fast, coarse) and MatAnyone2 (slow, fine) per-scene
5. **Distributed Processing**: Multi-GPU chunk parallelization (requires checkpoint coordination)

---

## Testing Recommendations

### Unit Tests
- [ ] `SceneChangeDetector`: Known scene cuts should trigger; slow pans should not
- [ ] `FrameScaler`: Downscale + upscale should preserve alpha values within tolerance
- [ ] `AlphaSmoother`: Blending weight = 0 vs 1 should bracket ground truth
- [ ] `PipelineCheckpoint`: Hash stability across identical configs; staleness detection

### Integration Tests
- [ ] **Resume from checkpoint**: Extract, cancel mid-chunk, verify resume from saved frame
- [ ] **SBS processing**: Left/right eye frames correctly split → matte → merged
- [ ] **POV mask refresh**: Scene cut detected; SAM2 regenerates mask
- [ ] **Frame range**: start_frame=100, end_frame=200 processes only 101 frames
- [ ] **Codec fallback**: Try libx265; if unavailable, fallback to libx264

### Performance Tests
- [ ] Throughput on RTX 5080 matches benchmarks (allow ±20% variance)
- [ ] VRAM peak stays within GPU tier recommendation
- [ ] Disk cleanup removes stale directories > 7 days old

---

## Glossary

| Term | Definition |
|------|-----------|
| **Matte** | Single-channel mask (0–255) where 255 = fully opaque, 0 = fully transparent |
| **Keyframe** | Video frame that can be decoded independently (I-frame in H.264/HEVC) |
| **Recurrent State** | Hidden activations passed between frames for temporal context (RVM) |
| **XMem** | Cross-Memory architecture for long-term temporal memory (MatAnyone2) |
| **POV Mode** | First-person passthrough; excludes POV body from matte |
| **SBS** | Side-by-side stereo (left eye | right eye in single frame) |
| **Checkpoint** | Saved progress state enabling resume after interruption |

---

## Code Quality Notes

**Strengths:**
- Comprehensive docstrings with Args/Returns/Raises
- Type hints throughout (Python 3.10+ syntax)
- Structured logging via `loguru` (not print statements)
- Protocol-based design avoids tight coupling
- Graceful error handling (fallbacks, not hard fails)

**Areas for Consideration:**
- Large runner.py (~400 lines) could benefit from extraction of stage classes
- Magic numbers (0.125 downsample ratio, 480 internal size) should have named constants
- No input validation in PipelineConfig (relies on caller)

---

**End of Analysis**
