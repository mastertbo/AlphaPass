# Contributor Onboarding — AlphaPass

Welcome. This guide gets you from a fresh clone to a running app and a successful test suite run, then walks you through the codebase well enough to make a meaningful change.

Assumes: Python proficiency. Does not assume: any knowledge of VR, video processing, or Qt.

For architecture decisions and design rationale, see [ARCHITECTURE.md](../ARCHITECTURE.md). For user-facing feature documentation, see [README.md](../README.md).

---

## Prerequisites

| Tool | Minimum | Install |
|---|---|---|
| Python | 3.10 | [python.org](https://python.org) or your OS package manager |
| FFmpeg | 4.x (7.x recommended) | See below |
| uv | any | `pip install uv` or `curl -Lsf https://astral.sh/uv/install.sh \| sh` |
| GPU | optional | CPU works, but too slow for development iteration |

**Install FFmpeg:**

```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux (Debian / Ubuntu)
sudo apt install ffmpeg
```

**NVIDIA note:** On Linux with WSL2, the NVIDIA driver must be installed on the Windows host — not inside the container. The CUDA toolkit inside WSL is provided automatically through the driver.

---

## Setup

```bash
git clone https://github.com/SifuInTheShell/AlphaPass.git
cd AlphaPass
uv sync
```

That's it for the core install. To also get the experimental MatAnyone 2 model:

```bash
uv sync --extra matanyone2
```

**Verify CUDA is available (NVIDIA GPU only):**

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If this prints `False` with an NVIDIA GPU present, don't worry — the bootstrap module will detect the mismatch and install the correct CUDA wheel on the first `uv run AlphaPass`. It re-execs the process once and then proceeds normally. See [ARCHITECTURE.md — Bootstrap](../ARCHITECTURE.md#bootstrap-and-pytorch-installation) for details.

---

## Running the App

```bash
uv run AlphaPass
```

**What to expect on first launch:**

1. Bootstrap check runs (a second or two; or longer if it needs to reinstall torch)
2. The GPU name appears in the status bar (e.g. `CUDA — NVIDIA GeForce RTX 4070`) — if you see `CPU` with a GPU installed, check the bootstrap note above
3. On your first process job, the RVM model (~15–55 MB) is downloaded automatically to `~/.cache/AlphaPass/models/`

---

## Running Tests

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

All 50 tests should pass. **No GPU and no FFmpeg required** — all hardware and filesystem operations are mocked via `unittest.mock`.

| Test file | What it covers |
|---|---|
| `test_matte_protocol.py` | Processor protocol, factory, settings, CPU fallback |
| `test_sbs.py` | SBS detection, frame / matte split + merge |
| `test_scene_detect.py` | Scene change detector, cooldown, threshold, reset |
| `test_pov_mask.py` | POV body mask selection, scoring, dilation |
| `test_integration_matanyone2.py` | MatAnyone 2 processor, SAM2 masks, re-exports |

Run a single file during development:

```bash
uv run python -m unittest tests/test_sbs.py
```

---

## Codebase Orientation

Rather than list every file (see [ARCHITECTURE.md](../ARCHITECTURE.md) for that), here is the navigation guide for common change types:

| I want to… | Start here |
|---|---|
| Change anything in the UI | `src/vrautomatte/ui/main_window.py` |
| Add a new matting model | `src/vrautomatte/pipeline/matte.py` + `src/vrautomatte/pipeline/rvm.py` (reference impl) |
| Change the pipeline processing loop | `src/vrautomatte/pipeline/runner.py` |
| Change encoding or FFmpeg commands | `src/vrautomatte/utils/ffmpeg.py` |
| Change settings defaults or add a new setting | `src/vrautomatte/utils/settings.py` |
| Change CLI batch tool behaviour | `src/vrautomatte/cli/queue_runner.py` or `encode_runner.py` |

**Three entry points** (from `pyproject.toml`):

```
AlphaPass       → AlphaPass.main:main          → Qt GUI
AlphaPass-queue → AlphaPass.cli.queue_runner:main   → batch matting CLI
AlphaPass-encode → AlphaPass.cli.encode_runner:main → AV1 re-encode CLI
```

The importable package name is `AlphaPass`; the source lives in `src/vrautomatte/` on disk.

---

## Adding a New Matting Model

Here is a concrete walkthrough. Use `src/vrautomatte/pipeline/rvm.py` as your reference implementation.

**Step 1 — Create the module**

```python
# src/vrautomatte/pipeline/your_model.py
import numpy as np

class YourModelProcessor:
    """One-line description of the model."""

    def __init__(self, device: str, use_fp16: bool = True) -> None:
        # Load weights here
        ...

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single RGB frame (H×W×3, uint8).

        Returns:
            Grayscale alpha matte (H×W, float32, range 0–1).
        """
        ...

    def reset(self) -> None:
        """Reset any recurrent state between clips or scene changes."""
        ...

    def cleanup(self) -> None:
        """Release GPU resources."""
        ...
```

No inheritance required — `MatteProcessor` is a structural Protocol. Any object with these three methods is valid.

**Step 2 — Register the variant**

In `src/vrautomatte/pipeline/matte.py`, add `"your_model"` to the `VARIANTS` list (or equivalent constant).

**Step 3 — Add a factory branch**

In `create_processor()` in `pipeline/matte.py`:

```python
elif config.model_variant == "your_model":
    from AlphaPass.pipeline.your_model import YourModelProcessor
    base = YourModelProcessor(device=device, use_fp16=config.use_fp16)
```

**Step 4 — Add the display name to the UI**

In `src/vrautomatte/ui/main_window.py`, find the line that adds `"mobilenetv3"` to the model combo box and add your display name:

```python
self.model_combo.addItems([
    "mobilenetv3",
    "resnet50",
    "MatAnyone 2 (experimental)",
    "Your Model",   # ← add here
])
```

Map the new combo index to your variant name wherever `model_combo.currentIndex()` is translated to `model_variant` (search `model_combo` in `main_window.py`).

**Step 5 — Write tests**

```python
# tests/test_your_model.py
from unittest.mock import MagicMock, patch
import unittest, numpy as np

class TestYourModelProcessor(unittest.TestCase):
    @patch("AlphaPass.pipeline.your_model.load_weights")  # mock weight loading
    def test_process_frame_returns_correct_shape(self, _mock):
        from AlphaPass.pipeline.your_model import YourModelProcessor
        proc = YourModelProcessor(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        matte = proc.process_frame(frame)
        self.assertEqual(matte.shape, (480, 640))
        self.assertEqual(matte.dtype, np.float32)
```

Tests must pass without a GPU. See `tests/test_matte_protocol.py` for more mocking patterns.

**Step 6 — Run the full suite**

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

---

## Common Development Tasks

**Syntax check without running the app:**

```bash
uv run python -c "import ast; ast.parse(open('src/vrautomatte/ui/main_window.py').read()); print('OK')"
```

**Find the per-run processing log after a crash:**

The log is at `{temp_dir}/AlphaPass_{stem}_{hash}/AlphaPass.log`. Find your system temp dir:

```bash
uv run python -c "import tempfile; print(tempfile.gettempdir())"
```

Then look for directories starting with `AlphaPass_`.

**Clean up stale temp dirs manually:**

The pipeline cleans them automatically on start (anything older than 7 days). To do it manually, delete `AlphaPass_*` directories from your system temp folder, or call `cleanup_stale_dirs()` from `pipeline/checkpoint.py`.

**Check GPU detection:**

```bash
uv run python -c "from AlphaPass.utils.gpu import get_device_info; print(get_device_info())"
```

---

## Code Style Quick Reference

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full style guide. At a glance:

| Rule | Detail |
|---|---|
| Logging | `from loguru import logger` — no `print()` in committed code |
| Strings | Double quotes |
| Line length | 100 characters max |
| Imports | stdlib → third-party → local, alphabetical within each group |
| Type hints | Required on all new or modified functions |
| Docstrings | Required on modules, classes, and public functions |
| Module size | Target ≤ 150 LOC; hard cap 200 LOC (CSS/theme files exempt) |
| Commit messages | Conventional Commits — `feat:`, `fix:`, `docs:`, `test:` etc. |

---

## Testing Philosophy

There is no CI configured currently — tests are run locally before pushing. The test suite is designed to be completely hardware-independent:

- **GPU operations** are mocked via `unittest.mock.patch`; no CUDA or MPS device is required
- **File system operations** that touch real video files are mocked; no FFmpeg is required
- **Model weights** are never loaded in tests — model constructors are patched at the import level

This means tests are always deterministic and fast, regardless of what hardware you have. When adding tests for a new model or pipeline change, follow the same mocking pattern so the CI constraint is preserved when CI is eventually configured.

---

## Key Contributor Gotchas

These are non-obvious bugs from `CLAUDE.md` that have already cost development hours:

**Never use `subprocess.PIPE` for long-running ffmpeg on Windows.**
The Windows pipe buffer fills at ~64 KB — roughly 977 frames of stderr — and the process deadlocks. Use `subprocess.DEVNULL` for stdin/stdout/stderr and poll the output directory instead. See `utils/ffmpeg.py` for the correct pattern.

**Bootstrap re-execs the process.**
If you are debugging startup behaviour and the process appears to restart, check whether `AlphaPass_TORCH_OK` is set in the environment. That env var is set by bootstrap before the re-exec to prevent an infinite restart loop. Unset it to force bootstrap to run again.

**Settings are index-based, not value-based.**
`settings.json` stores combo box indices. `"chunk_size": 2` means index 2 → 500 frames, not 2 frames. The index-to-value mapping is in `_DEFAULTS` in `utils/settings.py`. If you add a new combo option, update both the UI code and `_DEFAULTS`.

**The importable package name does not match the source directory.**
The source lives in `src/vrautomatte/` on disk, but the Python import path is `AlphaPass.*` (set by `pyproject.toml`). If your IDE's "find definition" jumps to a wrong location, check whether it is resolving the installed package vs. the editable source.

**ffmpeg's `select` filter without `-frames:v` decodes the entire video.**
`select='between(n,start,end)'` tells ffmpeg *which* frames to output, but ffmpeg still decodes every frame up to the end of the file to check the condition. Always pair it with `-frames:v <count>` to stop after outputting the desired frames. This was the root cause of a "hanging at frame 977" bug that took significant debugging to identify.
