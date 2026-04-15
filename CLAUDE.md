# Claude Code Project Notes

## Session Startup — Read This First

Every session must read these documents before beginning any task. This is how you
get oriented — skip this and you'll duplicate work, miss context, or break an
interface that another domain depends on.

**Step 1 — Architect's Handbook** (in this repo): `ARCHITECT.md`
Read the Foundation chapter (Chapter 1) always. Then read the chapter(s) for whichever
domain your task touches. This document defines the five ownership domains, the
interface contracts between them, and the rules for when changes require coordination.

**Step 2 — Work Log** (local, not in this repo): `C:\Users\Admin\projects\Major project logs\AlphaPass\WORKLOG.md`
Read the Current Sprint and Decisions Made sections to understand what's in progress,
what was recently decided, and what the last session left for you to pick up. If the
file doesn't exist yet, create the directory and copy the template from the repo's
`docs/templates/WORKLOG-TEMPLATE.md`.

**Step 3 — Before you close this session:**
Update the Work Log. Add a Session History entry, check off completed tasks in the
Current Sprint, and record any decisions or failed approaches in their sections. The
next session depends on this.


---

## Project Overview

AlphaPass — Qt-based desktop app for automated video matting and alpha channel
generation for VR passthrough content.

**Repository:** `C:\Users\Admin\projects\AlphaPass\AlphaPass`
**GitHub:** https://github.com/SifuInTheShell/AlphaPass
**Branch:** master

### Architecture at a Glance

For the full architecture, domain boundaries, and interface contracts, see `ARCHITECT.md`.
This section is a quick-reference summary only.

**Entry point:** `src/AlphaPass/main.py` → MainWindow (PySide6/Qt)
**Pipeline:** `pipeline/runner.py` — orchestrates ffmpeg extraction → AI matting → video assembly
**GPU bootstrap:** `utils/bootstrap.py` — detects NVIDIA GPU via nvidia-smi, installs correct CUDA PyTorch wheel at startup before any torch imports
**Settings:** `utils/settings.py` → `~/.config/AlphaPass/settings.json`
**Worker threading:** `ui/worker.py` — PipelineWorker (QThread) runs pipeline, InstallWorker handles in-app dependency install
**GPU detection:** `utils/gpu.py` — get_device() and get_device_info() for CUDA/MPS/CPU

### Domain Ownership (Quick Reference)

| Domain | Key Files | What It Owns |
|--------|-----------|--------------|
| Pipeline Core | `pipeline/matte.py`, `runner.py`, `rvm.py`, `matanyone2.py`, `scaler.py`, `checkpoint.py`, `scene_detect.py`, `sam2_masks.py` | Frame-to-matte transformation, MatteProcessor protocol, chunked pipeline orchestration |
| Encoding & FFmpeg | `utils/ffmpeg.py`, `cli/encode_runner.py` | Frame extraction, video assembly, DeoVR alpha packing, codec fallback chain |
| User Interface | `ui/main_window.py`, `preview.py`, `themes.py`, `worker.py` | Qt GUI, preview pane, worker threading, lens detection |
| CLI & Batch | `cli/queue_runner.py`, `cli/encode_runner.py` | Headless runners, batch processing, encode queue |
| Infrastructure | `utils/gpu.py`, `bootstrap.py`, `settings.py`, `sbs.py`, `masks.py` | GPU detection, PyTorch bootstrap, settings, SBS stereo, DeoVR masks |

See `ARCHITECT.md` Chapter 1 (Foundation) for the full explanation of domain boundaries
and how they communicate.

### Features by Domain (Team Assignment)

Use this table to assign teams to specific features. Each feature maps to a domain owner.

| Feature | Domain | Core Files | Status |
|---------|--------|-----------|--------|
| **AI Matting Engines** (RVM, MatAnyone2, SAM2) | Pipeline Core | `pipeline/rvm.py`, `pipeline/matanyone2.py`, `pipeline/sam2_masks.py` | Active |1
| **Chunked Pipeline & Checkpointing** | Pipeline Core | `pipeline/runner.py`, `pipeline/checkpoint.py`, `pipeline/scaler.py` | Active |
| **Frame Extraction** | Encoding & FFmpeg | `utils/ffmpeg.py` | Active |
| **Video Assembly & Encoding** | Encoding & FFmpeg | `cli/encode_runner.py` | Active |
| **DeoVR Format Support** | Encoding & FFmpeg | `utils/ffmpeg.py`, `pipeline/runner.py` | Active |
| **Qt Desktop GUI** | User Interface | `ui/main_window.py`, `ui/themes.py` | Active |
| **Preview Pane & Real-time Display** | User Interface | `ui/preview.py`, `ui/worker.py` | Active |
| **GPU Detection & Auto-Config** | Infrastructure | `utils/gpu.py` | Active |
| **PyTorch Bootstrap & CUDA Management** | Infrastructure | `utils/bootstrap.py` | Active |
| **Settings & Persistence** | Infrastructure | `utils/settings.py` | Active |
| **Stereo (SBS) Support** | Infrastructure | `utils/sbs.py` | Active |
| **Batch Processing & Queue** | CLI & Batch | `cli/queue_runner.py` | Active |
| **Headless Encode Runner** | CLI & Batch | `cli/encode_runner.py` | Active |

---sv p

## User Hardware

- Windows 11 Pro laptop with NVIDIA RTX 5080 (Blackwell architecture, sm_120, CUDA 13.2)
- Requires PyTorch cu128 wheels (cu126 doesn't support Blackwell)
- Synology NAS at `\\192.168.1.95\usbshare1` for storing large VR video files
- Typical content: 8K (5800x2900) HEVC, 60fps, 100k+ frames


---

## Documentation Navigation

Use this index to navigate project documentation. Each agent/developer should read the docs relevant to their task.

| Need to... | Read... | Location |
|-----------|---------|----------|
| **Understand domain boundaries and interfaces** | ARCHITECT.md (Chapter 1: Foundation) | In repo |
| **Deep-dive on a specific domain** | ARCHITECT.md (Chapter 2+) | In repo |
| **Find past bugs, fixes, and features** | CHANGELOG.md | In repo |
| **Learn Windows/ffmpeg/CUDA gotchas** | GOTCHAS.md | In repo |
| **Know current sprint and decisions** | WORKLOG.md | `C:\Users\Admin\projects\Major project logs\AlphaPass\WORKLOG.md` (local) |
| **Pipeline Core — SME deep-dive** | WORKLOG-pipeline.md | `C:\Users\Admin\projects\Major project logs\AlphaPass\WORKLOG-pipeline.md` (local) |
| **Encoding & FFmpeg — SME deep-dive** | WORKLOG-encoding-ffmpeg.md | `C:\Users\Admin\projects\Major project logs\AlphaPass\WORKLOG-encoding-ffmpeg.md` (local) |
| **Get oriented (first time here?)** | This file (CLAUDE.md) | In repo |

**Agent hint:** If a task touches a specific domain (e.g., "add GPU optimization"), read the corresponding ARCHITECT chapter + GOTCHAS.md before starting. This prevents duplicating work and catches architecture-breaking changes early. 