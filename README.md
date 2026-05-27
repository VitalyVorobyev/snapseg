# snapseg

Part of the [vitavision.dev](https://vitavision.dev/) computer-vision atlas.

[![CI](https://github.com/VitalyVorobyev/snapseg/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/snapseg/actions/workflows/ci.yml)
[![Security audit](https://github.com/VitalyVorobyev/snapseg/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/snapseg/actions/workflows/audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Interactive deep-model image segmentation for industrial inspection — and the label factory you build it with.** Smart models (MobileSAM, eventually RITM, FocalClick, SAM2-Tiny) take a few clicks from an operator and produce a mask. Classical subpixel edge refinement sharpens the boundary. The result becomes a label — and labels are the point. snapseg is designed so the masks you produce flow back into fine-tuning smaller, faster, domain-specialized models.

All-Rust, no Python at runtime. Single static binary. Optional GPU / NPU acceleration when present; clean CPU fallback when not.

## Status

**Alpha.** End-to-end pipeline works against MobileSAM ONNX (encoder + decoder). Label-export workflow, subpixel-edge refinement, and a second model family are next-up milestones — see [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Why snapseg

Most segmentation tooling is one of: (a) a research demo that needs Python + a GPU to even open the window, (b) a heavyweight cloud labeling service, or (c) a SAM viewer with no path from "I labeled this image" to "I trained my own model from a thousand of these". snapseg targets the gap:

- **Operator-friendly.** Click positive, right-click negative, hold to box — the mask updates in tens of milliseconds after the first encoder pass.
- **Hardware-friendly.** Modest CPU is the floor; CoreML / CUDA / DirectML / TensorRT light up when present, no recompile.
- **Industrial-friendly.** Grayscale-first, no cloud, no Python deploy. Drop the binary on an inspection PC; it works.
- **Label-flywheel-first.** The output is more than a mask — it's a labeled training record. Fine-tune your own model when you have enough; ship it in the same registry.

## What ships

| Layer | What it does |
|---|---|
| `snapseg-app` | Desktop GUI (egui). Open an image, load a model, click, get a mask. |
| `snapseg-models` | Per-family ONNX adapters. MobileSAM today; RITM / FocalClick scaffolded. |
| `snapseg-runtime` | `ort` (load-dynamic) wrapper. Execution-provider selection, SAM/ImageNet preprocessing helpers. |
| `snapseg-registry` | TOML model zoo with sha256-verified, downloadable, or air-gapped cache resolution. |
| `snapseg-edges` | Classical subpixel edge refinement (placeholder body; marching-squares + parabolic-fit lands next). |
| `snapseg-core` | The trait every adapter implements + the prompt / capability / mask types. Zero workspace dependencies. |

Workspace layering:

```
snapseg-app           ← apex, single binary `snapseg`
       ↓
snapseg-models        ← per-family adapters (mobile_sam, ritm, focalclick, ...)
       ↓
snapseg-runtime · snapseg-registry · snapseg-edges
       ↓
snapseg-core          ← types + trait, nothing else
```

## Requirements

- Rust 1.85+ (workspace `edition = "2024"`).
- **onnxruntime** on the system. `ort` is built with `load-dynamic`, so the library is discovered at runtime.
  - macOS: `brew install onnxruntime`
  - Linux: distribution package or [official release](https://github.com/microsoft/onnxruntime/releases)
  - Windows: drop `onnxruntime.dll` next to the binary or set `ORT_DYLIB_PATH`
- **A MobileSAM ONNX export** (encoder + decoder). Most public exports follow the canonical Meta SAM schema and work out of the box; the adapter falls back to substring matching on output names so minor naming variances are tolerated.

## Quick start

```bash
git clone https://github.com/VitalyVorobyev/snapseg
cd snapseg
brew install onnxruntime              # macOS
cargo build -p snapseg-app --release
./target/release/snapseg
```

In the app:

1. **Open image…** — load a grayscale (or color; it gets converted) industrial photo.
2. **Load MobileSAM…** — pick the encoder `.onnx`, then the decoder `.onnx`. The encoder runs once per image (~1–2 s on CPU); subsequent decoder calls are fast.

   The current default `models.toml` ships with empty URLs for MobileSAM — you bring the ONNX files yourself (see the [MobileSAM repo](https://github.com/ChaoningZhang/MobileSAM) for export instructions; a 1024×1024-input encoder + canonical SAM-ViT-H decoder pair is the standard combo). A real auto-download path lands when a blessed export is published.

3. **Click** on the part you want segmented. Right-click for negative clicks (background hints).
4. The translucent blue overlay is the predicted mask. **Last segment: NN ms** in the side panel tells you the decoder latency.

## Roadmap

The full milestone plan is in [`docs/ROADMAP.md`](docs/ROADMAP.md). Short version:

| Milestone | Theme |
|---|---|
| M1 (now) | Pipeline polish, agents, skills, CI |
| M2 | Label export → reusable dataset on disk |
| M3 | Subpixel edge refinement |
| M4 | Multi-mask + IoU selection |
| M5 | Threaded inference (no UI freeze) |
| M6 | Second model family (RITM / FocalClick) |
| M7 | Fine-tune loop: train your own model from your own labels |

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

Built on top of the open-source work of the SAM / MobileSAM teams (Meta AI, Zhang et al.), the `ort` crate (`https://github.com/pykeio/ort`), and the wider `egui` / `ndarray` ecosystems.
