---
name: snapseg-finetune-recipe
description: >
  Fine-tune a chosen model family on snapseg-exported labels and re-export
  to ONNX so the snapseg adapter can load the trained weights. Invoke when
  the user wants to fine-tune a model on collected labels, train a custom
  model, distill a smaller model from a larger one, or close the data
  flywheel loop. Also invoke for: "fine-tune MobileSAM on my data",
  "train RITM from snapseg labels", "how do I make a custom model",
  "export back to ONNX after training", "dataset adapter for snapseg
  labels". Covers dataset adapter from the snapseg-label-export format,
  LoRA vs full fine-tune trade-off, eval split protocol, and ONNX export
  with the snapseg adapter's expected I/O signature.
---

# snapseg: fine-tune a model on your own labels

You are guiding a fine-tune cycle from snapseg-collected labels.
Training is Python territory by necessity (PyTorch); the shipped
snapseg binary stays Python-free. The output is an ONNX file (one
or two, depending on the family) that the existing snapseg adapter
can load directly.

Read `snapseg-label-export` for the on-disk label format; this skill
assumes it.

## When to fine-tune

- You have at least ~200 hand-corrected labels in `labels/`. Less,
  and your eval set is too small to trust.
- The base model is failing on a *specific* failure mode (e.g.
  glossy metal, low contrast, thin glue beads). Fine-tuning fixes
  per-domain shifts; it won't fix a fundamentally wrong model
  family.
- You have a clear eval set carved out — at least 20% of labels,
  held out before any training experiment.

## Family choice

For the first fine-tune of a snapseg-collected dataset:

| Family | Pros | Cons |
|---|---|---|
| **RITM** | Smallest, well-documented training pipeline, single-network. Ideal for "make this fast on CPU on the shop floor". | Won't match MobileSAM out-of-the-box quality; click-only. |
| **MobileSAM** | Same model the operator was clicking with; minimal distribution shift. | Encoder/decoder split makes training harder; larger output. |
| **FocalClick** | Local refinement is genuinely useful at click N+1. | Two-stage training (global + local) more involved. |

The pragmatic default for industrial deployment: **fine-tune RITM**
on the snapseg dataset. If you need box prompts in production,
fine-tune MobileSAM's decoder while leaving the encoder frozen.

## Dataset adapter

The snapseg label format is dataset-agnostic. The training pipeline
needs an adapter:

```python
class SnapsegDataset(torch.utils.data.Dataset):
    def __init__(self, label_dir: Path, image_size: int = 1024):
        self.samples = sorted(p for p in label_dir.iterdir() if (p / "mask.png").exists())
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample / "image.png").convert("L")  # grayscale
        mask = (np.array(Image.open(sample / "mask.png")) > 127).astype(np.uint8)
        with open(sample / "prompts.json") as f:
            prompts = json.load(f)["session"]
        # Resize image + mask to self.image_size, scale prompt coords accordingly.
        # Return whatever shape your training loop expects.
        ...
```

For RITM, follow `ISTrainDataset` in the official repo; the snapseg
adapter is a drop-in for `IsicDataset` / `COCOLVISDataset`.

## LoRA vs full fine-tune

For training under 1k labels:

- **LoRA on the decoder head** (MobileSAM) is the right starting
  point. Small adapter weights (~MB), no need to re-export the
  encoder, can be merged at deploy time.
- **Full fine-tune of RITM** is feasible — the model is small (~10M
  params).
- **Full fine-tune of the SAM encoder** is rarely a good idea unless
  you have ≥10k labels and a clear pretraining-domain shift.

## Eval split protocol

Hold out at least 20% of labels for evaluation, stratified by:
- `meta.toml::operator.quality` (good vs needs_review)
- Image source directory (so labels from the same camera run aren't
  split across train and eval)
- Date (the most recent batch is the eval set; trains can use older
  labels — this catches drift)

Metrics that matter for snapseg:
- **IoU @ N clicks** (1, 3, 5) — does the fine-tuned model reach
  high IoU faster than the base?
- **Boundary F-score** (3 px tolerance) — is the boundary accurate
  enough that subpixel refinement converges quickly?
- **Failure rate** — fraction of eval images where IoU stays below
  0.7 at 5 clicks. The model is *practically useless* below this.

## ONNX export

After training, re-export to match the snapseg adapter's I/O.
**Re-use the adapter's documented schema verbatim.** Snapseg's
adapters look up tensors by name with a substring fallback, so a
minor renaming is forgiven, but stick to the canonical names where
possible.

For RITM, the export goes from a `.pth` checkpoint to a single ONNX
file with inputs `image`, `click_map`, `prev_mask`. Use the script
in the upstream RITM repo's `tools/onnx_export.py` or write a
two-line script around `torch.onnx.export`.

For MobileSAM, the encoder and decoder must be exported separately
(the encoder/decoder split is what makes the runtime cheap). The
official MobileSAM repo's `scripts/export_onnx_model.py` is the
reference.

## Closing the loop

After ONNX export:

1. Compute `sha256` of the file(s).
2. Add a new `[[model]]` entry to `models.toml` (or to a personal
   override `models-extra.toml`) with the local file path or URL
   and the sha256.
3. Restart snapseg → load the new model → smoke test on the eval
   split.
4. Compare against the base model on the eval metrics above.
5. If the fine-tune wins, ship the new model name as the default
   in `models.toml` for that family.

## Pitfalls

- **Label leak.** Labels with `quality = "needs_review"` should
  stay out of training until they're either corrected or rejected.
- **Click distribution shift.** The operator clicks in a different
  pattern than synthetic training (slower, more deliberate, more
  negative clicks). If you reuse RITM's synthetic-click
  augmentation in training, it should match the operator's
  observed pattern from `prompts.json`.
- **Encoder freeze.** When fine-tuning MobileSAM, freeze the
  encoder unless you have ≥5k labels — otherwise you'll overfit
  the encoder on industrial-domain images and lose the model's
  zero-shot strength on the next domain.
- **Versioning.** Pin the snapseg commit hash that produced the
  labels into `meta.toml::runtime.snapseg_commit`. A subtle bug
  in label export can poison the training set for months.
