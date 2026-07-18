# Changelog — Krea 2 Identity Edit

Weights: https://huggingface.co/conradlocke/krea2-identity-edit
v1.2 updates the **nodes** (see below); they stay backward-compatible with v1/v1.1
weights via `fit_mode: crop`.

## v1.2 — 2026-07-17 (recommended)

`krea2_identity_edit_v1_2.safetensors` — pair it with the v1.2 nodes in this repo.

### New

**In the model:**
- **Better face likeness** on restaged subjects.
- **Character reference sheets** — both *using* a sheet as reference and *creating*
  one from a character.
- **Head / face swap** (and eye / person replacement).
- **Outpainting.**
- **Inpainting.**
- **Try-on** — put a garment onto a person.
- **Better person removal.**
- **Higher fidelity across the board, from a 1024 pass** — v1.1 had no high-resolution
  adaptation; v1.2 does.

**In the nodes:**
- **`ref_boost` — a reference-fidelity dial.** Turn how hard an edit locks onto the
  reference's appearance up or down (1.0 = neutral, >1 = pull harder toward the
  reference). Best value is model-specific.
- **No more blurry/stretched results (new `fit` geometry).** Sources are resampled to
  the target grid at a training-matched offset — and the old "match the source aspect
  ratio" requirement is gone. Needs `vae` + `source_image` connected on the patch node.

### Thanks
Head / face / eye / person swap is trained on **stablellama**'s MIT-licensed
[`change_eye_face_head_person`](https://huggingface.co/datasets/stablellama/change_eye_face_head_person)
dataset — big thanks for making it available.

### Node changes (technical)
- `fit_mode` defaults to `fit` (training-matched); `crop` remains for v1/v1.1-legacy weights.
- Added `ref_boost` / `ref_boost_a` reference-fidelity dials.
- Added optional grounded `(phrase:weight)` prompt weighting. Weights apply only
  to positive conditioning rows, including at CFG values above 1.

## v1.1 — 2026-07-09

`krea2_identity_edit_v1_1.safetensors`

- **Substantially improved face likeness and image fidelity**
- **Much stronger edit locality** — camera, pose, and untouched elements stay
  fixed far more reliably
- Better two-person identity separation
- More reliable object remove / replace
- Better compound outfit-change compliance
- Corrected reference geometry handling (training refs are now center-cropped,
  matching the shipped workflows)

**Known limitations of v1.1:**
- *Person*-replacement ("replace the woman with an orangutan") is currently
  weaker than v1 — keep v1 for that use case until v1.2
- No high-resolution adaptation pass yet: at high resolutions (especially
  two-person edits) identities can bleed together — prefer ~1–1.5MP and upscale
- `grounding_px`: v1.1's trained range is 384–768 (1024 often still works).
  If you get duplicated/split compositions, lower `grounding_px`.

## v1 — 2026-07-07

`krea2_identity_edit_v1.safetensors` — initial release. Remains available for
workflow reproducibility and for person-replacement edits.
