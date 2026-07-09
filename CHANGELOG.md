# Changelog — Krea 2 Identity Edit (model weights)

The nodes in this repository are unchanged and work with all model versions.
Weights: https://huggingface.co/conradlocke/krea2-identity-edit

## v1.1 — 2026-07-09 (recommended)

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
