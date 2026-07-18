# ComfyUI-Krea2Edit

Instruction-based image editing for **Krea 2** in ComfyUI — the node pack that powers
the **Krea 2 Identity Edit** LoRA. Turns Krea 2 (Raw or Turbo) into an image editor with dual
conditioning: the source image is injected both as VAE latent tokens (appearance) and
into the Qwen3-VL text encoder (semantic grounding), matching how the LoRA was trained.

## Model versions

See [CHANGELOG.md](CHANGELOG.md) — **v1.2 is recommended** (better face likeness,
plus the new `fit` reference geometry and `ref_boost` fidelity dial).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lbouaraba/comfyui-krea2edit
# restart ComfyUI
```

Requirements: a ComfyUI version with native Krea 2 support, the Krea 2 model
(Raw or Turbo), the Qwen3-VL 4B text encoder used by Krea 2, and the Krea 2 Identity Edit
LoRA (`krea2_identity_edit_v1_2.safetensors`). No extra Python dependencies.

## Nodes

### `Krea2EditModelPatch`
Wraps the diffusion model so the VAE-encoded source image is prepended as clean
in-context tokens (RoPE frame 1). Inputs:
- `model` — Krea 2 (LoRA already applied)
- `source_latent` — VAEEncode of the image being edited
- `source_latent_b` *(optional)* — second reference (RoPE frame 2) for two-input
  edits (e.g. person + scene)
- `vae` + `source_image` *(optional, recommended)* — the blur-proof pixel path: give
  the raw image (and VAE) and the node fits it to the target grid in pixel space.
  Required for `fit_mode: fit`.
- `fit_mode` *(default `fit`)* — how a source fits a mismatched output aspect ratio.
  `fit` = training-matched resample at a centered offset (v1.2); `crop` = center-crop,
  the v1/v1.1-legacy geometry (use with older weights).
- `ref_boost` *(default 1.0)* — reference-fidelity dial; >1 pulls harder toward the
  reference's appearance, <1 loosens. `ref_boost_a` is the same dial for the scene ref in two-ref edits.
- `prompt_weights` *(optional)* — connect the matching output from
  `Krea2EditGroundedEncode` to enable per-phrase weights.

### `Krea2EditGroundedEncode`
Image-grounded instruction encoding — the text encoder *sees* the image while
reading your instruction, exactly as during training. Inputs:
- `clip` — the Krea 2 CLIP (Qwen3-VL, loaded with `type: krea2`)
- `prompt` — the edit instruction ("recolor the car to matte black")
- `image` — the same source image
- `image_b` *(optional)* — second reference for two-input edits
- `grounding_px` — grounding resolution (default 768; trained range 512–1536).
  This is a quality dial: lower = stronger edit adherence, higher = stronger
  identity/likeness. Try 1024+ for people, 512 for stubborn scene changes.
- `weight_strength` — global multiplier for `(phrase:weight)` effects (default 1.0).

The first output remains the grounded `CONDITIONING`. The second output carries
the token positions and weights for `Krea2EditModelPatch.prompt_weights`.

### Prompt weights

Use `(phrase:weight)` inside the grounded prompt, then connect
`Krea2EditGroundedEncode.prompt_weights` to `Krea2EditModelPatch.prompt_weights`:

```text
make the jacket (deep red:1.5) and suppress the (background crowd:-1)
```

- `weight > 1` emphasizes the phrase with attention-logit bias.
- `weight <= 1` scales its attention value; negative values can suppress or
  subtract a concept.
- `weight_strength` scales the effect globally. The change compounds through
  all Krea 2 transformer blocks, so start at 1.0 or lower.
- Positive emphasis uses PyTorch attention so it can apply an additive bias;
  it may use more VRAM than your configured optimized attention backend.

Prompt weights apply only to the positive conditioning rows, so they are safe at
CFG values above 1 and do not alter the grounded negative branch.

This is an independent grounded adaptation of the experimental approach introduced
by KJNodes in commits [`1271209`](https://github.com/kijai/ComfyUI-KJNodes/commit/1271209845da1e463f63ba9e9dadd81fa49986d9)
and [`780930c`](https://github.com/kijai/ComfyUI-KJNodes/commit/780930c1b4b6df347080f1f36d2845c4437358b2).

**Both nodes are required.** With a stock `CLIPTextEncode` the model never sees the
image semantically and quality drops sharply, especially for scene-referential
instructions ("the man on the left").

## Minimal wiring

```
LoadImage ─┬─ VAEEncode ── Krea2EditModelPatch.source_latent
           └─ Krea2EditGroundedEncode.image     (+ your prompt)
UNETLoader ── LoraLoaderModelOnly (krea2_identity_edit_v1_2 @1.0) ── Krea2EditModelPatch.model
Krea2EditGroundedEncode.prompt_weights ── Krea2EditModelPatch.prompt_weights
Krea2EditModelPatch ── KSampler.model
Krea2EditGroundedEncode ── KSampler.positive
Krea2EditGroundedEncode (empty prompt, same image) ── KSampler.negative
EmptySD3LatentImage ── KSampler.latent_image
```

Example workflow in `workflows/`: `krea2_identity_edit.json` — single-image editor by
default; enable group 2 (toggle its Bypass off) for two-image person-into-scene edits.

## Usage notes (read these — they matter)

1. **Aspect ratio.** With `fit_mode: fit` (default in v1.2) and `vae` + `source_image`
   connected, mismatched source/output aspect ratios are handled — the source is
   resampled to the target grid. On `crop`/legacy weights, still match the AR: a
   mismatched AR is out of distribution and degrades identity/preservation.
2. **Turbo, 8 steps, CFG 1** is the fast path (~1 min at 2MP) and works for most
   edits: recolor, add/insert, attribute changes, restyles, scene translation.
3. **Removals and other "delete salient content" edits need real guidance:**
   use the **Raw** model at **CFG 3, ~20 steps**. Distilled Turbo at CFG 1 will
   usually re-render the subject instead of removing it.
4. At CFG > 1, ground the negative too: a second `Krea2EditGroundedEncode` with an
   empty prompt and the same image (this is the trained unconditional).
5. Two-input edits: scene image → `source_latent`/`image`, subject image →
   `source_latent_b`/`image_b`. Leave the b-inputs unconnected for single-image use.
6. **Generate at ≤2MP.** Above the trained range, source content can bleed into
   the output or subjects duplicate.
7. **Two people with distinct faces:** chain single-ref inserts (place person A,
   then run a second edit adding person B from their reference) — currently more
   face-faithful than one two-ref pass.
8. **Prompt weights target only the positive branch.** Connect the positive grounded
   encoder's `prompt_weights` output to the source patch. The negative encoder's
   output stays disconnected, including in Raw/CFG > 1 workflows.

## License / credits

Nodes: Apache-2.0. The **Krea 2 Identity Edit** weights ship separately under the
Krea 2 Community License Agreement (see the model card, `LICENSE.pdf`, and `NOTICE`
in the weights repo).
Built on Krea 2 by Krea AI; text encoder Qwen3-VL (Alibaba).
