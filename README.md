# ComfyUI-Krea2Edit

Instruction-based image editing for **Krea 2** in ComfyUI — the node pack that powers
the **Krea 2 Identity Edit** LoRA. Turns Krea 2 (Raw or Turbo) into an image editor with dual
conditioning: the source image is injected both as VAE latent tokens (appearance) and
into the Qwen3-VL text encoder (semantic grounding), matching how the LoRA was trained.

## Model versions

See [CHANGELOG.md](CHANGELOG.md) — **v1.1 is recommended** (better likeness,
locality, and remove/replace; two honest caveats listed there).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lbouaraba/comfyui-krea2edit
# restart ComfyUI
```

Requirements: a ComfyUI version with native Krea 2 support, the Krea 2 model
(Raw or Turbo), the Qwen3-VL 4B text encoder used by Krea 2, and the Krea 2 Identity Edit
LoRA (`krea2_identity_edit_v1.safetensors`). No extra Python dependencies.

## Nodes

### `Krea2EditModelPatch`
Wraps the diffusion model so the VAE-encoded source image is prepended as clean
in-context tokens (RoPE frame 1). Inputs:
- `model` — Krea 2 (LoRA already applied)
- `source_latent` — VAEEncode of the image being edited
- `source_latent_b` *(optional)* — second reference (RoPE frame 2) for two-input
  edits (e.g. person + scene)

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

**Both nodes are required.** With a stock `CLIPTextEncode` the model never sees the
image semantically and quality drops sharply, especially for scene-referential
instructions ("the man on the left").

## Minimal wiring

```
LoadImage ─┬─ VAEEncode ── Krea2EditModelPatch.source_latent
           └─ Krea2EditGroundedEncode.image     (+ your prompt)
UNETLoader ── LoraLoaderModelOnly (krea2_identity_edit_v1 @1.0) ── Krea2EditModelPatch.model
Krea2EditModelPatch ── KSampler.model
Krea2EditGroundedEncode ── KSampler.positive
Krea2EditGroundedEncode (empty prompt, same image) ── KSampler.negative
EmptySD3LatentImage ── KSampler.latent_image
```

Example workflows in `workflows/`: `krea2_edit_single_ref.json`, `krea2_edit_two_ref.json`.

## Usage notes (read these — they matter)

1. **Match the aspect ratio.** The target latent (`EmptySD3LatentImage`) must have
   the same aspect ratio as the source image. Training pairs are same-size; a
   mismatched AR is out of distribution and visibly degrades identity/preservation.
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

## License / credits

Nodes: Apache-2.0. The **Krea 2 Identity Edit** weights ship separately under the
Krea 2 Community License Agreement (see the model card, `LICENSE.pdf`, and `NOTICE`
in the weights repo).
Built on Krea 2 by Krea AI; text encoder Qwen3-VL (Alibaba).
