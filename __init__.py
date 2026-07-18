"""ComfyUI-Krea2Edit — in-context edit forward for the Krea2 model.

ComfyUI's native Krea2 `_forward` is text-to-image only: it builds the sequence
`[text | target]`. The krea2_edit LoRA (trained in ai-toolkit) needs the *appearance
path*: the VAE-encoded SOURCE latent prepended as a block of clean tokens, distinguished
from the (noisy) target purely by the 3-axis RoPE frame index (source=1, target=0, h/w
aligned). This node adds that by wrapping the model's DIFFUSION_MODEL forward and rebuilding
the sequence as `[text | source(frame=1) | target(frame=0)]`, keeping only the target tokens
out — mirroring ai-toolkit's `predict_velocity_edit` exactly, using the model's own submodules.

Wiring:  LoadImage -> VAEEncode(source) --\
                                            Krea2EditModelPatch(model, source_latent) -> KSampler
         UNETLoader -> LoraLoaderModelOnly -/
KSampler.latent_image <- EmptySD3LatentImage (noise). Text and optional phrase weights come
from Krea2EditGroundedEncode.
"""
import logging
import math
import re
import types

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.patcher_extension
import comfy.utils
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import attention_pytorch, optimized_attention_masked


logger = logging.getLogger(__name__)

_PROMPT_WEIGHT_RE = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
_QWEN_IM_START = 151644
_QWEN_USER = 872
_QWEN_NEWLINE = 198
_QWEN_IM_END = 151645


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    """Stride-1 integer positions at a centered integer offset. For `fit` refs the
    pixels are already resampled to target grid density, so the position grid is
    stride-1 BY CONSTRUCTION — scaling it again only manufactures skip/collision
    artifacts. Requires gh<=th, gw<=tw (guaranteed by the floor+cap in fit)."""
    off_h, off_w = max(0, (th - gh) // 2), max(0, (tw - gw) // 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


def _to_4d(v):
    """(B,C,T,H,W) -> (B*T,C,H,W); pass 4D through. Images use T=1."""
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _parse_prompt_weights(text):
    """Return prompt text without weight markup and its ``(phrase, weight)`` terms."""
    terms = []

    def replace(match):
        terms.append((match.group(1).strip(), float(match.group(2))))
        return match.group(1)

    return _PROMPT_WEIGHT_RE.sub(replace, text), terms


def _token_ids(tokenized):
    """Read the first token batch from a ComfyUI CLIP token dictionary."""
    key = next(iter(tokenized))
    return [token[0] for token in tokenized[key][0]]


def _user_content_span(ids):
    """Locate the Qwen chat user turn inside a token-id sequence."""
    for index in range(len(ids) - 2):
        if (ids[index], ids[index + 1], ids[index + 2]) == (
            _QWEN_IM_START,
            _QWEN_USER,
            _QWEN_NEWLINE,
        ):
            end = index + 3
            while end < len(ids) and ids[end] != _QWEN_IM_END:
                end += 1
            return index + 3, end
    return None, None


def _subsequence_starts(sequence, subsequence, start, end):
    if not subsequence:
        return []
    width = len(subsequence)
    return [
        index
        for index in range(start, end - width + 1)
        if sequence[index:index + width] == subsequence
    ]


def _phrase_token_ids(clip, phrase):
    ids = _token_ids(clip.tokenize(phrase))
    start, end = _user_content_span(ids)
    return [] if start is None else ids[start:end]


def _build_krea2_token_weights(clip, tokenized, conditioning, terms, strength):
    """Map parsed prompt terms onto positions in Krea2's visible conditioning."""
    if not terms:
        return []

    ids = _token_ids(tokenized)
    conditioning_length = conditioning[0][0].shape[1]

    # Krea2 removes the system/user prefix from its conditioning. Image placeholders,
    # when present, expand during encoding; the length delta accounts for that expansion.
    visible_start = len(ids) - conditioning_length
    content_start, content_end = _user_content_span(ids)
    if content_start is None:
        content_start, content_end = visible_start, len(ids)

    mapped = []
    for phrase, weight in terms:
        if weight > 1.0:
            value_factor = 1.0
            attention_bias = 2.0 * strength * (weight - 1.0)
        else:
            value_factor = 1.0 + strength * (weight - 1.0)
            attention_bias = 0.0

        raw_positions = set()
        # A word at the start of the prompt and the same word later in a sentence can
        # have different Qwen BPE tokens. Check both forms and merge their matches.
        for variant in (" " + phrase, phrase):
            phrase_ids = _phrase_token_ids(clip, variant)
            for match_start in _subsequence_starts(
                ids, phrase_ids, content_start, content_end
            ):
                raw_positions.update(
                    range(match_start, match_start + len(phrase_ids))
                )

        conditioning_positions = sorted(
            position - visible_start
            for position in raw_positions
            if 0 <= position - visible_start < conditioning_length
        )
        if not conditioning_positions:
            logger.warning(
                "Krea2 prompt-weight phrase %r was not found; skipping it", phrase
            )
            continue

        mapped.extend(
            (position, value_factor, attention_bias)
            for position in conditioning_positions
        )

    return mapped


def _positive_conditioning_rows(transformer_options, batch_size):
    """Expand Comfy's per-conditioning branch labels to attention batch rows."""
    labels = transformer_options.get("cond_or_uncond")
    if labels is None:
        # Direct calls and older Comfy versions have no branch metadata. Preserve
        # the original apply-to-all behavior in that compatibility path.
        return [True] * batch_size
    labels = list(labels)
    if not labels or batch_size % len(labels) != 0:
        logger.warning(
            "Krea2 prompt weights skipped: attention batch %d cannot be mapped "
            "to cond_or_uncond=%r",
            batch_size,
            labels,
        )
        return [False] * batch_size
    rows_per_conditioning = batch_size // len(labels)
    return [int(label) == 0 for label in labels for _ in range(rows_per_conditioning)]


def _attention_mask_to_additive(mask, q):
    """Normalize an attention mask to a broadcastable additive SDPA mask."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        additive = q.new_zeros(mask.shape).masked_fill_(~mask, float("-inf"))
    else:
        additive = mask.to(device=q.device, dtype=q.dtype)
    if additive.ndim == 1:
        additive = additive.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif additive.ndim == 2:
        additive = additive.unsqueeze(0).unsqueeze(0)
    elif additive.ndim == 3:
        additive = additive.unsqueeze(1)
    elif additive.ndim != 4:
        raise ValueError(
            f"Krea2 attention mask must have 1-4 dimensions, got {additive.ndim}"
        )
    return additive


def _weighted_krea2_attention(self, x, freqs=None, mask=None, transformer_options={}):
    """Krea2 attention with positive-branch per-phrase prompt weighting."""
    q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)

    weights = transformer_options.get("krea2_token_weights")
    positive_rows = (
        _positive_conditioning_rows(transformer_options, v.shape[0])
        if weights
        else [False] * v.shape[0]
    )
    has_positive_rows = any(positive_rows)
    if weights and has_positive_rows:
        v = v.clone()
        row_mask = v.new_tensor(positive_rows).view(v.shape[0], 1, 1)
        for position, value_factor, _ in weights:
            if 0 <= position < v.shape[2] and value_factor != 1.0:
                factor = 1.0 + row_mask * (value_factor - 1.0)
                v[:, :, position] = v[:, :, position] * factor

    q, k = self.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if self.kvheads != self.heads:
        repeats = self.heads // self.kvheads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    has_prompt_bias = (
        weights
        and has_positive_rows
        and any(bias != 0.0 for _, _, bias in weights)
    )
    if has_prompt_bias:
        # (batch, heads=1, queries=1, keys) broadcasts across every head and
        # query while leaving neutral/unconditional rows unchanged.
        prompt_bias = q.new_zeros((q.shape[0], 1, 1, k.shape[2]))
        row_bias = q.new_tensor(positive_rows).view(q.shape[0], 1, 1)
        for position, _, bias in weights:
            if 0 <= position < prompt_bias.shape[-1] and bias != 0.0:
                prompt_bias[..., position].add_(row_bias * bias)

        additive_mask = _attention_mask_to_additive(mask, q)
        if additive_mask is not None:
            prompt_bias = prompt_bias + additive_mask

        # Positive emphasis needs an additive logit bias. PyTorch SDPA accepts it
        # directly; other configured attention backends do not do so consistently.
        out = attention_pytorch(
            q, k, v, self.heads, mask=prompt_bias, skip_reshape=True
        )
    else:
        out = optimized_attention_masked(
            q,
            k,
            v,
            self.heads,
            mask=mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
    return self.wo(out * torch.sigmoid(gate))


def _fit_src(src, H, W):
    """Fit a source latent to the target grid the way TRAINING did: center-crop to
    the target aspect ratio, then resize. A plain interpolate (the pre-fix behavior)
    STRETCHES mixed-AR sources — users saw stretched people whenever their input AR
    differed from the output resolution."""
    sh, sw = src.shape[-2:]
    if (sh, sw) == (H, W):
        return src
    s = max(H / sh, W / sw)
    ch, cw = min(sh, int(round(H / s))), min(sw, int(round(W / s)))
    y0, x0 = (sh - ch) // 2, (sw - cw) // 2
    src = src[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(src.float(), size=(H, W), mode="bilinear")


def _fit_encode_image(image, vae, H, W, cache, key, fit_mode="crop"):
    """Pixel-space source prep (blur-proof path): center-crop the IMAGE to the
    target AR, resize to the exact target pixel grid, VAE-encode. Latent-space
    resizing (the old fallback) softens VAE latents — this path never resizes
    latents at all. Cached per target resolution (encode once, not per step)."""
    key = key + (fit_mode,)
    if key in cache:
        return cache[key]
    print(f"[krea2edit] _fit_encode_image: mode={fit_mode} in={tuple(image.shape)} target_latent={H}x{W}", flush=True)
    px_h, px_w = H * 8, W * 8
    img = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    ih, iw = img.shape[-2:]
    if fit_mode == "fit":
        # "bilinear" answer to scale mismatch: resample CONTENT (pixel space, bicubic)
        # to the target's grid density instead of moving positions. AR-preserving
        # fit-inside, no crop, no grey canvas — the forward places it at an integer
        # centered offset (scaled-pos with s=1 -> stride 1, no rounding artifacts).
        sc = min(px_h / ih, px_w / iw)
        # NEAR-MATCHED AR: fill the target grid EXACTLY via a minimal center-crop.
        # Fit-inside margins of 1-2 tokens are not harmless: target edge columns
        # with no ref correspondence get filled by repeating adjacent ref content
        # (2026-07-14 edge-duplication bug: ref (74,54) vs target (74,56)).
        # This also restores the design promise fit == crop at matched AR.
        CROP_TOL = 0.08
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., y0:y0 + ch, x0:x0 + cw]
            nh, nw = px_h, px_w
        else:
            # genuine AR mismatch: MUST match the trainer's _fit_prep EXACTLY
            # (krea2_edit.py) — /16 floor snap capped at the target's /16 floor.
            # The model is trained on this geometry; a /8-round node grid would
            # produce a different ref latent size -> different centered offset ->
            # a visible margin-boundary seam even from a well-trained model
            # (train/infer geometry must be byte-identical). 2026-07-15 alignment.
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
        img = F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True)
        lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
        cache[key] = lat
        return lat
    # crop (default / "v1 legacy"): center-crop to the target AR, then resize.
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    img = img[..., y0:y0 + ch, x0:x0 + cw]
    img = F.interpolate(img.float(), size=(px_h, px_w), mode="bicubic", antialias=True)
    lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
    cache[key] = lat
    return lat


def _ref_attn_bias(boosts, boost_mask, txtlen, slens, tgtlen, mask_hw, device, dtype):
    """Additive attention-logit bias on the [text | refs... | target] sequence.

    boosts: per-ref factor on target->ref attention, aligned with the source blocks
    (last entry = last ref = the subject by workflow convention). Equivalent to
    multiplying those keys' post-softmax attention weight before renormalization.
    boost_mask (ComfyUI MASK, ref-image pixel space) restricts the LAST ref's boost
    to a region (e.g. the face).
    """
    nsrc = len(slens)
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, sl = offs[i], slens[i]
        if boost_mask is not None and i == nsrc - 1 and mask_hw is not None:
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(mask[None].float(), size=mask_hw[i], mode="area")[0, 0]
            cols = off + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + sl, device=device)
        bias[:, :, rows0:, cols] = math.log(max(b, 1e-4))
    return bias


def krea2_edit_forward(m, x, timesteps, context, src_latent, transformer_options,
                       ref_boost=1.0, ref_boost_a=1.0, ref_boost_mask=None,
                       ref_native=False, pos_mode="anchor"):
    """Krea2 SingleStreamDiT._forward, but with source block(s) prepended.

    m           : the SingleStreamDiT (LoRA-patched at sample time)
    x           : (B,C,H,W) or (B,C,T,H,W) noisy TARGET latent
    src_latent  : clean SOURCE latent (VAE-encoded), 4D/5D — or a LIST of them
                  (multi-ref: [scene, subject], frames 1..N, training-matched)
    context     : (B, seq, txtlayers*txtdim) — the 12-layer Qwen3-VL stack
    """
    patch = m.patch

    # Mirror ComfyUI _forward: latents may arrive 5D (B,C,T,H,W) for this model.
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    # source(s) -> (bs, C, H, W): flatten temporal, match batch, fit to the target grid
    # (center-crop to target AR then resize — training-matched; never stretch).
    src_list = src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
    srcs = []
    for sl in src_list:
        src = _to_4d(sl).to(x.device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        if not ref_native and src.shape[-2:] != (H, W):
            print(f"[krea2edit] LATENT-PATH fit_src (crop): src={tuple(src.shape[-2:])} -> {H}x{W}", flush=True)
            src = _fit_src(src, H, W).to(x.dtype)
        srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch), padding_mode="replicate"))
    src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

    context = m._unpack_context(context)                       # (B, seq, 12, 2560)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    src_imgs = [m.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                for s_ in srcs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    srclen = sum(si.shape[1] for si in src_imgs)
    combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)  # [text | refs... | target]

    device = combined.device
    if pos_mode == "stride1" and ref_native:
        print(f"[krea2edit] STRIDE1-POS fit: ref grids {src_grids} centered in ({h_},{w_})", flush=True)
        if any(h_ - gh > 2 or w_ - gw > 2 for gh, gw in src_grids):
            print("[krea2edit] NOTE: fit margins >2 tokens (large source/output aspect-ratio "
                  "gap). fit is trained for matched/near-matched AR; for a big AR change "
                  "prefer 'crop', or set the output AR closer to the source.", flush=True)
        ref_ids = [_imgids_offset(bs, i + 1, gh, gw, h_, w_, device)
                   for i, (gh, gw) in enumerate(src_grids)]
    else:
        ref_ids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(src_grids)]
    pos = torch.cat([
        torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]   # text @ 0
        + ref_ids
        + [_imgids(bs, 0, h_, w_, device)],                                    # target frame=0
        dim=1)
    freqs = m.pe_embedder(pos)

    attn_bias = None
    if ref_boost != 1.0 or ref_boost_a != 1.0:
        # last ref = subject (single-ref: the only ref); earlier refs (scene) get ref_boost_a
        boosts = [ref_boost_a] * (len(src_imgs) - 1) + [ref_boost]
        attn_bias = _ref_attn_bias(boosts, ref_boost_mask, txtlen,
                                   [si.shape[1] for si in src_imgs], tgtlen,
                                   src_grids, combined.device, combined.dtype)

    for block in m.blocks:
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]         # target tokens only
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class Krea2EditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "source_latent": ("LATENT",),
        }, "optional": {
            "source_latent_b": ("LATENT", {"tooltip": "2nd reference (subject photo) for multi-ref LoRAs -> RoPE frame=2, training-matched order: scene first, subject second"}),
            "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                     "tooltip": "reference-fidelity dial: multiplies target->reference attention. Applies to the LAST ref (= the subject in two-ref workflows, the only ref in single-ref). 1.0 = off, >1 pulls harder toward the reference's appearance, <1 loosens. Optimal value is model-specific"}),
            "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                       "tooltip": "same dial for the FIRST ref (= the scene in two-ref workflows). No effect in single-ref workflows. 1.0 = off"}),
            "fit_mode": (["fit", "crop (legacy)"], {"default": "fit",
                          "tooltip": "how an image source fits a mismatched output aspect ratio (needs vae + source_image connected): fit = resample the source to the target grid at a centered offset — matches how this model was trained (default, use this); crop (legacy) = center-crop to the target AR then resize (v1/v1.1 geometry, only for older weights)"}),
            "ref_boost_mask": ("MASK", {"tooltip": "optional region on the (last) reference to boost, e.g. the face; empty = whole reference"}),
            "vae": ("VAE", {"tooltip": "RECOMMENDED with source_image: enables the blur-proof pixel-space path (crop+resize in pixels, encode internally) — immune to input/output resolution mismatches"}),
            "source_image": ("IMAGE", {"tooltip": "source as IMAGE (with vae connected): overrides source_latent with exact pixel-space fitting — fixes blurry results from mismatched resolutions"}),
            "source_image_b": ("IMAGE", {"tooltip": "2nd reference as IMAGE (with vae)"}),
            "prompt_weights": ("KREA2_PROMPT_WEIGHTS", {"tooltip": "Connect Krea2 Edit (grounded encode).prompt_weights to enable positive-branch (phrase:weight) syntax"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Adds the krea2_edit source-preservation path, reference-fidelity controls, and optional grounded phrase weights to Krea2."

    def patch(self, model, source_latent, source_latent_b=None, ref_boost=1.0, ref_boost_a=1.0,
              ref_boost_mask=None, vae=None, source_image=None,
              source_image_b=None, fit_mode="fit", prompt_weights=None):
        m = model.clone()
        # The target latent reaches the diffusion model already scaled (process_latent_in);
        # scale the source(s) the same way so all share one latent space.
        src_samples = model.model.process_latent_in(source_latent["samples"])
        if source_latent_b is not None:
            src_samples = [src_samples, model.model.process_latent_in(source_latent_b["samples"])]

        px_cache = {}   # pixel-path encoded sources, keyed per target resolution
        mm = model.model  # for process_latent_in on the pixel path

        if fit_mode == "fit" and (vae is None or source_image is None):
            print(f"[krea2edit] WARNING: fit_mode='fit' has NO EFFECT — it needs both "
                  f"'vae' and 'source_image' connected (the pixel path). Falling back to the "
                  f"latent crop path.", flush=True)

        def wrapper(executor, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
            dm = executor.class_obj  # the SingleStreamDiT instance
            src = src_samples
            if vae is not None and source_image is not None:
                if not px_cache:
                    print(f"[krea2edit] pixel path ACTIVE (fit_mode={fit_mode})", flush=True)
                xx = _to_4d(x)
                Hh, Ww = xx.shape[-2], xx.shape[-1]
                lat = mm.process_latent_in(_fit_encode_image(source_image, vae, Hh, Ww, px_cache, ("a", Hh, Ww), fit_mode))
                if source_image_b is not None:
                    lat = [lat, mm.process_latent_in(_fit_encode_image(source_image_b, vae, Hh, Ww, px_cache, ("b", Hh, Ww), fit_mode))]
                src = lat
            v = krea2_edit_forward(dm, x, timesteps, context, src, transformer_options,
                                   ref_boost=ref_boost, ref_boost_a=ref_boost_a,
                                   ref_boost_mask=ref_boost_mask,
                                   ref_native=(fit_mode == "fit" and vae is not None
                                               and source_image is not None),
                                   pos_mode=("stride1" if fit_mode == "fit" else "anchor"))
            return v

        to = m.model_options.get("transformer_options", {}).copy()
        m.model_options["transformer_options"] = to
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit", wrapper, to
        )

        if prompt_weights:
            to["krea2_token_weights"] = prompt_weights
            diffusion_model = m.get_model_object("diffusion_model")
            for index, block in enumerate(diffusion_model.blocks):
                patched_attention = types.MethodType(
                    _weighted_krea2_attention, block.attn
                )
                m.add_object_patch(
                    f"diffusion_model.blocks.{index}.attn.forward",
                    patched_attention,
                )
        return (m,)


class Krea2EditGroundedEncode:
    """Image-grounded instruction encode — the SEMANTIC path of krea2_edit.

    Training always encodes the instruction TOGETHER with the source image through
    Qwen3-VL (user turn = <vision tokens: source> + instruction) and taps 12 layers.
    Stock CLIPTextEncode is text-only, so inference was running with the grounding
    half of the recipe missing (the VAE source tokens carry appearance; THIS carries
    scene semantics: "the man on the left", "the sign in the back").

    Requires a qwen3vl TE checkpoint WITH the vision tower (all local ones have it).
    grounding_px caps the longest side fed to the VLM — the 2026-07-02 LoRA trained
    with 384-768px jitter, so 640-768 is in-distribution; 0 = native res (the jitter
    training makes that tolerable too). For CFG, ground the NEGATIVE too: second node,
    empty prompt, same image (matches training's unconditional).

    ``(phrase:weight)`` markup can be emitted through the second output and connected
    to Krea2EditModelPatch. It affects only positive/edit conditioning rows.
    """
    DEFAULT_SYSTEM = (
        "Describe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
    )

    KREA2_EDIT_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "{}<|im_end|>\n<|im_start|>assistant\n"
    )

    @classmethod
    def _template(cls, nimg, system_prompt=""):
        sp = system_prompt.strip() or cls.DEFAULT_SYSTEM
        vis = "<|vision_start|><|image_pad|><|vision_end|>" * nimg
        return ("<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
                + vis + "{}<|im_end|>\n<|im_start|>assistant\n")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_b": ("IMAGE", {"tooltip": "2nd reference (subject) for multi-ref LoRAs; vision blocks in training order: scene, subject"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64,
                                          "tooltip": "cap longest side fed to Qwen3-VL; 0 = native"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "advanced (optional): override the grounding system prompt (empty = training default). Steers what the vision encoder attends to, e.g. facial identity detail."}),
                "weight_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                                "tooltip": "Global strength for positive-branch (phrase:weight); connect prompt_weights to the source patch"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "KREA2_PROMPT_WEIGHTS")
    RETURN_NAMES = ("conditioning", "prompt_weights")
    FUNCTION = "encode"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Encodes a grounded edit instruction and emits optional positive-branch per-phrase prompt weights."

    KREA2_EDIT_TEMPLATE_2REF = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "{}<|im_end|>\n<|im_start|>assistant\n"
    )

    def _prep(self, image, grounding_px):
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            s = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(self, clip, prompt, image=None, image_b=None, grounding_px=768,
               system_prompt="", weight_strength=1.0):
        clean_prompt, terms = _parse_prompt_weights(prompt)
        if image is None:  # text-only fallback = old behavior
            tokens = clip.tokenize(clean_prompt)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            weights = _build_krea2_token_weights(
                clip, tokens, conditioning, terms, weight_strength
            )
            return conditioning, weights
        imgs = [self._prep(image, grounding_px)]
        if image_b is not None:
            imgs.append(self._prep(image_b, grounding_px))
        template = self._template(len(imgs), system_prompt)
        tokens = clip.tokenize(clean_prompt, images=imgs, llama_template=template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        weights = _build_krea2_token_weights(
            clip, tokens, conditioning, terms, weight_strength
        )
        return conditioning, weights


NODE_CLASS_MAPPINGS = {
    "Krea2EditModelPatch": Krea2EditModelPatch,
    "Krea2EditGroundedEncode": Krea2EditGroundedEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditModelPatch": "Krea2 Edit (source patch)",
    "Krea2EditGroundedEncode": "Krea2 Edit (grounded encode)",
}
