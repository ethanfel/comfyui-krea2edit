import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from conftest import ATTENTION, PATCHER_EXTENSION


ROOT = Path(__file__).resolve().parents[1]
IM_START = 151644
USER = 872
NEWLINE = 198
IM_END = 151645
VISION_START = 151652
VISION_END = 151653


class FakeClip:
    """Qwen-like tokenizer whose image placeholders expand during encoding."""

    WORD_IDS = {
        "make": 10,
        "red": 20,
        "fox": 30,
        "and": 40,
        "blue": 50,
        "beside": 60,
        "fog": 70,
        "otter": 80,
    }
    IMAGE_TOKENS = 4

    def __init__(self):
        self.tokenize_calls = []
        self.encoded = []

    def _word_ids(self, text):
        ids = []
        for match in re.finditer(r"\S+", text):
            token_id = self.WORD_IDS[match.group().strip(".,")]
            # Model the BPE distinction between a prompt-initial token and the
            # same word carrying a leading-space marker later in a sentence.
            if match.start() > 0 and text[match.start() - 1].isspace():
                token_id += 100
            ids.append(token_id)
        return ids

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append((text, kwargs))
        ids = [101, IM_START, USER, NEWLINE]
        for image in kwargs.get("images", []):
            ids.extend(
                [
                    VISION_START,
                    {
                        "type": "image",
                        "data": image,
                        "expanded_tokens": self.IMAGE_TOKENS,
                    },
                    VISION_END,
                ]
            )
        ids.extend(self._word_ids(text))
        ids.extend([IM_END, 102])
        return {"qwen3vl_4b": [[(token_id, 1.0) for token_id in ids]]}

    def encode_from_tokens_scheduled(self, tokenized):
        row = next(iter(tokenized.values()))[0]
        ids = [entry[0] for entry in row]
        visible_start = ids.index(NEWLINE) + 1
        expanded_length = 0
        for token_id in ids[visible_start:]:
            if isinstance(token_id, dict):
                expanded_length += token_id["expanded_tokens"]
            else:
                expanded_length += 1
        conditioning = [
            [torch.zeros(1, expanded_length, 8), {"sentinel": "conditioning"}]
        ]
        self.encoded.append((tokenized, conditioning))
        return conditioning


def _encode_for_mapping(clip, text, **tokenize_kwargs):
    tokenized = clip.tokenize(text, **tokenize_kwargs)
    conditioning = clip.encode_from_tokens_scheduled(tokenized)
    return tokenized, conditioning


def test_parse_plain_prompt_is_unchanged(node_module):
    clean, terms = node_module._parse_prompt_weights("make the fox blue")

    assert clean == "make the fox blue"
    assert terms == []


def test_parse_weighted_phrases_removes_only_weight_markup(node_module):
    clean, terms = node_module._parse_prompt_weights(
        "make the (red fox:1.75) blue and remove (fog:-0.5)"
    )

    assert clean == "make the red fox blue and remove fog"
    assert terms == [("red fox", 1.75), ("fog", -0.5)]


def test_mapping_weights_every_occurrence_and_every_token(node_module):
    clip = FakeClip()
    tokenized, conditioning = _encode_for_mapping(clip, "red fox and red fox")

    weights = node_module._build_krea2_token_weights(
        clip, tokenized, conditioning, [("red fox", 0.5)], strength=1.0
    )

    assert weights == [
        (0, 0.5, 0.0),
        (1, 0.5, 0.0),
        (3, 0.5, 0.0),
        (4, 0.5, 0.0),
    ]


def test_mapping_skips_a_phrase_that_is_not_in_the_prompt(node_module):
    clip = FakeClip()
    tokenized, conditioning = _encode_for_mapping(clip, "make fox blue")

    weights = node_module._build_krea2_token_weights(
        clip, tokenized, conditioning, [("red fox", -1.0)], strength=1.0
    )

    assert weights == []


@pytest.mark.parametrize(
    ("weight", "strength", "expected"),
    [
        (2.0, 0.5, (1.0, 1.0)),
        (1.0, 0.5, (1.0, 0.0)),
        (0.0, 0.5, (0.5, 0.0)),
        (-1.0, 0.5, (0.0, 0.0)),
    ],
)
def test_weight_factor_rules(node_module, weight, strength, expected):
    clip = FakeClip()
    tokenized, conditioning = _encode_for_mapping(clip, "fox")

    weights = node_module._build_krea2_token_weights(
        clip, tokenized, conditioning, [("fox", weight)], strength=strength
    )

    assert len(weights) == 1
    assert weights[0][0] == 0
    assert weights[0][1:] == pytest.approx(expected)


class TinyAttention:
    heads = 1
    kvheads = 1

    def wq(self, value):
        return value

    def wk(self, value):
        return value

    def wv(self, value):
        return value

    def gate(self, value):
        return torch.zeros_like(value)

    def qknorm(self, q, k):
        return q, k

    def wo(self, value):
        return value


def test_weighted_attention_dispatch_and_effects(node_module):
    attention = TinyAttention()
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    baseline = node_module._weighted_krea2_attention(
        attention, x, transformer_options={}
    )
    assert baseline.shape == x.shape
    assert [call["path"] for call in ATTENTION.calls] == ["optimized_masked"]

    ATTENTION.calls.clear()
    identity = node_module._weighted_krea2_attention(
        attention,
        x,
        transformer_options={"krea2_token_weights": [(1, 1.0, 0.0)]},
    )
    assert torch.equal(identity, baseline)
    assert [call["path"] for call in ATTENTION.calls] == ["optimized_masked"]

    ATTENTION.calls.clear()
    de_emphasized = node_module._weighted_krea2_attention(
        attention,
        x,
        transformer_options={"krea2_token_weights": [(1, 0.0, 0.0)]},
    )
    assert not torch.equal(de_emphasized, baseline)
    assert [call["path"] for call in ATTENTION.calls] == ["optimized_masked"]

    ATTENTION.calls.clear()
    node_module._weighted_krea2_attention(
        attention,
        x,
        transformer_options={"krea2_token_weights": [(1, 1.0, 2.5)]},
    )
    assert [call["path"] for call in ATTENTION.calls] == ["pytorch"]
    assert torch.equal(
        ATTENTION.calls[0]["mask"],
        torch.tensor([[[[0.0, 2.5, 0.0]]]]),
    )


def test_prompt_value_weights_apply_only_to_positive_cfg_batch_rows(node_module):
    x = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2) + 1.0

    node_module._weighted_krea2_attention(
        TinyAttention(),
        x,
        transformer_options={
            "krea2_token_weights": [(1, 0.0, 0.0)],
            # Two conditioning chunks with a latent batch of two in each chunk.
            "cond_or_uncond": [0, 1],
        },
    )

    values = ATTENTION.calls[0]["v"]
    assert torch.equal(values[:2, :, 1], torch.zeros_like(values[:2, :, 1]))
    assert torch.equal(values[2:, :, 1], x[2:, 1].unsqueeze(1))


def test_prompt_bias_uses_cond_labels_instead_of_assuming_first_batch(node_module):
    x = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2) + 1.0

    node_module._weighted_krea2_attention(
        TinyAttention(),
        x,
        transformer_options={
            "krea2_token_weights": [(1, 1.0, 2.5)],
            "cond_or_uncond": [1, 0],
        },
    )

    call = ATTENTION.calls[0]
    assert call["path"] == "pytorch"
    assert torch.equal(
        call["mask"],
        torch.tensor(
            [
                [[[0.0, 0.0, 0.0]]],
                [[[0.0, 2.5, 0.0]]],
            ]
        ),
    )


def test_prompt_weights_are_skipped_for_a_separate_neutral_cfg_call(node_module):
    x = torch.arange(3 * 2, dtype=torch.float32).reshape(1, 3, 2) + 1.0
    baseline = node_module._weighted_krea2_attention(
        TinyAttention(), x, transformer_options={}
    )

    ATTENTION.calls.clear()
    neutral = node_module._weighted_krea2_attention(
        TinyAttention(),
        x,
        transformer_options={
            "krea2_token_weights": [(1, 0.0, 3.0)],
            "cond_or_uncond": [1],
        },
    )

    assert torch.equal(neutral, baseline)
    assert [call["path"] for call in ATTENTION.calls] == ["optimized_masked"]
    assert ATTENTION.calls[0]["mask"] is None


def test_prompt_bias_composes_with_ref_boost_attention_mask(node_module):
    x = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2) + 1.0
    ref_mask = node_module._ref_attn_bias(
        [2.0], None, txtlen=2, slens=[1], tgtlen=1, mask_hw=None,
        device=x.device, dtype=x.dtype
    )

    node_module._weighted_krea2_attention(
        TinyAttention(),
        x,
        mask=ref_mask,
        transformer_options={
            "krea2_token_weights": [(1, 1.0, 2.5)],
            "cond_or_uncond": [1, 0],
        },
    )

    call = ATTENTION.calls[0]
    assert call["path"] == "pytorch"
    expected = ref_mask.expand(2, -1, -1, -1).clone()
    expected[1, :, :, 1] += 2.5
    assert torch.equal(call["mask"], expected)
    assert call["mask"][0, 0, 3, 2] == pytest.approx(math.log(2.0))


@pytest.mark.parametrize(
    ("image_count", "expected_position"),
    [
        (1, 7),
        (2, 13),
    ],
)
def test_grounded_encode_forwards_images_and_aligns_expanded_vision_tokens(
    node_module, image_count, expected_position
):
    clip = FakeClip()
    node = node_module.Krea2EditGroundedEncode()
    image = torch.zeros(1, 2, 3, 3)
    image_b = torch.ones(1, 2, 3, 3) if image_count == 2 else None

    conditioning, weights = node.encode(
        clip,
        "make (fox:0) blue",
        image=image,
        image_b=image_b,
        grounding_px=768,
        system_prompt="Focus on faces and clothing.",
        weight_strength=1.0,
    )

    clean_text, kwargs = clip.tokenize_calls[0]
    assert clean_text == "make fox blue"
    assert len(kwargs["images"]) == image_count
    assert kwargs["llama_template"] == node._template(
        image_count, "Focus on faces and clothing."
    )
    assert conditioning is clip.encoded[0][1]
    assert weights == [(expected_position, 0.0, 0.0)]


def test_grounded_encode_preserves_positional_system_prompt_contract(node_module):
    clip = FakeClip()
    image = torch.zeros(1, 2, 3, 3)

    # `weight_strength` was appended after upstream's `system_prompt`; this old
    # positional call must continue to treat its sixth argument as text.
    _conditioning, weights = node_module.Krea2EditGroundedEncode().encode(
        clip, "make (fox:0) blue", image, None, 768, "Focus on the face."
    )

    clean_text, kwargs = clip.tokenize_calls[0]
    assert clean_text == "make fox blue"
    assert kwargs["llama_template"] == node_module.Krea2EditGroundedEncode._template(
        1, "Focus on the face."
    )
    assert weights == [(7, 0.0, 0.0)]


def test_grounded_encode_public_contract_keeps_conditioning_first(node_module):
    node = node_module.Krea2EditGroundedEncode
    assert node.RETURN_TYPES == ("CONDITIONING", "KREA2_PROMPT_WEIGHTS")
    assert node.RETURN_NAMES == ("conditioning", "prompt_weights")
    optional = node.INPUT_TYPES()["optional"]
    assert "system_prompt" in optional
    assert "weight_strength" in optional
    assert list(optional).index("system_prompt") < list(optional).index("weight_strength")

    clip = FakeClip()
    conditioning, weights = node().encode(
        clip, "make fox blue", system_prompt="kept for API compatibility"
    )
    assert conditioning is clip.encoded[0][1]
    assert weights == []


class FakeDiffusionModel:
    def __init__(self, count=3):
        self.blocks = [SimpleNamespace(attn=SimpleNamespace()) for _ in range(count)]


class FakeInnerModel:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model

    def process_latent_in(self, samples):
        return samples + 1


class FakeModel:
    def __init__(self, diffusion_model=None, transformer_options=None):
        diffusion_model = diffusion_model or FakeDiffusionModel()
        self.model = FakeInnerModel(diffusion_model)
        self.model_options = {
            "transformer_options": dict(
                transformer_options or {"existing_option": "preserved"}
            )
        }
        self.object_patches = []
        self.cloned = None

    def clone(self):
        clone = FakeModel(
            self.model.diffusion_model,
            self.model_options.get("transformer_options", {}),
        )
        self.cloned = clone
        return clone

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.model.diffusion_model

    def add_object_patch(self, path, value):
        self.object_patches.append((path, value))


def test_model_patch_adds_prompt_attention_without_losing_upstream_wrapper(node_module):
    model = FakeModel()
    prompt_weights = [(7, 0.0, 0.0), (8, 1.0, 1.5)]

    (patched,) = node_module.Krea2EditModelPatch().patch(
        model,
        {"samples": torch.zeros(1, 2, 2, 2)},
        prompt_weights=prompt_weights,
    )

    assert patched is model.cloned
    options = patched.model_options["transformer_options"]
    assert options["existing_option"] == "preserved"
    assert options["krea2_token_weights"] == prompt_weights
    assert len(PATCHER_EXTENSION.calls) == 1
    wrapper_type, key, wrapper, wrapper_options = PATCHER_EXTENSION.calls[0]
    assert wrapper_type == PATCHER_EXTENSION.WrappersMP.DIFFUSION_MODEL
    assert key == "krea2_edit"
    assert callable(wrapper)
    assert wrapper_options is options
    assert [path for path, _patch in patched.object_patches] == [
        "diffusion_model.blocks.0.attn.forward",
        "diffusion_model.blocks.1.attn.forward",
        "diffusion_model.blocks.2.attn.forward",
    ]
    assert all(callable(patch) for _path, patch in patched.object_patches)


def test_model_patch_without_weights_keeps_native_attention_path(node_module):
    model = FakeModel()

    (patched,) = node_module.Krea2EditModelPatch().patch(
        model, {"samples": torch.zeros(1, 2, 2, 2)}
    )

    options = patched.model_options["transformer_options"]
    assert options["existing_option"] == "preserved"
    assert "krea2_token_weights" not in options
    assert patched.object_patches == []
    assert len(PATCHER_EXTENSION.calls) == 1
    assert PATCHER_EXTENSION.calls[0][1] == "krea2_edit"
    optional = node_module.Krea2EditModelPatch.INPUT_TYPES()["optional"]
    assert "prompt_weights" in optional
    assert "ref_boost" in optional
    assert "ref_boost_mask" in optional
    assert "system_prompt" not in optional


def test_unified_workflow_connects_positive_prompt_weights_to_model_patch():
    workflow = json.loads((ROOT / "workflows" / "krea2_identity_edit.json").read_text())
    model_patch = next(
        node for node in workflow["nodes"] if node["type"] == "Krea2EditModelPatch"
    )
    positive_encoder = next(node for node in workflow["nodes"] if node["id"] == 84)
    negative_encoder = next(node for node in workflow["nodes"] if node["id"] == 85)

    model_input = next(
        item for item in model_patch["inputs"] if item["name"] == "prompt_weights"
    )
    positive_output = next(
        output
        for output in positive_encoder["outputs"]
        if output["name"] == "prompt_weights"
    )
    negative_output = next(
        output
        for output in negative_encoder["outputs"]
        if output["name"] == "prompt_weights"
    )
    assert positive_output["links"] == [model_input["link"]]
    assert negative_output["links"] is None

    link = next(item for item in workflow["links"] if item[0] == model_input["link"])
    assert link[1:3] == [positive_encoder["id"], 1]
    assert link[3] == model_patch["id"]
    assert link[4] == 7
    assert link[5] == "KREA2_PROMPT_WEIGHTS"
