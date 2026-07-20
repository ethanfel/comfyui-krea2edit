import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _install_comfy_stubs():
    """Install the small ComfyUI surface imported by the node module."""
    comfy = types.ModuleType("comfy")
    patcher_extension = types.ModuleType("comfy.patcher_extension")
    utils = types.ModuleType("comfy.utils")
    ldm = types.ModuleType("comfy.ldm")
    common_dit = types.ModuleType("comfy.ldm.common_dit")
    flux = types.ModuleType("comfy.ldm.flux")
    flux_layers = types.ModuleType("comfy.ldm.flux.layers")
    flux_math = types.ModuleType("comfy.ldm.flux.math")
    modules = types.ModuleType("comfy.ldm.modules")
    attention = types.ModuleType("comfy.ldm.modules.attention")

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"

    class WrapperExecutor:
        """Minimal Comfy-compatible wrapper executor used by regression tests."""

        def __init__(self, original, class_obj, wrappers, idx):
            self.original = original
            self.class_obj = class_obj
            self.wrappers = wrappers.copy()
            self.idx = idx
            self.is_last = idx == len(wrappers)

        def execute(self, *args, **kwargs):
            if self.is_last:
                return self.original(*args, **kwargs)
            return self.wrappers[self.idx](self, *args, **kwargs)

    patcher_extension.WrappersMP = WrappersMP
    patcher_extension.WrapperExecutor = WrapperExecutor
    patcher_extension.calls = []
    attention.calls = []

    def add_wrapper_with_key(wrapper_type, key, wrapper, options):
        patcher_extension.calls.append((wrapper_type, key, wrapper, options))
        wrappers = options.setdefault("wrappers", {})
        wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    patcher_extension.add_wrapper_with_key = add_wrapper_with_key
    utils.common_upscale = lambda samples, width, height, *_args: (
        torch.nn.functional.interpolate(
            samples.float(), size=(height, width), mode="area"
        ).to(samples.dtype)
    )
    common_dit.pad_to_patch_size = lambda value, _patch, **_kwargs: value
    flux_layers.timestep_embedding = lambda timesteps, dim: torch.zeros(
        timesteps.shape[0], dim, device=timesteps.device
    )
    flux_math.apply_rope = lambda q, k, _freqs: (q, k)

    def fake_attention(path, q, k, v, heads, mask=None, **_kwargs):
        attention.calls.append(
            {
                "path": path,
                "q": q.detach().clone(),
                "k": k.detach().clone(),
                "v": v.detach().clone(),
                "heads": heads,
                "mask": None if mask is None else mask.detach().clone(),
            }
        )
        # Deterministic, shape-compatible output: every query receives mean(V).
        out = v.mean(dim=2, keepdim=True).expand_as(q)
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    attention.attention_pytorch = (
        lambda q, k, v, heads, mask=None, **kwargs: fake_attention(
            "pytorch", q, k, v, heads, mask=mask, **kwargs
        )
    )
    attention.optimized_attention = (
        lambda q, k, v, heads, mask=None, **kwargs: fake_attention(
            "optimized", q, k, v, heads, mask=mask, **kwargs
        )
    )
    attention.optimized_attention_masked = (
        lambda q, k, v, heads, mask=None, **kwargs: fake_attention(
            "optimized_masked", q, k, v, heads, mask=mask, **kwargs
        )
    )

    comfy.patcher_extension = patcher_extension
    comfy.utils = utils
    comfy.ldm = ldm
    ldm.common_dit = common_dit
    ldm.flux = flux
    ldm.modules = modules
    flux.layers = flux_layers
    flux.math = flux_math
    modules.attention = attention

    sys.modules.update(
        {
            "comfy": comfy,
            "comfy.patcher_extension": patcher_extension,
            "comfy.utils": utils,
            "comfy.ldm": ldm,
            "comfy.ldm.common_dit": common_dit,
            "comfy.ldm.flux": flux,
            "comfy.ldm.flux.layers": flux_layers,
            "comfy.ldm.flux.math": flux_math,
            "comfy.ldm.modules": modules,
            "comfy.ldm.modules.attention": attention,
        }
    )
    return patcher_extension, attention


PATCHER_EXTENSION, ATTENTION = _install_comfy_stubs()


@pytest.fixture(scope="session")
def node_module():
    spec = importlib.util.spec_from_file_location(
        "krea2edit_under_test", ROOT / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clear_recorded_calls():
    PATCHER_EXTENSION.calls.clear()
    ATTENTION.calls.clear()
    yield
    PATCHER_EXTENSION.calls.clear()
    ATTENTION.calls.clear()
