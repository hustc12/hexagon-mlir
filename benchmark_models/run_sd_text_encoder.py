# ===- run_sd_text_encoder.py -----------------------------------------------===
#
# Stable Diffusion — Text Encoder (CLIP) Phase-4 harness.
# Main = published CLIP structure; debug shrinks via debug_running/run_sd_text_encoder_debug.py
#
# ===------------------------------------------------------------------------===

from __future__ import annotations

import argparse
import types
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.activations import GELUActivation

from sd_utils import (
    SD_MODEL_ID,
    compile_to_linalg,
    hex_execution,
    x86_execution,
    compare,
    add_phase4_cli,
    options_from_args,
    process_lwp,
)


def get_text_inputs(tokenizer, prompt: str, max_length: int = None):
    if max_length is None:
        max_length = tokenizer.model_max_length
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_input.input_ids


def _patch_gelu_tanh(model):
    def _tanh_forward(self, x):
        return F.gelu(x, approximate="tanh")

    for module in model.modules():
        if isinstance(module, GELUActivation):
            module.forward = types.MethodType(_tanh_forward, module)


def customize_clip_config(config):
    """Identity hook. Debug scripts may shrink layers/hidden."""
    return config


def customize_seq_len(default: int = 77) -> int:
    """Published CLIP context length (77). Debug may shrink for DSP stack."""
    return default


def load_clip(config):
    return CLIPTextModel(config)


def test_text_encoder(
    enablelwp: bool = False,
    enable_hexkl: bool = False,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
    enable_omnifetch_items_1_7: bool = False,
):
    print("\n=== Stable Diffusion — Text Encoder ===")

    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
    config = CLIPTextModel.config_class.from_pretrained(
        SD_MODEL_ID, subfolder="text_encoder"
    )
    config = customize_clip_config(config)
    print(
        f"[Config] layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads} vocab={config.vocab_size}"
    )

    _clip = load_clip(config)
    _clip.eval()
    _patch_gelu_tanh(_clip)

    class CLIPWrapper(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            return self.clip(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

    model = CLIPWrapper(_clip).eval()

    seq_len = customize_seq_len(77)
    input_ids = get_text_inputs(
        tokenizer, "A beautiful picture of a Hexagon NPU", max_length=seq_len
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    position_ids = model.clip.text_model.embeddings.position_ids[
        :, :seq_len
    ].detach()
    print(
        f"input_ids: {input_ids.shape} attention_mask: {attention_mask.shape} "
        f"position_ids: {position_ids.shape} HexKL={enable_hexkl}"
    )

    print("\nCompiling Text Encoder to linalg …")
    module = compile_to_linalg(model, input_ids, attention_mask)

    class _Args:
        pass

    args = _Args()
    args.lwp = enablelwp
    args.enable_hexkl = enable_hexkl
    args.enable_omnifetch_vdae = enable_omnifetch_vdae
    args.disable_layout_aware = not enable_omnifetch_layout_aware
    args.omnifetch_lookahead = omnifetch_lookahead
    args.disable_omnifetch_adaptive = not enable_omnifetch_adaptive
    args.enable_omnifetch_items_1_7 = enable_omnifetch_items_1_7
    options = options_from_args(args)

    print("Running Text Encoder on Hexagon NPU …")
    hex_out = hex_execution(
        module, "CLIPWrapper", [position_ids, input_ids, attention_mask], options
    )
    print("Running reference on x86 …")
    x86_out = x86_execution(model, input_ids, attention_mask)
    compare(hex_out, x86_out, atol=0.05, fail_on_mismatch=True)
    if enablelwp:
        process_lwp()
    print("\nText Encoder test PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD Text Encoder Hexagon benchmark")
    add_phase4_cli(parser)
    args = parser.parse_args()
    test_text_encoder(
        enablelwp=args.lwp,
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        enable_omnifetch_items_1_7=args.enable_omnifetch_items_1_7,
    )
