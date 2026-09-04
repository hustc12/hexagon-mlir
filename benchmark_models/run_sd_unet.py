# ===- run_sd_unet.py ------------------------------------------------------===
#
# Stable Diffusion — UNet Phase-4 harness.
# Main keeps published UNet config hooks; debug shrinks channels/blocks.
#
# ===------------------------------------------------------------------------===

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

_orig_gelu = F.gelu


def _tanh_gelu(input, approximate="none"):
    return _orig_gelu(input, approximate="tanh")


F.gelu = _tanh_gelu

from diffusers import UNet2DConditionModel

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


def customize_unet_config(config: dict) -> dict:
    """Identity hook. Debug scripts may shrink channels/blocks."""
    return config


def customize_unet_inputs(config: dict):
    """Return (latent, timestep, encoder_hidden_states) matching config."""
    sample = int(config.get("sample_size", 64))
    cross = int(config.get("cross_attention_dim", 768))
    latent = torch.rand(1, 4, sample, sample, dtype=torch.float32)
    timestep = torch.tensor([1.0], dtype=torch.float32)
    encoder = torch.rand(1, 77, cross, dtype=torch.float32)
    return latent, timestep, encoder


def load_unet(config):
    return UNet2DConditionModel.from_config(config)


def customize_phase4_options(options: dict, _module) -> dict:
    """Identity hook for Debug topology-specific option gating."""
    return options


def test_unet(
    enablelwp: bool = False,
    enable_hexkl: bool = False,
    enable_alps_vdae: bool = False,
    enable_alps_layout_aware: bool = True,
    alps_lookahead: int = 2,
    enable_alps_adaptive: bool = True,
    enable_alps_items_1_7: bool = False,
):
    print("\n=== Stable Diffusion — UNet ===")
    config = UNet2DConditionModel.load_config(SD_MODEL_ID, subfolder="unet")
    config = customize_unet_config(config)
    print(
        f"[Config] blocks={config.get('down_block_types')} "
        f"channels={config.get('block_out_channels')} "
        f"sample={config.get('sample_size')} cross={config.get('cross_attention_dim')}"
    )

    model = load_unet(config)
    model.eval()
    latent, timestep, encoder = customize_unet_inputs(config)
    inputs = (latent, timestep, encoder)
    print(
        f"Inputs — latent: {latent.shape}, timestep: {timestep.shape}, "
        f"encoder: {encoder.shape} HexKL={enable_hexkl}"
    )

    print("\nCompiling UNet to linalg …")
    module = compile_to_linalg(model, *inputs)

    class _Args:
        pass

    args = _Args()
    args.lwp = enablelwp
    args.enable_hexkl = enable_hexkl
    args.enable_alps_vdae = enable_alps_vdae
    args.disable_layout_aware = not enable_alps_layout_aware
    args.alps_lookahead = alps_lookahead
    args.disable_alps_adaptive = not enable_alps_adaptive
    args.enable_alps_items_1_7 = enable_alps_items_1_7
    options = options_from_args(args)
    module_text = str(module)
    if enable_alps_items_1_7 and not (
        "linalg.matmul" in module_text or "linalg.batch_matmul" in module_text
    ):
        # Persistent-WH replay is meaningful only when a WH cache site exists.
        # Avoid three redundant executions of a conv-only graph on DSP heap.
        options["enableAlpsPersistentWhCache"] = False
        print(
            "[AlpsAdaptiveNoop] no matmul/WH-cache site; "
            "disabled persistent-WH replay"
        )
    options = customize_phase4_options(options, module)

    print("Running UNet on Hexagon NPU …")
    hex_out = hex_execution(module, "UNet2DConditionModel", list(inputs), options)
    print("Running reference on x86 …")
    x86_out = x86_execution(model, *inputs)
    compare(hex_out, x86_out, atol=0.05, fail_on_mismatch=True)
    if enablelwp:
        process_lwp()
    print("\nUNet test PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD UNet Hexagon benchmark")
    add_phase4_cli(parser)
    args = parser.parse_args()
    test_unet(
        enablelwp=args.lwp,
        enable_hexkl=args.enable_hexkl,
        enable_alps_vdae=args.enable_alps_vdae,
        enable_alps_layout_aware=not args.disable_layout_aware,
        alps_lookahead=args.alps_lookahead,
        enable_alps_adaptive=not args.disable_alps_adaptive,
        enable_alps_items_1_7=args.enable_alps_items_1_7,
    )
