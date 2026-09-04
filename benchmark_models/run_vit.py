"""ViT-Base full-structure Hexagon/HexKL/Alps benchmark."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageClassification, AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_full_model_dsp_heap,
)


class ViTWrapper(torch.nn.Module):
    def __init__(self, vit_model: torch.nn.Module):
        super().__init__()
        self.vit = vit_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).logits


def customize_model_config(config):
    """Identity hook except gelu_new (Hexagon lacks math.erf)."""
    config.hidden_act = "gelu_new"
    return config


def run(args: argparse.Namespace) -> None:
    patch_full_model_dsp_heap()
    torch.manual_seed(871)
    model_name = "google/vit-base-patch16-224"

    config = AutoConfig.from_pretrained(model_name)
    config = customize_model_config(config)
    print(
        f"[Config] layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"patch={config.patch_size} image={config.image_size} "
        f"heads={config.num_attention_heads}"
    )

    model = AutoModelForImageClassification.from_config(config).half().eval()
    # Externalize the learned position embedding as a non-persistent buffer so
    # torch-mlir lifts it to the leading public function argument (matching the
    # DINOv2/DeiT runners).  When the position embedding is left as an inlined
    # nn.Parameter constant the on-device run faults with a NULL/TLBMISS in the
    # async worker threads; supplying it as a runtime memref arg avoids that.
    embeddings = model.vit.embeddings
    fixed_position_embeddings = embeddings.position_embeddings.detach().clone()
    del embeddings.position_embeddings
    embeddings.register_buffer(
        "position_embeddings", fixed_position_embeddings, persistent=False
    )
    wrapped = ViTWrapper(model).eval()
    func_name = wrapped.__class__.__name__

    img = args.image_size or config.image_size
    pixel_values = torch.rand(1, 3, img, img, dtype=torch.float16)
    inputs = [pixel_values]
    # torch-mlir lifts the buffer to the first argument; the Python model still
    # exposes only pixel_values to callers, so the device call must supply the
    # lifted position-embedding buffer explicitly before the pixel input.
    device_inputs = [fixed_position_embeddings, pixel_values]
    params = sum(parameter.numel() for parameter in model.parameters())
    model_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        model_hash.update(name.encode("utf-8"))
        model_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    input_hash = hashlib.sha256(
        pixel_values.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    print(
        f"[Input] pixel_values={tuple(pixel_values.shape)} params={params} "
        f"HexKL={args.enable_hexkl}"
    )
    print(
        f"[ViT Identity] model_sha256={model_hash.hexdigest()} "
        f"input_sha256={input_hash}"
    )

    module = compile_to_linalg(wrapped, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )

    patched = None
    if args.enable_hexkl:
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(
            ir, enable_m_pad=args.enable_alps_m_pad_hmx
        )
        patched = candidate if n_batch or n_f16 else None
        print(f"[HexKL] batch_matmul→matmul={n_batch}, f16-input rewrite={n_f16}")

    options = hexagon_options_phase4(
        args.enable_hexkl,
        args.enable_alps_vdae,
        not args.disable_layout_aware,
        args.alps_lookahead,
        not args.disable_alps_adaptive,
        args.enable_alps_items_1_7,
        lower_constants_separate=True,
        backend_profile=args.backend_profile,
        enable_alps_m_pad_hmx=args.enable_alps_m_pad_hmx,
        enable_out_params=args.enable_out_params,
        prefetch_baseline=args.prefetch_baseline,
        prefetch_baseline_distance=args.prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=args.apt_get_hx_manual_candidate_ids,
        enable_alps_kv_cache_prefetch=(
            args.enable_alps_kv_cache_prefetch
        ),
        disable_alps_persistent_wh_cache=(
            args.disable_alps_persistent_wh_cache
        ),
        enable_alps_fp16_hvx_arithmetic=(
            args.enable_alps_fp16_hvx_arithmetic
        ),
    )
    output = hex_execution(
        module,
        func_name,
        device_inputs,
        options,
        mlir_text=patched,
        iterations=args.device_iterations,
    )
    print("Successfully ran ViT on Hexagon DSP!")

    with torch.no_grad():
        reference = wrapped(*inputs)
    finite = bool(torch.isfinite(output[0]).all())
    diff = (output[0].float() - reference.float()).abs().max().item()
    top1_match = output[0].argmax().item() == reference.argmax().item()
    print(
        f"[Compare] finite={finite} max_abs_diff={diff:.4f} "
        f"top1_match={top1_match}"
    )
    if not finite or not top1_match:
        raise AssertionError("ViT-Base Hexagon result failed correctness gate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=None)
    run(parser.parse_args())
