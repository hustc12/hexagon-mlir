#!/usr/bin/env python3
"""Local-LWP vehicle for complete-shape Speech encoders (not a Debug model).

The formal latency remains the complete monolithic wrapper.  This script keeps
the published FP16 shapes and weights, but compiles the frontend, positional
formation, one representative encoder layer, and CTC head independently so
LWP does not duplicate a 12-layer single-function IR and exhaust host memory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import (
    HubertConfig,
    HubertForCTC,
    UniSpeechConfig,
    UniSpeechForCTC,
    UniSpeechSatConfig,
    UniSpeechSatForCTC,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmark_models"))
from full_audio_encoder import (  # noqa: E402
    _patch_group_norm_f32_accumulation,
    _reformulate_pos_conv,
)
from hexkl_utils import (  # noqa: E402
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_full_model_dsp_heap,
)


MODELS = {
    "wav2vec2-base": (Wav2Vec2Config, Wav2Vec2ForCTC, "wav2vec2", 960),
    "hubert-base": (HubertConfig, HubertForCTC, "hubert", 961),
    "unispeech-base": (UniSpeechConfig, UniSpeechForCTC, "unispeech", 377),
    "unispeech-sat-base": (
        UniSpeechSatConfig,
        UniSpeechSatForCTC,
        "unispeech_sat",
        991,
    ),
}


class AudioFrontendStage(torch.nn.Module):
    def __init__(self, root: torch.nn.Module, projection_returns_tuple: bool):
        super().__init__()
        self.feature_extractor = root.feature_extractor
        self.feature_projection = root.feature_projection
        self.projection_returns_tuple = projection_returns_tuple

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(samples).transpose(1, 2)
        projected = self.feature_projection(features)
        return projected[0] if self.projection_returns_tuple else projected


class AudioPositionStage(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.pos_conv_embed = encoder.pos_conv_embed
        self.layer_norm = encoder.layer_norm

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(hidden + self.pos_conv_embed(hidden))


class AudioEncoderLayerStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layer(hidden, attention_mask=None, output_attentions=False)[0]


class AudioCTCHeadStage(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


def make_options():
    return hexagon_options_phase4(
        True,
        False,
        False,
        2,
        False,
        False,
        lower_constants_separate=True,
        backend_profile="hvx-vector",
        enable_lwp=True,
        lwp_loop_depth=1,
        disable_lwp_loop=False,
        enable_omnifetch_kv_cache_prefetch=True,
    )


def run_stage(label: str, stage: torch.nn.Module, example: torch.Tensor) -> torch.Tensor:
    stage = stage.half().eval()
    with torch.no_grad():
        reference = stage(example)
    module = compile_to_linalg(stage, (example,), decomp_pow=False)
    ir = str(module)
    candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(ir)
    patched = candidate if n_batch or n_f16 else None
    print(
        f"[AudioHotspotIR] stage={label} batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')} batch_rewrite={n_batch} f16_rewrite={n_f16}"
    )
    output = hex_execution(
        module,
        stage.__class__.__name__,
        [example],
        make_options(),
        mlir_text=patched,
        iterations=1,
    )[0]
    finite = bool(torch.isfinite(output).all())
    max_abs = float((output.float() - reference.float()).abs().max())
    scale = max(float(reference.float().abs().max()), 1.0)
    tolerance = max(0.02 * scale, 0.05)
    correct = finite and max_abs <= tolerance
    print(
        f"[AudioHotspotCompare] stage={label} finite={finite} max_abs={max_abs:.9g} "
        f"tolerance={tolerance:.9g} correct={correct}"
    )
    if not correct:
        raise AssertionError(f"full-shape audio hotspot stage failed: {label}")
    return reference.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(MODELS))
    args = parser.parse_args()
    patch_full_model_dsp_heap()
    config_cls, model_cls, root_name, seed = MODELS[args.model]
    torch.manual_seed(seed)
    config = config_cls()
    config.apply_spec_augment = False
    config.hidden_act = "gelu_new"
    config.feat_extract_activation = "gelu_new"
    config._attn_implementation = "eager"
    model = model_cls(config).half().eval()
    _patch_group_norm_f32_accumulation(model)
    root = getattr(model, root_name)
    conv = getattr(root.encoder.pos_conv_embed, "conv", None)
    if conv is not None and hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
        torch.nn.utils.parametrize.remove_parametrizations(
            conv, "weight", leave_parametrized=True
        )
    _reformulate_pos_conv(root.encoder.pos_conv_embed)

    samples = torch.rand(1, 20560, dtype=torch.float16) * 2 - 1
    print(
        f"[AudioHotspotFullShape] model={args.model} samples=20560 frames=64 "
        f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"params={sum(p.numel() for p in model.parameters())}"
    )
    hidden = run_stage(
        "frontend",
        AudioFrontendStage(root, projection_returns_tuple=root_name != "hubert"),
        samples,
    )
    positioned = run_stage("position", AudioPositionStage(root.encoder), hidden)
    layer_output = run_stage(
        "encoder_layer_0", AudioEncoderLayerStage(root.encoder.layers[0]), positioned
    )
    run_stage("ctc_head", AudioCTCHeadStage(model), layer_output)


if __name__ == "__main__":
    main()
