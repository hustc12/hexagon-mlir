#!/usr/bin/env python3
"""Full-checkpoint, layer-granular Falcon-RW-1B v73 execution probe."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from run_falcon_rw_1b import _install_alibi_patch, encode_fixed_seq


torch.set_num_threads(1)


class FalconEmbeddingStage(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.embedding = model.transformer.word_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class FalconLayerStage(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        alibi: torch.Tensor | None,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        super().__init__()
        self.layer = layer
        constants = (
            ("alibi", alibi),
            ("causal_mask", causal_mask),
            ("position_ids", position_ids),
            ("cache_position", position_ids[0]),
            ("cos", position_embeddings[0]),
            ("sin", position_embeddings[1]),
        )
        for name, value in constants:
            object.__setattr__(self, name, value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer(
            hidden_states,
            alibi=self.alibi,
            attention_mask=self.causal_mask,
            position_ids=self.position_ids,
            layer_past=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
            cache_position=self.cache_position,
            position_embeddings=(self.cos, self.sin),
        )[0]


class FalconHeadStage(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.norm = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))


class FalconLMHeadStage(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(normalized_hidden)


class FalconAttentionStage(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        alibi: torch.Tensor | None,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        super().__init__()
        self.attention = layer.self_attention
        for name, value in (
            ("alibi", alibi),
            ("causal_mask", causal_mask),
            ("position_ids", position_ids),
            ("cache_position", position_ids[0]),
            ("cos", position_embeddings[0]),
            ("sin", position_embeddings[1]),
        ):
            object.__setattr__(self, name, value)

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        return self.attention(
            normalized_hidden,
            alibi=self.alibi,
            attention_mask=self.causal_mask,
            position_ids=self.position_ids,
            layer_past=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
            cache_position=self.cache_position,
            position_embeddings=(self.cos, self.sin),
        )[0]


class FalconNormStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.norm = layer.input_layernorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class FalconStableNormStage(torch.nn.Module):
    """Algebraically scaled LayerNorm that avoids f16 square overflow."""

    def __init__(self, layer: torch.nn.Module, input_scale: float = 1024.0):
        super().__init__()
        self.norm = layer.input_layernorm
        self.input_scale = input_scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return functional.layer_norm(
            hidden_states / self.input_scale,
            self.norm.normalized_shape,
            self.norm.weight,
            self.norm.bias,
            self.norm.eps / (self.input_scale * self.input_scale),
        )


class FalconMLPStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.mlp = layer.mlp

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(normalized_hidden)


class FalconCombineStage(torch.nn.Module):
    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        return stacked[0] + stacked[1] + stacked[2]


def export_stage(stage: torch.nn.Module, example: torch.Tensor, path: Path) -> None:
    module = fx.export_and_import(
        stage,
        example,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=stage.__class__.__name__,
    )
    path.write_bytes(module.operation.get_asm(binary=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/falcon-layered"))
    parser.add_argument("--device-iterations", type=int, default=1)
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--resume-output-raw", type=Path)
    parser.add_argument("--diagnose-layer", type=int)
    parser.add_argument("--split-layers", action="store_true")
    parser.add_argument(
        "--effective-layers",
        type=int,
        help="explicit cropped depth; preserves widths, vocabulary and prefix weights",
    )
    parser.add_argument("--split-head", action="store_true")
    args = parser.parse_args()
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")

    model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = encode_fixed_seq(
        tokenizer, "What is nature of our existence?", args.seq_len
    )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    config.use_cache = False
    config.activation = "gelu_fast"
    _install_alibi_patch()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager",
    ).eval()
    input_ids = input_ids.clamp(0, config.vocab_size - 1)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    embedding = FalconEmbeddingStage(model).eval()
    initial_hidden = embedding(input_ids)

    import transformers.models.falcon.modeling_falcon as falcon_modeling

    alibi = None
    if model.transformer.use_alibi:
        alibi = falcon_modeling.build_alibi_tensor(
            attention_mask, config.num_attention_heads, initial_hidden.dtype
        )
    cache_position = position_ids[0]
    causal_mask = model.transformer._update_causal_mask(
        attention_mask,
        initial_hidden,
        cache_position,
        None,
        False,
        None,
        alibi,
    )
    if causal_mask.is_floating_point():
        causal_mask = torch.where(
            causal_mask < -1.0e3,
            torch.tensor(-1.0e4, dtype=causal_mask.dtype),
            causal_mask,
        )
    position_embeddings = model.transformer.rotary_emb(initial_hidden, position_ids)
    layers = [
        FalconLayerStage(
            layer, alibi, causal_mask, position_ids, position_embeddings
        ).eval()
        for layer in model.transformer.h
    ]
    full_depth = len(layers)
    effective_depth = args.effective_layers or full_depth
    if not 1 <= effective_depth <= full_depth:
        raise ValueError("--effective-layers is outside the checkpoint depth")
    layers = layers[:effective_depth]
    head = FalconHeadStage(model).eval()

    with torch.no_grad():
        staged = initial_hidden
        for layer in layers:
            staged = layer(staged)
        staged = head(staged)
        if effective_depth == full_depth:
            reference = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits
        else:
            reference = staged.detach().clone()
    cpu_max_abs = float((staged.float() - reference.float()).abs().max())
    cpu_top5 = torch.topk(reference[0, -1].float(), 5).indices
    staged_top5 = torch.topk(staged[0, -1].float(), 5).indices
    print(
        f"[FalconLayeredCPU] checkpoint={model_name} layers={len(layers)}/{full_depth} "
        f"cropped={effective_depth != full_depth} "
        f"hidden={config.hidden_size} heads={config.num_attention_heads} "
        f"vocab={config.vocab_size} seq_len={args.seq_len} "
        f"max_abs={cpu_max_abs:.9g} top5_match={bool(torch.equal(cpu_top5, staged_top5))}"
    )
    if cpu_max_abs > 0.2 or not torch.equal(cpu_top5, staged_top5):
        raise AssertionError("staged Falcon differs from the formal CPU model")
    if args.cpu_only:
        return

    from hexkl_utils import patch_full_model_dsp_heap
    from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
    from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import (
        TorchMLIRHexagonLauncher,
    )

    patch_full_model_dsp_heap()
    options = HexagonOptions().__dict__.copy()
    options["enableVectorization"] = True
    options["enableHexKL"] = bool(args.enable_hexkl)
    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = bool(args.enable_hexkl)
    options["enableConversionToFp16"] = False
    options["lowerConstantsInSeparateSharedObjects"] = True
    if "enableBufferResultsToOutParams" in options:
        options["enableBufferResultsToOutParams"] = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    launcher = TorchMLIRHexagonLauncher()

    def run_stage(
        stage: torch.nn.Module,
        stage_input: torch.Tensor,
        label: str,
        *,
        enforce: bool = True,
    ) -> tuple[torch.Tensor, bool]:
        path = args.output_dir / f"falcon_{label}.mlirbc"
        export_stage(stage, stage_input, path)
        with torch.no_grad():
            expected = stage(stage_input)
        print(f"[FalconDeviceStage] begin={label}")
        actual = launcher.run_torch_mlir(
            str(path),
            [stage_input],
            stage.__class__.__name__,
            iterations=args.device_iterations,
            options=options,
        )[0]
        finite = bool(torch.isfinite(actual).all())
        max_abs = float((actual.float() - expected.float()).abs().max())
        scale = float(expected.float().abs().max())
        relative = max_abs / max(scale, 1.0e-12)
        tolerance = max(0.2, scale * 0.02)
        correct = finite and max_abs <= tolerance
        print(
            f"[FalconDeviceStage] end={label} finite={finite} "
            f"expected_abs_max={scale:.9g} max_abs={max_abs:.9g} "
            f"relative={relative:.9g} tolerance={tolerance:.9g} correct={correct}"
        )
        if enforce and not correct:
            raise AssertionError(f"Falcon device stage {label} failed correctness")
        return actual.detach(), correct

    def load_resume_hidden() -> torch.Tensor:
        if args.resume_output_raw is None:
            raise ValueError("this mode requires --resume-output-raw")
        payload = args.resume_output_raw.read_bytes()
        expected_bytes = args.seq_len * config.hidden_size * 2
        if len(payload) != expected_bytes + 36:
            raise ValueError(
                f"resume raw has {len(payload)} bytes; expected {expected_bytes + 36}"
            )
        return torch.frombuffer(
            bytearray(payload[36:]), dtype=torch.float16
        ).clone().reshape(1, args.seq_len, config.hidden_size)

    if args.diagnose_layer is not None:
        hidden = load_resume_hidden()
        raw_layer = model.transformer.h[args.diagnose_layer]
        with torch.no_grad():
            normalized = raw_layer.input_layernorm(hidden)
            attention_stage = FalconAttentionStage(
                raw_layer, alibi, causal_mask, position_ids, position_embeddings
            ).eval()
            mlp_stage = FalconMLPStage(raw_layer).eval()
            attention_output = attention_stage(normalized)
            mlp_output = mlp_stage(normalized)
            stacked = torch.stack((hidden, attention_output, mlp_output))
        probes = (
            (attention_stage, normalized, "diag_attention"),
            (mlp_stage, normalized, "diag_mlp"),
            (FalconCombineStage().eval(), stacked, "diag_residual_combine"),
            (layers[args.diagnose_layer], hidden, "diag_full_layer"),
        )
        failures = []
        for stage, stage_input, label in probes:
            _, correct = run_stage(stage, stage_input, label, enforce=False)
            if not correct:
                failures.append(label)
        print(f"[FalconLayerDiagnosis] layer={args.diagnose_layer} failures={failures}")
        if failures:
            raise AssertionError(f"Falcon layer diagnosis failed: {failures}")
        return

    if args.resume_output_raw is not None:
        hidden = load_resume_hidden()
        print(
            f"[FalconResume] start_layer={args.start_layer} "
            f"input={args.resume_output_raw} finite={bool(torch.isfinite(hidden).all())}"
        )
    else:
        if args.start_layer != 0:
            raise ValueError("--start-layer requires --resume-output-raw")
        hidden, _ = run_stage(embedding, input_ids, "embedding")
    for index, layer in enumerate(layers[args.start_layer :], args.start_layer):
        if args.split_layers:
            raw_layer = model.transformer.h[index]
            normalized, _ = run_stage(
                FalconStableNormStage(raw_layer).eval(),
                hidden,
                f"layer_{index:02d}_norm",
            )
            attention, _ = run_stage(
                FalconAttentionStage(
                    raw_layer, alibi, causal_mask, position_ids, position_embeddings
                ).eval(),
                normalized,
                f"layer_{index:02d}_attention",
            )
            mlp, _ = run_stage(
                FalconMLPStage(raw_layer).eval(),
                normalized,
                f"layer_{index:02d}_mlp",
            )
            hidden, _ = run_stage(
                FalconCombineStage().eval(),
                torch.stack((hidden, attention, mlp)),
                f"layer_{index:02d}_combine",
            )
        else:
            hidden, _ = run_stage(layer, hidden, f"layer_{index:02d}")
        if args.stop_after_layer == index:
            print(f"[FalconStoppedAfterLayer] layer={index}")
            return
    if args.split_head:
        with torch.no_grad():
            normalized_hidden = model.transformer.ln_f(hidden)
        actual, _ = run_stage(
            FalconLMHeadStage(model).eval(), normalized_hidden, "lm_head_projection"
        )
        print("[FalconHostFallback] operation=final_layer_norm")
    else:
        actual, _ = run_stage(head, hidden, "head")
    final_max_abs = float((actual.float() - reference.float()).abs().max())
    actual_top5 = torch.topk(actual[0, -1].float(), 5).indices
    top5_match = bool(torch.equal(actual_top5, cpu_top5))
    print(
        f"[FalconDeviceFullCompare] layers={len(layers)} finite="
        f"{bool(torch.isfinite(actual).all())} max_abs={final_max_abs:.9g} "
        f"top5_match={top5_match}"
    )
    if not top5_match:
        raise AssertionError("full staged Falcon failed top-5 correctness gate")


if __name__ == "__main__":
    main()
