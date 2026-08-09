#!/usr/bin/env python3
"""Full-checkpoint, layer-granular Qwen2.5-0.5B v73 execution probe."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


torch.set_num_threads(1)


def stable_silu(value: torch.Tensor) -> torch.Tensor:
    """Overflow-safe SiLU in the input dtype.

    exp(-abs(x)) is always in [0, 1], unlike the common x/(1+exp(-x))
    formulation whose f16 intermediate overflows for sufficiently negative x.
    """
    exponential = torch.exp(-torch.abs(value))
    denominator = 1.0 + exponential
    return torch.where(
        value >= 0,
        value / denominator,
        value * exponential / denominator,
    )


class QwenEmbeddingStage(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class QwenSafeLayerStage(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        super().__init__()
        self.layer = layer
        for name, value in (
            ("causal_mask", causal_mask),
            ("position_ids", position_ids),
            ("cache_position", position_ids[0]),
            ("cos", cos),
            ("sin", sin),
        ):
            object.__setattr__(self, name, value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer(
            hidden_states,
            attention_mask=self.causal_mask,
            position_ids=self.position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=self.cache_position,
            position_embeddings=(self.cos, self.sin),
        )[0]


class QwenHeadStage(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))


class QwenInputNormStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.norm = layer.input_layernorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class QwenAttentionCoreStage(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        super().__init__()
        self.attention = layer.self_attn
        for name, value in (("causal_mask", causal_mask), ("cos", cos), ("sin", sin)):
            object.__setattr__(self, name, value)

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        return self.attention(
            normalized_hidden,
            position_embeddings=(self.cos, self.sin),
            attention_mask=self.causal_mask,
            past_key_value=None,
        )[0]


class QwenPostNormStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.norm = layer.post_attention_layernorm

    def forward(self, attention_residual: torch.Tensor) -> torch.Tensor:
        return self.norm(attention_residual)


class QwenGateProjectionStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.projection = layer.mlp.gate_proj

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(normalized_hidden)


class QwenUpProjectionStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.projection = layer.mlp.up_proj

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(normalized_hidden)


class QwenActivatedProductStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.gate_projection = layer.mlp.gate_proj
        self.up_projection = layer.mlp.up_proj
        self.activation = layer.mlp.act_fn

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        return self.activation(self.gate_projection(normalized_hidden)) * self.up_projection(normalized_hidden)


class QwenDownProjectionStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.projection = layer.mlp.down_proj

    def forward(self, activated_product: torch.Tensor) -> torch.Tensor:
        return self.projection(activated_product)


class QwenMLPStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.mlp = layer.mlp

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(normalized_hidden)


class QwenActivationStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.activation = layer.mlp.act_fn

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        return self.activation(gate)


class QwenProductStage(torch.nn.Module):
    def forward(self, stacked_inputs: torch.Tensor) -> torch.Tensor:
        return stacked_inputs[0] * stacked_inputs[1]


class QwenStableActivatedProductStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.gate_projection = layer.mlp.gate_proj
        self.up_projection = layer.mlp.up_proj

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        return stable_silu(self.gate_projection(normalized_hidden)) * self.up_projection(normalized_hidden)


class QwenStableMLPStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.gate_projection = layer.mlp.gate_proj
        self.up_projection = layer.mlp.up_proj
        self.down_projection = layer.mlp.down_proj

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        product = stable_silu(self.gate_projection(normalized_hidden)) * self.up_projection(normalized_hidden)
        return self.down_projection(product)


class QwenStableLayerStage(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        super().__init__()
        self.input_norm = layer.input_layernorm
        self.attention = layer.self_attn
        self.post_attention_norm = layer.post_attention_layernorm
        self.gate_projection = layer.mlp.gate_proj
        self.up_projection = layer.mlp.up_proj
        self.down_projection = layer.mlp.down_proj
        for name, value in (("causal_mask", causal_mask), ("cos", cos), ("sin", sin)):
            object.__setattr__(self, name, value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention = self.attention(
            self.input_norm(hidden_states),
            position_embeddings=(self.cos, self.sin),
            attention_mask=self.causal_mask,
            past_key_value=None,
        )[0]
        attention_residual = hidden_states + attention
        normalized = self.post_attention_norm(attention_residual)
        product = stable_silu(self.gate_projection(normalized)) * self.up_projection(normalized)
        return attention_residual + self.down_projection(product)


def fixed_input(tokenizer, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = tokenizer.encode("Hi", add_special_tokens=False)
    filler = tokenizer.encode(" true", add_special_tokens=False) or [
        tokenizer.eos_token_id
    ]
    while len(ids) < seq_len:
        ids.extend(filler)
    input_ids = torch.tensor([ids[:seq_len]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    return input_ids, attention_mask, position_ids


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
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/qwen-layered"))
    parser.add_argument("--device-iterations", type=int, default=1)
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument(
        "--resume-output-raw",
        type=Path,
        help="launcher output raw (36-byte memref header followed by tensor data)",
    )
    parser.add_argument(
        "--diagnose-layer",
        type=int,
        help="run component probes for one layer using a CPU-reconstructed input",
    )
    parser.add_argument(
        "--diagnose-repair",
        action="store_true",
        help="with --diagnose-layer, only test the SiLU repair boundaries",
    )
    args = parser.parse_args()
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.start_layer < 0:
        raise ValueError("--start-layer must be non-negative")

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    input_ids, attention_mask, position_ids = fixed_input(tokenizer, args.seq_len)
    embedding = QwenEmbeddingStage(model).eval()
    initial_hidden = embedding(input_ids)
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(initial_hidden, position_ids)
    causal_mask = torch.zeros(
        (1, 1, args.seq_len, args.seq_len), dtype=initial_hidden.dtype
    )
    causal_mask.masked_fill_(
        torch.triu(
            torch.ones(args.seq_len, args.seq_len, dtype=torch.bool), diagonal=1
        ),
        -1.0e4,
    )
    original_layers = [
        QwenSafeLayerStage(layer, causal_mask, position_ids, cos, sin).eval()
        for layer in model.model.layers
    ]
    layers = [
        QwenStableLayerStage(layer, causal_mask, cos, sin).eval()
        for layer in model.model.layers
    ]
    head = QwenHeadStage(model).eval()

    with torch.no_grad():
        reference = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits
        safe_reference = initial_hidden
        for layer in layers:
            safe_reference = layer(safe_reference)
        safe_reference = head(safe_reference)
    cpu_max_abs = float((safe_reference.float() - reference.float()).abs().max())
    cpu_top5 = torch.topk(reference[0, -1].float(), 5).indices
    safe_top5 = torch.topk(safe_reference[0, -1].float(), 5).indices
    print(
        f"[QwenLayeredCPU] checkpoint={model_name} layers={len(layers)} "
        f"hidden={config.hidden_size} intermediate={config.intermediate_size} "
        f"heads={config.num_attention_heads}/{config.num_key_value_heads} "
        f"vocab={config.vocab_size} seq_len={args.seq_len} "
        f"max_abs={cpu_max_abs:.9g} top5_match={bool(torch.equal(cpu_top5, safe_top5))}"
    )
    if cpu_max_abs > 0.2 or not torch.equal(cpu_top5, safe_top5):
        raise AssertionError("safe staged Qwen differs from the formal CPU model")
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
        path = args.output_dir / f"qwen_{label}.mlirbc"
        export_stage(stage, stage_input, path)
        with torch.no_grad():
            expected = stage(stage_input)
        print(f"[QwenDeviceStage] begin={label}")
        actual = launcher.run_torch_mlir(
            str(path), [stage_input], stage.__class__.__name__,
            iterations=args.device_iterations, options=options,
        )[0]
        finite = bool(torch.isfinite(actual).all())
        max_abs = float((actual.float() - expected.float()).abs().max())
        expected_scale = float(expected.float().abs().max())
        relative = max_abs / max(expected_scale, 1.0e-12)
        tolerance = max(0.2, expected_scale * 0.02)
        correct = finite and max_abs <= tolerance
        print(
            f"[QwenDeviceStage] end={label} finite={finite} "
            f"expected_abs_max={expected_scale:.9g} max_abs={max_abs:.9g} "
            f"relative={relative:.9g} tolerance={tolerance:.9g} correct={correct}"
        )
        if enforce and not correct:
            raise AssertionError(f"Qwen device stage {label} failed correctness")
        return actual.detach(), correct

    if args.diagnose_layer is not None:
        if not 0 <= args.diagnose_layer < len(layers):
            raise ValueError("--diagnose-layer is outside the model depth")
        # Reconstruct the exact failing layer input entirely on the host.  This
        # avoids rerunning already validated device prefixes during diagnosis.
        with torch.no_grad():
            layer_input = initial_hidden
            for prefix_layer in layers[: args.diagnose_layer]:
                layer_input = prefix_layer(layer_input)
            raw_layer = model.model.layers[args.diagnose_layer]
            normalized = raw_layer.input_layernorm(layer_input)
            attention = raw_layer.self_attn(
                normalized,
                position_embeddings=(cos, sin),
                attention_mask=causal_mask,
                past_key_value=None,
            )[0]
            attention_residual = layer_input + attention
            post_normalized = raw_layer.post_attention_layernorm(attention_residual)
            activated_product = raw_layer.mlp.act_fn(
                raw_layer.mlp.gate_proj(post_normalized)
            ) * raw_layer.mlp.up_proj(post_normalized)
            stable_product = stable_silu(
                raw_layer.mlp.gate_proj(post_normalized)
            ) * raw_layer.mlp.up_proj(post_normalized)
            stable_product_error = float(
                (stable_product.float() - activated_product.float()).abs().max()
            )
        print(
            f"[QwenStableSiLUCPU] layer={args.diagnose_layer} "
            f"product_max_abs={stable_product_error:.9g}"
        )
        if stable_product_error > 0.2:
            raise AssertionError("stable SiLU is not equivalent in checkpoint dtype")
        base_probes = (
            (QwenInputNormStage(raw_layer).eval(), layer_input, "diag_input_norm"),
            (QwenAttentionCoreStage(raw_layer, causal_mask, cos, sin).eval(), normalized, "diag_attention_core"),
            (QwenPostNormStage(raw_layer).eval(), attention_residual, "diag_post_norm"),
            (QwenGateProjectionStage(raw_layer).eval(), post_normalized, "diag_gate_projection"),
            (QwenUpProjectionStage(raw_layer).eval(), post_normalized, "diag_up_projection"),
            (QwenActivatedProductStage(raw_layer).eval(), post_normalized, "diag_activated_product"),
            (QwenDownProjectionStage(raw_layer).eval(), activated_product, "diag_down_projection"),
            (QwenMLPStage(raw_layer).eval(), post_normalized, "diag_full_mlp"),
            (original_layers[args.diagnose_layer], layer_input, "diag_full_layer"),
        )
        repair_probes = (
            (QwenActivationStage(raw_layer).eval(), raw_layer.mlp.gate_proj(post_normalized), "repair_original_activation"),
            (
                QwenProductStage().eval(),
                torch.stack(
                    (
                        raw_layer.mlp.act_fn(raw_layer.mlp.gate_proj(post_normalized)),
                        raw_layer.mlp.up_proj(post_normalized),
                    )
                ),
                "repair_product_only",
            ),
            (QwenStableActivatedProductStage(raw_layer).eval(), post_normalized, "repair_stable_activated_product"),
            (QwenStableMLPStage(raw_layer).eval(), post_normalized, "repair_stable_full_mlp"),
            (
                QwenStableLayerStage(raw_layer, causal_mask, cos, sin).eval(),
                layer_input,
                "repair_stable_full_layer",
            ),
        )
        probes = repair_probes if args.diagnose_repair else base_probes
        failures = []
        for stage, stage_input, label in probes:
            _, correct = run_stage(stage, stage_input, label, enforce=False)
            if not correct:
                failures.append(label)
        print(f"[QwenLayerDiagnosis] layer={args.diagnose_layer} failures={failures}")
        if failures:
            raise AssertionError(f"Qwen layer diagnosis failed: {failures}")
        return

    if args.resume_output_raw is not None:
        payload = args.resume_output_raw.read_bytes()
        expected_bytes = args.seq_len * config.hidden_size * 2
        if len(payload) != expected_bytes + 36:
            raise ValueError(
                f"resume raw has {len(payload)} bytes; expected {expected_bytes + 36}"
            )
        hidden = torch.frombuffer(
            bytearray(payload[36:]), dtype=torch.float16
        ).clone().reshape(1, args.seq_len, config.hidden_size)
        print(
            f"[QwenResume] start_layer={args.start_layer} "
            f"input={args.resume_output_raw} finite={bool(torch.isfinite(hidden).all())}"
        )
    else:
        if args.start_layer != 0:
            raise ValueError("--start-layer requires --resume-output-raw")
        hidden, _ = run_stage(embedding, input_ids, "embedding")
    for index, layer in enumerate(layers[args.start_layer :], args.start_layer):
        hidden, _ = run_stage(layer, hidden, f"layer_{index:02d}")
    actual, _ = run_stage(head, hidden, "head")
    final_max_abs = float((actual.float() - reference.float()).abs().max())
    actual_top5 = torch.topk(actual[0, -1].float(), 5).indices
    top5_match = bool(torch.equal(actual_top5, cpu_top5))
    print(
        f"[QwenDeviceFullCompare] layers={len(layers)} finite="
        f"{bool(torch.isfinite(actual).all())} max_abs={final_max_abs:.9g} "
        f"top5_match={top5_match}"
    )
    if not top5_match:
        raise AssertionError("full staged Qwen failed top-5 correctness gate")


if __name__ == "__main__":
    main()
