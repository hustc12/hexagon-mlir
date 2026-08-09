#!/usr/bin/env python3
"""Validate and export a device-resident staged GPT-2 compilation boundary.

This is an engineering probe, not a benchmark.  It keeps the published GPT-2
weights and all 12 transformer blocks, checks that a staged CPU execution
matches the original model, and exports each stage independently so host
codegen never sees the fully unrolled compute graph.  Artifacts default to
/tmp and must not be committed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# Staged and monolithic CPU references may otherwise choose different
# parallel reduction schedules.  Keep the correctness oracle deterministic.
torch.set_num_threads(1)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark_models"))
from run_gpt2lmheadmodel import freeze_gpt2_attn_bias_buffers  # noqa: E402


class GPT2EmbeddingStage(torch.nn.Module):
    def __init__(self, model: GPT2LMHeadModel, seq_len: int):
        super().__init__()
        self.wte = model.transformer.wte
        self.wpe = model.transformer.wpe
        self.drop = model.transformer.drop
        # Fixed-length probe: keep positions as an exported constant instead
        # of an extra ciface argument unsupported by the launcher wrapper.
        object.__setattr__(
            self,
            "position_ids",
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.drop(self.wte(input_ids) + self.wpe(self.position_ids))


class GPT2BlockStage(torch.nn.Module):
    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
        )[0]


class GPT2HeadStage(torch.nn.Module):
    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.ln_f(hidden_states))


class GPT2IdentityStage(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 0.0


class GPT2LayerNormStage(torch.nn.Module):
    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.layer_norm = block.ln_1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(hidden_states)


class GPT2AttentionStage(torch.nn.Module):
    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.layer_norm = block.ln_1
        self.attention = block.attn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.layer_norm(hidden_states)
        return self.attention(
            normalized,
            past_key_value=None,
            cache_position=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
        )[0]


class GPT2AttentionPrefixStage(torch.nn.Module):
    """Expose exact eager-attention prefix boundaries for lowering diagnosis."""

    def __init__(self, block: torch.nn.Module, boundary: str):
        super().__init__()
        self.layer_norm = block.ln_1
        self.c_attn = block.attn.c_attn
        self.c_proj = block.attn.c_proj
        # Keep this a plain tensor attribute. Registering it as a buffer adds a
        # second ciface argument that the current device launcher cannot supply.
        object.__setattr__(self, "causal_mask", block.attn.bias)
        self.safe_mask = boundary.endswith("_safe_mask")
        self.boundary = boundary.removesuffix("_safe_mask")
        self.embed_dim = block.attn.embed_dim
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.layer_norm(hidden_states)
        qkv = self.c_attn(normalized)
        if self.boundary == "attention_qkv":
            return qkv

        query, key, value = qkv.split(self.embed_dim, dim=2)
        query = query.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / torch.full(
            [], self.head_dim**0.5, dtype=scores.dtype, device=scores.device
        )
        if self.boundary == "attention_scores":
            return scores

        seq_len = scores.shape[-1]
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        mask_value_scalar = -1.0e4 if self.safe_mask else torch.finfo(scores.dtype).min
        mask_value = torch.full(
            [], mask_value_scalar, dtype=scores.dtype, device=scores.device
        )
        masked_scores = torch.where(causal_mask, scores, mask_value)
        if self.boundary == "attention_masked_scores":
            return masked_scores

        probabilities = torch.nn.functional.softmax(masked_scores, dim=-1)
        if self.boundary == "attention_softmax":
            return probabilities

        context = torch.matmul(probabilities, value).transpose(1, 2)
        context = context.reshape(1, -1, self.embed_dim).contiguous()
        if self.boundary == "attention_context":
            return context
        return self.c_proj(context)


class GPT2MLPStage(torch.nn.Module):
    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.layer_norm = block.ln_2
        self.mlp = block.mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.layer_norm(hidden_states))


class GPT2SafeBlockStage(torch.nn.Module):
    """GPT-2 block with an HVX-safe, FP32-equivalent causal sentinel."""

    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.attention = GPT2AttentionPrefixStage(
            block, "attention_projection_safe_mask"
        )
        self.layer_norm_2 = block.ln_2
        self.mlp = block.mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(hidden_states)
        return hidden_states + self.mlp(self.layer_norm_2(hidden_states))


def fixed_input(tokenizer: GPT2Tokenizer, seq_len: int) -> torch.Tensor:
    ids = tokenizer.encode(
        "What is nature of our existence? answer carefully", add_special_tokens=False
    )
    filler = tokenizer.encode(" true", add_special_tokens=False) or [
        tokenizer.eos_token_id
    ]
    while len(ids) < seq_len:
        ids.extend(filler)
    return torch.tensor([ids[:seq_len]], dtype=torch.long)


def export_stage(stage: torch.nn.Module, example: torch.Tensor, path: Path) -> int:
    module = fx.export_and_import(
        stage,
        example,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=stage.__class__.__name__,
    )
    data = module.operation.get_asm(binary=True)
    path.write_bytes(data)
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/gpt2-layered-export")
    )
    parser.add_argument(
        "--device-stage",
        choices=(
            "identity",
            "layer_norm",
            "attention_qkv",
            "attention_scores",
            "attention_masked_scores",
            "attention_softmax",
            "attention_softmax_safe_mask",
            "attention_context",
            "attention_projection",
            "attention_projection_safe_mask",
            "attention",
            "mlp",
            "safe_block_00",
            "safe_full_model",
            "embedding",
            "block_00",
            "head",
        ),
        help="Compile and execute one exported stage on the v73 device.",
    )
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--device-iterations", type=int, default=1)
    parser.add_argument(
        "--skip-full-export",
        action="store_true",
        help="Validate the full staged CPU graph but export only the selected probe.",
    )
    args = parser.parse_args()
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")

    model_name = "openai-community/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    config = GPT2Config.from_pretrained(model_name)
    config.use_cache = False
    model = GPT2LMHeadModel.from_pretrained(
        model_name, config=config, torch_dtype=torch.float32
    ).eval()
    freeze_gpt2_attn_bias_buffers(model, seq_len=args.seq_len)

    input_ids = fixed_input(tokenizer, args.seq_len)
    embedding = GPT2EmbeddingStage(model, args.seq_len).eval()
    blocks = [GPT2BlockStage(block).eval() for block in model.transformer.h]
    head = GPT2HeadStage(model).eval()

    with torch.no_grad():
        reference = model(input_ids=input_ids, use_cache=False).logits
        hidden = embedding(input_ids)
        for block in blocks:
            hidden = block(hidden)
        staged = head(hidden)
    max_abs = float((reference - staged).abs().max())
    exact_top1 = bool(
        reference[0, -1].argmax().item() == staged[0, -1].argmax().item()
    )
    print(
        f"[LayeredCPU] layers={len(blocks)} seq_len={args.seq_len} "
        f"max_abs={max_abs:.9g} last_token_top1_match={exact_top1}"
    )
    if max_abs > 1.0e-5 or not exact_top1:
        raise AssertionError("staged GPT-2 does not match the original model")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_full_export:
        sizes = []
        hidden = embedding(input_ids).detach()
        sizes.append(
            (
                "embedding",
                export_stage(
                    embedding, input_ids, args.output_dir / "gpt2_embedding.mlirbc"
                ),
            )
        )
        for index, block in enumerate(blocks):
            sizes.append(
                (
                    f"block_{index:02d}",
                    export_stage(
                        block,
                        hidden,
                        args.output_dir / f"gpt2_block_{index:02d}.mlirbc",
                    ),
                )
            )
            with torch.no_grad():
                hidden = block(hidden).detach()
        sizes.append(
            (
                "head",
                export_stage(head, hidden, args.output_dir / "gpt2_head.mlirbc"),
            )
        )
        for name, size in sizes:
            print(f"[LayeredExport] stage={name} bytecode_bytes={size}")
        print(
            f"[LayeredExport] stages={len(sizes)} total_bytecode_bytes="
            f"{sum(size for _, size in sizes)} output_dir={args.output_dir}"
        )

    if args.device_stage:
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
        options["enableConvertToHexagonmem"] = True
        # The project deliberately excludes mixed precision. HexKL must not
        # silently change the model dtype in this probe.
        options["enableConversionToFp16"] = False
        options["lowerConstantsInSeparateSharedObjects"] = True
        if "enableBufferResultsToOutParams" in options:
            options["enableBufferResultsToOutParams"] = True

        block_input = embedding(input_ids).detach()
        if args.device_stage == "safe_full_model":
            launcher = TorchMLIRHexagonLauncher()

            def run_stage_on_device(
                stage: torch.nn.Module, stage_input: torch.Tensor, label: str
            ) -> torch.Tensor:
                path = args.output_dir / f"gpt2_safe_{label}.mlirbc"
                export_stage(stage, stage_input, path)
                with torch.no_grad():
                    stage_expected = stage(stage_input)
                print(f"[LayeredDeviceStage] begin={label}")
                stage_actual = launcher.run_torch_mlir(
                    str(path),
                    [stage_input],
                    stage.__class__.__name__,
                    iterations=args.device_iterations,
                    options=options,
                )[0]
                stage_max_abs = float(
                    (stage_actual.float() - stage_expected.float()).abs().max()
                )
                print(
                    f"[LayeredDeviceStage] end={label} finite="
                    f"{bool(torch.isfinite(stage_actual).all())} "
                    f"max_abs={stage_max_abs:.9g}"
                )
                if not torch.isfinite(stage_actual).all() or stage_max_abs > 0.03:
                    raise AssertionError(f"device stage {label} failed correctness")
                return stage_actual.detach()

            device_hidden = run_stage_on_device(embedding, input_ids, "embedding")
            for index, block in enumerate(model.transformer.h):
                safe_block = GPT2SafeBlockStage(block).eval()
                device_hidden = run_stage_on_device(
                    safe_block, device_hidden, f"block_{index:02d}"
                )
            device_logits = run_stage_on_device(head, device_hidden, "head")
            final_max_abs = float((device_logits.float() - reference.float()).abs().max())
            final_top1 = bool(
                device_logits[0, -1].argmax().item()
                == reference[0, -1].argmax().item()
            )
            print(
                f"[LayeredDeviceFullCompare] layers=12 finite="
                f"{bool(torch.isfinite(device_logits).all())} "
                f"max_abs={final_max_abs:.9g} last_token_top1_match={final_top1}"
            )
            if not final_top1 or final_max_abs > 0.2:
                raise AssertionError("full staged GPT-2 failed correctness gate")
            return

        auxiliary_stages = {
            "identity": GPT2IdentityStage().eval(),
            "layer_norm": GPT2LayerNormStage(model.transformer.h[0]).eval(),
            "attention": GPT2AttentionStage(model.transformer.h[0]).eval(),
            "mlp": GPT2MLPStage(model.transformer.h[0]).eval(),
            "safe_block_00": GPT2SafeBlockStage(model.transformer.h[0]).eval(),
        }
        for boundary in (
            "attention_qkv",
            "attention_scores",
            "attention_masked_scores",
            "attention_softmax",
            "attention_context",
            "attention_projection",
            "attention_softmax_safe_mask",
            "attention_projection_safe_mask",
        ):
            auxiliary_stages[boundary] = GPT2AttentionPrefixStage(
                model.transformer.h[0], boundary
            ).eval()
        if args.device_stage in auxiliary_stages:
            auxiliary = auxiliary_stages[args.device_stage]
            func_name = auxiliary.__class__.__name__
            path = args.output_dir / f"gpt2_{args.device_stage}.mlirbc"
            export_stage(auxiliary, block_input, path)
            device_inputs = [block_input]
            with torch.no_grad():
                expected = auxiliary(block_input)
        elif args.device_stage == "embedding":
            path = args.output_dir / "gpt2_embedding.mlirbc"
            device_inputs = [input_ids]
            with torch.no_grad():
                expected = embedding(input_ids)
            func_name = "GPT2EmbeddingStage"
        elif args.device_stage == "block_00":
            path = args.output_dir / "gpt2_block_00.mlirbc"
            with torch.no_grad():
                expected = blocks[0](block_input)
            device_inputs = [block_input]
            func_name = "GPT2BlockStage"
        else:
            path = args.output_dir / "gpt2_head.mlirbc"
            with torch.no_grad():
                head_input = embedding(input_ids)
                for block in blocks:
                    head_input = block(head_input)
                head_input = head_input.detach()
                expected = head(head_input)
            device_inputs = [head_input]
            func_name = "GPT2HeadStage"

        print(
            f"[LayeredDevice] stage={args.device_stage} "
            f"vectorization=1 hexkl={int(args.enable_hexkl)}"
        )
        actual = TorchMLIRHexagonLauncher().run_torch_mlir(
            str(path),
            device_inputs,
            func_name,
            iterations=args.device_iterations,
            options=options,
        )[0]
        finite = bool(torch.isfinite(actual).all())
        max_abs = float((actual.float() - expected.float()).abs().max())
        mean_abs = float((actual.float() - expected.float()).abs().mean())
        print(
            f"[LayeredDeviceCompare] finite={finite} max_abs={max_abs:.9g} "
            f"mean_abs={mean_abs:.9g} actual_range="
            f"[{float(actual.min()):.9g},{float(actual.max()):.9g}] "
            f"expected_range=[{float(expected.min()):.9g},{float(expected.max()):.9g}]"
        )
        if not finite or max_abs > (0.1 if args.enable_hexkl else 0.03):
            raise AssertionError("layered GPT-2 device stage failed correctness gate")


if __name__ == "__main__":
    main()
