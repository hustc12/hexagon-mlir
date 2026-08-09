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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark_models"))
from run_gpt2lmheadmodel import freeze_gpt2_attn_bias_buffers  # noqa: E402


class GPT2EmbeddingStage(torch.nn.Module):
    def __init__(self, model: GPT2LMHeadModel, seq_len: int):
        super().__init__()
        self.wte = model.transformer.wte
        self.wpe = model.transformer.wpe
        self.drop = model.transformer.drop
        self.register_buffer(
            "position_ids",
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
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
        choices=("embedding", "block_00", "head"),
        help="Compile and execute one exported stage on the v73 device.",
    )
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--device-iterations", type=int, default=1)
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
        options["enableConversionToFp16"] = bool(args.enable_hexkl)
        options["lowerConstantsInSeparateSharedObjects"] = True
        if "enableBufferResultsToOutParams" in options:
            options["enableBufferResultsToOutParams"] = True

        if args.device_stage == "embedding":
            path = args.output_dir / "gpt2_embedding.mlirbc"
            device_inputs = [input_ids]
            with torch.no_grad():
                expected = embedding(input_ids)
            func_name = "GPT2EmbeddingStage"
        elif args.device_stage == "block_00":
            path = args.output_dir / "gpt2_block_00.mlirbc"
            with torch.no_grad():
                block_input = embedding(input_ids).detach()
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
        print(
            f"[LayeredDeviceCompare] finite={finite} max_abs={max_abs:.9g}"
        )
        if not finite or max_abs > (0.1 if args.enable_hexkl else 0.03):
            raise AssertionError("layered GPT-2 device stage failed correctness gate")


if __name__ == "__main__":
    main()
