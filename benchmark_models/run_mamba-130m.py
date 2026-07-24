"""Mamba-130M Hexagon Phase-4 harness (full published architecture)."""
from __future__ import annotations

from typing import Optional
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Allow `from hexkl_utils import ...` when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    patch_dsp_heap_256mb,
    encode_fixed_seq,
    compile_to_linalg,
    hex_execution,
    apply_hexkl_ir_rewrites,
    hexagon_options_phase4,
    add_phase4_args,
    phase4_kwargs_from_args,
)


def get_top_5(logits: torch.Tensor, tokenizer, run_type: str):
    print(f"\n-------Printing the top5 probable tokens for {run_type}--------\n")
    last_row = logits[0, -1, :]
    top_values, top_indices = torch.topk(last_row, 5)
    top_tokens = []
    for idx in top_indices:
        try:
            top_tokens.append(tokenizer.decode([idx]))
        except Exception:
            top_tokens.append(f"<id:{idx}>")
    for token, conf in zip(top_tokens, top_values.tolist()):
        print(f"Token: {[token]}, Confidence: {conf:.4f}")
    print("---------------------------------------------------\n")
    return top_tokens, top_values.tolist()


def compare(
    hex_outputs,
    x86_logits,
    tokenizer,
    atol=0.03,
    fail_on_mismatch: bool = False,
    require_exact_top5: bool = True,
):
    t_hex, c_hex = get_top_5(hex_outputs[0], tokenizer, "hexagon")
    t_x86, c_x86 = get_top_5(x86_logits, tokenizer, "x86")
    if require_exact_top5:
        ok = t_x86 == t_hex and torch.allclose(
            torch.tensor(c_x86), torch.tensor(c_hex), atol
        )
        msg = "The top5 tokens and their probabilities matched"
    else:
        ok = t_x86[0] == t_hex[0]
        msg = "Top-1 token matched (HexKL numerical tolerance)"
    if ok:
        print(msg)
    else:
        print("Hexagon and CPU results do not match")
        assert not fail_on_mismatch, "Correctness issue: Hexagon vs x86"


def customize_model_config(config):
    """Identity hook. Debug scripts may replace this to shrink topology."""
    return config


def load_mamba_model(model_name, config):
    # from_config: checkpoint vocab padding (50280 vs 50277) mismatches.
    return AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)


def mamba_130m(
    enable_hexkl: bool = False,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
    seq_len: Optional[int] = None,
):
    patch_dsp_heap_256mb()

    model_name = "state-spaces/mamba-130m"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = "Hi"

    if seq_len is None and enable_hexkl:
        seq_len = 32
    if seq_len is not None:
        if seq_len <= 0:
            raise ValueError(f"--seq-len must be positive, got {seq_len}")
        if enable_hexkl and seq_len % 32 != 0:
            raise ValueError(
                f"--seq-len={seq_len} is not a multiple of 32 (required for HexKL)"
            )
        input_ids, _, _ = encode_fixed_seq(tokenizer, prompt, seq_len)
        print(
            f"[Input] fair-compare seq_len={seq_len} "
            f"(content-filled, HexKL={enable_hexkl})"
        )
    else:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(torch.int64)
        print(f"[Input] default prompt seq_len={input_ids.shape[-1]}")

    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    config = customize_model_config(config)
    print(
        f"[Config] layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"vocab={config.vocab_size}"
    )

    model = load_mamba_model(model_name, config)
    model.eval()

    class MambaWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, input_ids):
            input_ids = torch.clamp(input_ids, 0, self.model.config.vocab_size - 1)
            return self.model(input_ids=input_ids).logits

    wrapped = MambaWrapper(model).eval()
    func_name = wrapped.__class__.__name__

    module = compile_to_linalg(wrapped, (input_ids,))
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )

    mlir_text = None
    if enable_hexkl:
        ir2, n_bm, n_f16 = apply_hexkl_ir_rewrites(ir)
        mlir_text = ir2
        print(f"[HexKL] batch_matmul→matmul={n_bm}, f16-input rewrite={n_f16}")

    options = hexagon_options_phase4(
        enable_hexkl,
        enable_omnifetch_vdae,
        enable_omnifetch_layout_aware,
        omnifetch_lookahead,
        enable_omnifetch_adaptive,
    )
    hex_outputs = hex_execution(
        module, func_name, [input_ids], options, mlir_text=mlir_text
    )
    print("Successfully ran Mamba-130M on Hexagon DSP!")

    with torch.no_grad():
        x86_logits = wrapped(input_ids)
    compare(
        hex_outputs,
        x86_logits,
        tokenizer,
        fail_on_mismatch=True,
        require_exact_top5=not enable_hexkl,
    )
    max_abs = (hex_outputs[0].float() - x86_logits.float()).abs().max().item()
    print(f"[Compare] max_abs_diff(logits)={max_abs:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mamba-130M Hexagon smoke (optional HexKL/OmniFetch)."
    )
    add_phase4_args(parser)
    args = parser.parse_args()
    mamba_130m(**phase4_kwargs_from_args(args))
