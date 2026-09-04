"""GeBERT / GraphSAGE Hexagon Phase-4 harness (full published BERT structure)."""
from __future__ import annotations

from typing import Optional
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

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


def compare(
    hex_outputs,
    x86_tensor,
    atol=0.03,
    fail_on_mismatch: bool = False,
):
    hexagon_output = hex_outputs[0]
    max_diff = (hexagon_output.float() - x86_tensor.float()).abs().max().item()
    print(f"\nMax difference between Hexagon and x86 outputs: {max_diff:.4f}")
    match = torch.allclose(hexagon_output.float(), x86_tensor.float(), atol=atol)
    if match:
        print("Hexagon and CPU results matched within the specified tolerance.")
    else:
        print("Hexagon and CPU results do not match.")
        assert not fail_on_mismatch, "Correctness issue: Hexagon vs x86"


def customize_model_config(config):
    config.hidden_act = "gelu_new"
    return config


def load_graphsage_model(_model_name, config):
    # Eager attention avoids tm_tensor.attention (cannot re-parse for HexKL rewrites).
    return AutoModel.from_config(
        config, torch_dtype=torch.float16, attn_implementation="eager"
    )


def graphsage_bert(
    enable_hexkl: bool = False,
    enable_alps_vdae: bool = False,
    enable_alps_layout_aware: bool = True,
    alps_lookahead: int = 2,
    enable_alps_adaptive: bool = True,
    enable_alps_items_1_7: bool = False,
    seq_len: Optional[int] = None,
):
    patch_dsp_heap_256mb()
    model_name = "andorei/gebert_eng_graphsage"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Hi"

    if seq_len is None and enable_hexkl:
        seq_len = 32
    if seq_len is not None:
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
    config = customize_model_config(config)
    print(
        f"[Config] layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"vocab={config.vocab_size} heads={config.num_attention_heads}"
    )

    model = load_graphsage_model(model_name, config)
    model.eval()

    class BertSingleOutputWrapper(torch.nn.Module):
        def __init__(self, bert_model):
            super().__init__()
            self.bert = bert_model

        def forward(self, input_ids):
            input_ids = torch.clamp(input_ids, 0, self.bert.config.vocab_size - 1)
            return self.bert(input_ids=input_ids).last_hidden_state

    wrapped = BertSingleOutputWrapper(model).eval()
    func_name = wrapped.__class__.__name__

    module = compile_to_linalg(wrapped, (input_ids,), decomp_pow=False)
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

    # BERT lifts position_ids / token_type_ids buffers as extra ABI args.
    position_ids_buffer = torch.arange(config.max_position_embeddings).unsqueeze(0)
    token_type_ids_buffer = torch.zeros(
        1, config.max_position_embeddings, dtype=torch.long
    )
    inputs = [position_ids_buffer, token_type_ids_buffer, input_ids]

    options = hexagon_options_phase4(
        enable_hexkl,
        enable_alps_vdae,
        enable_alps_layout_aware,
        alps_lookahead,
        enable_alps_adaptive,
        enable_alps_items_1_7,
    )
    hex_outputs = hex_execution(
        module, func_name, inputs, options, mlir_text=mlir_text
    )
    print("Successfully ran GraphSAGE/BERT on Hexagon DSP!")

    with torch.no_grad():
        x86 = wrapped(input_ids)
    compare(
        hex_outputs,
        x86,
        atol=0.5 if enable_hexkl else 0.03,
        fail_on_mismatch=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GraphSAGE/GeBERT Hexagon smoke (optional HexKL/Alps)."
    )
    add_phase4_args(parser)
    args = parser.parse_args()
    graphsage_bert(**phase4_kwargs_from_args(args))
