from typing import Optional
import sys, os
import torch
import argparse
import subprocess
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from pathlib import Path
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher

def get_encodings(tokenizer, *inputs):
    encodings = tokenizer(*inputs, return_tensors="pt")
    return encodings

def x86_execution(model, encoding):
    x86_outputs = model(**encoding)
    return x86_outputs

def hex_execution(module, func_name, inputs, options: dict=None):
    linalg_filename = Path(__file__).parent / (str(func_name) + ".mlirbc")

    bytecode = module.operation.get_asm(binary=True)
    # Save the bytecode to a file
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    hex_outputs = TorchMLIRHexagonLauncher().run_torch_mlir(str(linalg_filename), inputs, func_name, options=options)
    return hex_outputs

# logits is expected to be "[batch_size, sequence_length, vocab_size]"
def get_top_5(logits: torch.Tensor, tokenizer, run_type: str):
    print(f"\n-------Printing the top5 probable tokens for {run_type}--------\n")
    top_k = 5

    if logits.ndim != 3:
        raise ValueError(f"Expected logits to be a 3D tensor, but got shape {logits.shape}")
    
    last_row_logits = logits[0, -1, :]
    top_values, top_indices= torch.topk(last_row_logits, top_k)
    top_confidences = top_values.tolist()

    # Convert indices to tokens
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    for token, confidence in zip(top_tokens, top_confidences):
        print(f"Token: {[token]}, Confidence: {confidence:.4f}")
    print("---------------------------------------------------\n")
    return top_tokens, top_confidences

def compare(hex_outputs, x86_outputs, tokenizer, atol=0.03, fail_on_mismatch: bool=False):
    hexagon_logits = hex_outputs[0]
    t_hex, c_hex = get_top_5(hexagon_logits, tokenizer, "hexagon")

    x86_logits = x86_outputs.logits
    t_x86, c_x86 = get_top_5(x86_logits, tokenizer, "x86")

    tokens_match = (t_x86 == t_hex)
    confidences_match = torch.allclose(torch.tensor(c_x86), torch.tensor(c_hex), atol)

    if tokens_match and confidences_match:
        print("The top5 tokens and their probabilities matched")
    else:
        print("Hexagon and CPU results do not match")
        assert not fail_on_mismatch, "Correctness issue: the results obtained on Hexagon (with code produced by the hexagon-mlir compiler) and on x86 (executed from PyTorch) do not match"

def compile_to_linalg(model, input, dump_to_file=None, debug=False) -> str:
    if isinstance(input, torch.Tensor):
        input = (input,)

    # Generate linalg-IR using torch-mlir's fx
    linalg = fx.export_and_import(
        model,
        *input,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug
    )

    if dump_to_file:
        with open(dump_to_file, "w") as file:
            file.write(str(linalg))

    return linalg

def process_lwp():
    HEXAGON_MLIR_ROOT = os.environ.get("HEXAGON_MLIR_ROOT")
        
    if not HEXAGON_MLIR_ROOT:
        print("Cannot process lwp data as path to process_lwp.py is unknown")
        return

    try:
        subprocess.run(
            [
                "python3",
                f"{HEXAGON_MLIR_ROOT}/test/python/process_lwp.py",
                "/tmp/lwp.json",
                "/tmp/lwp_infodump.txt",
                "/tmp/initial-linalg.mlir"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print("LWP processing completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error processing LWP data: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Error output: {e.stderr}")

def gpt2lmheadmodel(
    enablelwp: bool = False,
    # Mirror HexagonOptions defaults: HexKL off, VTCM tiling and hexagonmem
    # conversion also off for GPT2 (mixed f32/f16 pipeline).
    enable_hexkl: bool = False,
    enable_vtcm_tiling: bool = False,
    enable_convert_to_hexagonmem: bool = False,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
): 

    model_name = "openai-community/gpt2"
    prompt = "What is nature of our existence?"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    config = GPT2Config.from_pretrained(model_name)
    config.n_layer = 2 # layer == 1 isn't practical for checking accuracy

    model = GPT2LMHeadModel.from_pretrained(model_name, config=config, torch_dtype=torch.float16)
    func_name=model.__class__.__name__

    encoding = get_encodings(tokenizer, prompt)
    module = compile_to_linalg(model, encoding["input_ids"])

    options = HexagonOptions().__dict__
    options["enableHexKL"] = enable_hexkl
    options["enableVTCMTiling"] = enable_vtcm_tiling
    options["enableConvertToHexagonmem"] = enable_convert_to_hexagonmem
    # Omni-Fetch Plan-A: V-DAE (Component 3) implies Prefetch (Component 1).
    options["enablePrefetch"] = enable_omnifetch_vdae
    options["enableOmniFetchLayoutAware"] = enable_omnifetch_layout_aware
    options["omniFetchLookahead"] = omnifetch_lookahead
    options["enableOmniFetchVDAE"] = enable_omnifetch_vdae
    options["enableOmniFetchAdaptive"] = enable_omnifetch_adaptive
    if enablelwp:
        options['enableLWP'] = True
    inputs = [encoding["input_ids"]]
    hex_outputs = hex_execution(module, func_name, inputs, options)
    x86_outputs = x86_execution(model, encoding)

    compare(hex_outputs, x86_outputs, tokenizer, fail_on_mismatch=True)
    if enablelwp:
        process_lwp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT2 LMHead on Hexagon with optional Omni-Fetch ablation toggles.")
    parser.add_argument("--enable-lwp", action="store_true", help="Enable lightweight profiling instrumentation.")

    parser.add_argument("--enable-hexkl", action="store_true",
                        help="Enable HexKL lowering (requires a fully-f16 pipeline; off by default for GPT2).")
    parser.add_argument("--enable-vtcm-tiling", action="store_true",
                        help="Enable VTCM tiling (requires a fully-f16 pipeline; off by default for GPT2).")
    parser.add_argument("--enable-convert-to-hexagonmem", action="store_true",
                        help="Enable memref->hexagonmem conversion (off by default for GPT2).")

    parser.add_argument("--enable-omnifetch-vdae", action="store_true",
                        help="Enable Omni-Fetch V-DAE prefetch pass.")
    parser.add_argument("--disable-layout-aware", action="store_true",
                        help="Disable layout-aware in-situ mapping (linear prefetch only).")
    parser.add_argument("--omnifetch-lookahead", type=int, default=2,
                        help="Static prefetch look-ahead distance.")
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true",
                        help="Disable PMU-driven adaptive prefetch distance.")

    args = parser.parse_args()

    gpt2lmheadmodel(
        enablelwp=args.enable_lwp,
        enable_hexkl=args.enable_hexkl,
        enable_vtcm_tiling=args.enable_vtcm_tiling,
        enable_convert_to_hexagonmem=args.enable_convert_to_hexagonmem,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
    )


