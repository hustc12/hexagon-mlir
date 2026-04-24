from typing import Optional
import sys, os
import torch
import argparse
import subprocess
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher

def get_encodings(tokenizer, *inputs):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = False 
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

def x86_execution(model, inputs):
    with torch.no_grad():
        x86_outputs = model(*inputs)
    return x86_outputs

def qwen2_5_0_5b():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "What is nature of our existence?"
    
    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    
    # Restore FULL model parameters (BERT-base scale or full scale)
    # The DSP memory size and execution environment are now stable.
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    class QwenWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, input_ids, attention_mask, position_ids):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            ).logits

    wrapped_model = QwenWrapper(model)
    wrapped_model.eval()
    func_name = wrapped_model.__class__.__name__

    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"]
    # Provide explicit attention_mask and position_ids to avoid dynamic mask generation inside the model
    attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long).unsqueeze(0)

    # Convert to float32 to avoid float16 precision alignment issues on some Hexagon ops
    wrapped_model = wrapped_model.to(torch.float32)
    input_ids = input_ids.to(torch.int64)

    module = compile_to_linalg(wrapped_model, (input_ids, attention_mask, position_ids))

    options = HexagonOptions().__dict__
    options['enableLWP'] = False
    options['lowerConstantsInSeparateSharedObjects'] = True

    # Lifted buffer extraction
    inv_freq_buffer = wrapped_model.model.model.rotary_emb.inv_freq.float()
    inputs = [inv_freq_buffer, input_ids, attention_mask, position_ids]

    hex_outputs = hex_execution(module, func_name, inputs, options)
    print("Successfully ran full Qwen model on Hexagon DSP!")

if __name__ == '__main__':
    qwen2_5_0_5b()
