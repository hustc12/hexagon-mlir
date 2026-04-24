from typing import Optional
import sys, os
import torch
import argparse
import subprocess
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import AutoModel, AutoTokenizer, AutoConfig
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

def compare(hex_outputs, x86_outputs, atol=0.03, fail_on_mismatch: bool=False):
    hexagon_output = hex_outputs[0]
    
    if hasattr(x86_outputs, "last_hidden_state"):
        x86_tensor = x86_outputs.last_hidden_state
    else:
        x86_tensor = x86_outputs[0]

    max_diff = torch.max(torch.abs(hexagon_output - x86_tensor))
    print(f"\nMax difference between Hexagon and x86 outputs: {max_diff.item():.4f}")

    match = torch.allclose(hexagon_output, x86_tensor, atol=atol)

    if match:
        print("Hexagon and CPU results matched within the specified tolerance.")
    else:
        print("Hexagon and CPU results do not match.")
        assert not fail_on_mismatch, "Correctness issue: the results obtained on Hexagon and on x86 do not match"

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

def graphsage_bert(enablelwp=False): 

    model_name = "andorei/gebert_eng_graphsage"
    prompt = "Understanding graph neural networks"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    
    # Using 2 layers for a "Lite" compilation test
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = 2 

    # BERT's default "gelu" activation uses math.erf, which the Hexagon MLIR backend
    # does not support in LLVM IR translation. Switch to "gelu_new" which uses a
    # tanh-based polynomial approximation instead, avoiding the math.erf op entirely.
    config.hidden_act = "gelu_new"

    # # Aggressively shrink model dimensions for DSP memory constraints.
    # config.vocab_size = 1000
    # config.hidden_size = 128
    # config.intermediate_size = 256
    # config.num_attention_heads = 2
    # # Reduce max_position_embeddings to minimize the registered buffer sizes
    # # that torch-mlir lifts as extra function arguments.
    # config.max_position_embeddings = 16

    # config.vocab_size = 30522
    # config.hidden_size = 768
    # config.intermediate_size = 3072
    # config.num_attention_heads = 12
    # # Reduce max_position_embeddings to minimize the registered buffer sizes
    # # that torch-mlir lifts as extra function arguments.
    # config.max_position_embeddings = 512
    # Use float16 to halve memory footprint, matching the GPT-2 test approach.
    # Using from_config with random weights since we changed vocab_size.
    model = AutoModel.from_config(config, torch_dtype=torch.float32)
    model.eval()

    # Wrap BERT to return only last_hidden_state for single-output compilation.
    #
    # CRITICAL: torch-mlir fx.export traces BERT's BertEmbeddings registered buffers
    # (position_ids and token_type_ids) and lifts them as extra function arguments.
    # The compiled function signature becomes:
    #   func(position_ids_buffer, token_type_ids_buffer, input_ids) -> output
    # We must pass all 3 tensors to match this signature, otherwise the DSP
    # reads garbage from unmapped registers and crashes with TLBMISS.
    class BertSingleOutputWrapper(torch.nn.Module):
        def __init__(self, bert_model):
            super().__init__()
            self.bert = bert_model
        def forward(self, input_ids):
            return self.bert(input_ids=input_ids).last_hidden_state

    wrapped_model = BertSingleOutputWrapper(model)
    wrapped_model.eval()
    func_name = wrapped_model.__class__.__name__

    encoding = get_encodings(tokenizer, prompt)
    
    # Clamp input_ids to the reduced vocab_size
    encoding["input_ids"] = encoding["input_ids"].clamp(max=config.vocab_size - 1)

    # Compile with just input_ids — torch-mlir will trace and discover the
    # registered buffers, adding them as extra arguments automatically
    module = compile_to_linalg(wrapped_model, encoding["input_ids"])

    options = HexagonOptions().__dict__
    if enablelwp:
        options['enableLWP'] = True
    
    # Construct the registered buffer tensors that torch-mlir lifted as extra args.
    # BERT's BertEmbeddings has:
    #   position_ids: shape [1, max_position_embeddings] = [1, 16]
    #   token_type_ids: shape [1, max_position_embeddings] = [1, 16]
    seq_len = encoding["input_ids"].shape[1]
    position_ids_buffer = torch.arange(config.max_position_embeddings).unsqueeze(0)
    token_type_ids_buffer = torch.zeros(1, config.max_position_embeddings, dtype=torch.long)
    
    # The compiled function signature is:
    #   func(position_ids_buffer, token_type_ids_buffer, input_ids) -> output
    inputs = [position_ids_buffer, token_type_ids_buffer, encoding["input_ids"]]
    
    # Run Hexagon
    hex_outputs = hex_execution(module, func_name, inputs, options)
    
    # Run x86
    with torch.no_grad():
        x86_outputs = wrapped_model(encoding["input_ids"])

    compare(hex_outputs, x86_outputs, fail_on_mismatch=True)
    if enablelwp:
        process_lwp()

if __name__ == "__main__":
    graphsage_bert()
