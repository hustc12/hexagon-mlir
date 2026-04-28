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

def qwen2_5_0_5b(enablelwp=False): 

    model_name = "Qwen/Qwen2.5-0.5B"
    prompt = "What is nature of our existence?"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(model_name)
    # Using 2 layers for a "Lite" compilation test, similar to GPT-2
    config.num_hidden_layers = 1
    config.use_cache = False  # Disable KV cache to prevent multiple outputs (past_key_values)
    config.vocab_size = 128   # Reduce vocab size to prevent stack overflow from huge intermediate logits
    config.max_position_embeddings = 128 # Reduce RoPE cache size to prevent stack overflow
    config.hidden_size = 128
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_key_value_heads = 4

    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()

    import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling
    # Monkey patch RoPE
    def identity_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        return q, k
    qwen2_modeling.apply_rotary_pos_emb = identity_rope

    class QwenWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.attn = m.model.layers[0].self_attn
            self.rotary_emb = m.model.rotary_emb

        def forward(self, hidden_states, attention_mask, position_ids):
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            return self.attn(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings
            )[0]

    wrapped_model = QwenWrapper(model)
    wrapped_model.eval()
    func_name = wrapped_model.__class__.__name__

    # Generate dummy inputs
    hidden_states = torch.randn(1, 7, config.hidden_size, dtype=torch.float16)
    attention_mask = torch.zeros(1, 1, 7, 7, dtype=torch.float16)
    position_ids = torch.arange(0, 7, dtype=torch.long).unsqueeze(0)
    
    module = compile_to_linalg(wrapped_model, (hidden_states, attention_mask, position_ids))

    options = HexagonOptions().__dict__
    if enablelwp:
        options['enableLWP'] = True
    options['lowerConstantsInSeparateSharedObjects'] = False
    
    inv_freq_buffer = wrapped_model.rotary_emb.inv_freq
    inputs = [inv_freq_buffer, hidden_states, attention_mask, position_ids]
    
    # Run Hexagon
    hex_outputs = hex_execution(module, func_name, inputs, options)

    # Run x86
    with torch.no_grad():
        x86_outputs = wrapped_model(*inputs)

    compare(hex_outputs, [x86_outputs], tokenizer, fail_on_mismatch=True)
    if enablelwp:
        process_lwp()

if __name__ == "__main__":
    qwen2_5_0_5b()
