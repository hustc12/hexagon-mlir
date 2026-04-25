from typing import Optional
import sys, os
import torch
import argparse
import subprocess
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from pathlib import Path
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher

def get_text_inputs(tokenizer, prompt):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_input.input_ids

def x86_execution(model, *inputs):
    x86_outputs = model(*inputs)
    return x86_outputs

def hex_execution(module, func_name, inputs, options: dict=None):
    linalg_filename = Path(__file__).parent / (str(func_name) + ".mlirbc")

    bytecode = module.operation.get_asm(binary=True)
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = False 
    hex_outputs = TorchMLIRHexagonLauncher().run_torch_mlir(str(linalg_filename), inputs, func_name, options=options)
    return hex_outputs

def compare(hex_outputs, x86_outputs, atol=0.05, fail_on_mismatch: bool=False):
    # Hexagon executor returns a tuple, we take the first element
    hexagon_tensor = hex_outputs[0]
    
    # Depending on the model, x86_outputs could be a tensor or a tuple/dataclass
    if isinstance(x86_outputs, torch.Tensor):
        x86_tensor = x86_outputs
    elif hasattr(x86_outputs, "sample"): 
        x86_tensor = x86_outputs.sample
    elif hasattr(x86_outputs, "last_hidden_state"): 
        x86_tensor = x86_outputs.last_hidden_state
    else:
        x86_tensor = x86_outputs[0]
    
    max_diff = torch.max(torch.abs(hexagon_tensor - x86_tensor))
    print(f"\nMax difference between Hexagon and x86 outputs: {max_diff.item():.4f}")

    match = torch.allclose(hexagon_tensor, x86_tensor, atol=atol)

    if match:
        print("Hexagon and CPU results matched within the specified tolerance.")
    else:
        print("Hexagon and CPU results do not match.")
        assert not fail_on_mismatch, "Correctness issue: results do not match"

def compile_to_linalg(model, *inputs, dump_to_file=None, debug=False) -> str:
    linalg = fx.export_and_import(
        model,
        *inputs,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug
    )

    if dump_to_file:
        with open(dump_to_file, "w") as file:
            file.write(str(linalg))

    return linalg

# Wrapper to explicitly test the VAE decode method which is the most compute-intensive part of VAE used in SD
class VAEDecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, latents):
        return self.vae.decode(latents).sample

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

def test_stable_diffusion(enablelwp=False):
    model_id = "CompVis/stable-diffusion-v1-4"
    options = HexagonOptions().__dict__
    if enablelwp:
        options['enableLWP'] = True

    # ---------------------------------------------------------
    # 1. Text Encoder
    # ---------------------------------------------------------
    print("\n--- Testing Text Encoder ---")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    # For quick testing compilation accuracy, we use from_config. 
    # To use real weights, change to from_pretrained
    config_te = CLIPTextModel.config_class.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder = CLIPTextModel(config_te)
    text_encoder.eval()
    
    input_ids = get_text_inputs(tokenizer, "A beautiful picture of a Hexagon NPU")
    
    print("Compiling Text Encoder...")
    te_module = compile_to_linalg(text_encoder, input_ids)
    print("Running Text Encoder on Hexagon...")
    te_hex_outputs = hex_execution(te_module, "CLIPTextModel", [input_ids], options)
    with torch.no_grad():
        te_x86_outputs = x86_execution(text_encoder, input_ids)
    compare(te_hex_outputs, te_x86_outputs, fail_on_mismatch=True)

    # ---------------------------------------------------------
    # 2. UNet
    # ---------------------------------------------------------
    print("\n--- Testing UNet ---")
    # Use from_config to initialize without downloading huge 3.4GB weights for local test compilation.
    config_unet = UNet2DConditionModel.load_config(model_id, subfolder="unet")
    # Note: To download real weights, use UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    unet = UNet2DConditionModel.from_config(config_unet)
    unet.eval()

    # Inputs for UNet (batch_size=1, in_channels=4, height=64, width=64 for 512x512 image)
    latent_model_input = torch.rand(1, 4, 64, 64)
    timestep = torch.tensor([1.0], dtype=torch.float32)
    # encoder_hidden_states shape from Text Encoder (batch_size=1, seq_len=77, hidden_size=768)
    encoder_hidden_states = torch.rand(1, 77, 768)

    unet_inputs = (latent_model_input, timestep, encoder_hidden_states)

    print("Compiling UNet...")
    unet_module = compile_to_linalg(unet, *unet_inputs)
    print("Running UNet on Hexagon...")
    unet_hex_outputs = hex_execution(unet_module, "UNet2DConditionModel", list(unet_inputs), options)
    with torch.no_grad():
        unet_x86_outputs = x86_execution(unet, *unet_inputs)
    compare(unet_hex_outputs, unet_x86_outputs, fail_on_mismatch=True)

    # ---------------------------------------------------------
    # 3. VAE (Decoder)
    # ---------------------------------------------------------
    print("\n--- Testing VAE Decoder ---")
    config_vae = AutoencoderKL.load_config(model_id, subfolder="vae")
    vae = AutoencoderKL.from_config(config_vae)
    vae_wrapper = VAEDecodeWrapper(vae)
    vae_wrapper.eval()

    # Input to VAE decode is the latent representation
    latents = torch.rand(1, 4, 64, 64)
    vae_inputs = (latents,)

    print("Compiling VAE...")
    vae_module = compile_to_linalg(vae_wrapper, *vae_inputs)
    print("Running VAE on Hexagon...")
    vae_hex_outputs = hex_execution(vae_module, "VAEDecodeWrapper", list(vae_inputs), options)
    with torch.no_grad():
        vae_x86_outputs = x86_execution(vae_wrapper, *vae_inputs)
    compare(vae_hex_outputs, vae_x86_outputs, fail_on_mismatch=True)

    if enablelwp:
        process_lwp()

if __name__ == "__main__":
    test_stable_diffusion()
