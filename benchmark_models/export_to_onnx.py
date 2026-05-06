import os
import torch
import torch.onnx
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    CLIPTextModel,
    CLIPTokenizer,
)
from diffusers import UNet2DConditionModel, AutoencoderKL
from pathlib import Path

# Note: Real-ESRGAN needs to be installed via `pip install RealESRGAN`
# If not available, we skip it or use a placeholder.
try:
    from RealESRGAN import RealESRGAN
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    HAS_REAL_ESRGAN = True
except ImportError:
    HAS_REAL_ESRGAN = False

def export_falcon():
    print("Exporting Falcon-1B...")
    model_name = "Rocketknight1/falcon-rw-1b"
    config = AutoConfig.from_pretrained(model_name)
    # Original size (default config)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    
    dummy_input = torch.randint(0, config.vocab_size, (1, 128))
    torch.onnx.export(
        model,
        dummy_input,
        "falcon_rw_1b.onnx",
        opset_version=14,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {1: 'sequence_length'}, 'logits': {1: 'sequence_length'}}
    )
    print("Falcon-1B exported.")

def export_gpt2():
    print("Exporting GPT2...")
    model_name = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 128))
    torch.onnx.export(
        model,
        dummy_input,
        "gpt2.onnx",
        opset_version=14,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {1: 'sequence_length'}, 'logits': {1: 'sequence_length'}}
    )
    print("GPT2 exported.")

def export_graphsage():
    print("Exporting GraphSAGE-BERT...")
    model_name = "andorei/gebert_eng_graphsage"
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 128))
    torch.onnx.export(
        model,
        dummy_input,
        "graphsage_bert.onnx",
        opset_version=14,
        input_names=['input_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={'input_ids': {1: 'sequence_length'}, 'last_hidden_state': {1: 'sequence_length'}}
    )
    print("GraphSAGE-BERT exported.")

def export_mamba():
    print("Exporting Mamba-130M...")
    model_name = "state-spaces/mamba-130m-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 128))
    torch.onnx.export(
        model,
        dummy_input,
        "mamba_130m.onnx",
        opset_version=14,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {1: 'sequence_length'}, 'logits': {1: 'sequence_length'}}
    )
    print("Mamba-130M exported.")

def export_qwen():
    print("Exporting Qwen-2.5-0.5B...")
    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    
    seq_len = 128
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
    attention_mask = torch.ones((1, seq_len))
    position_ids = torch.arange(0, seq_len).unsqueeze(0)
    
    torch.onnx.export(
        model,
        (input_ids, attention_mask, position_ids),
        "qwen2.5_0.5b.onnx",
        opset_version=14,
        input_names=['input_ids', 'attention_mask', 'position_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {1: 'sequence_length'}, 'attention_mask': {1: 'sequence_length'}, 'position_ids': {1: 'sequence_length'}, 'logits': {1: 'sequence_length'}}
    )
    print("Qwen-2.5-0.5B exported.")

def export_real_esrgan():
    if not HAS_REAL_ESRGAN:
        print("Skipping Real-ESRGAN (package not installed).")
        return
    print("Exporting Real-ESRGAN...")
    device = torch.device('cpu')
    model_wrapper = RealESRGAN(device, scale=4)
    weights_path = huggingface_hub.hf_hub_download('ai-forever/Real-ESRGAN', 'RealESRGAN_x4.pth')
    model_wrapper.load_weights(weights_path, download=False)
    model = model_wrapper.model
    model.eval()
    
    dummy_input = torch.rand(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "real_esrgan_x4.onnx",
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
    )
    print("Real-ESRGAN exported.")

def export_swin():
    print("Exporting Swin-Tiny...")
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    model = SwinForImageClassification.from_pretrained(model_name)
    model.eval()
    
    dummy_input = torch.rand(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "swin_tiny.onnx",
        opset_version=14,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={'pixel_values': {2: 'height', 3: 'width'}}
    )
    print("Swin-Tiny exported.")

def export_tinyllama():
    print("Exporting TinyLlama-1.1B...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 128))
    torch.onnx.export(
        model,
        dummy_input,
        "tinyllama_1.1b.onnx",
        opset_version=14,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {1: 'sequence_length'}, 'logits': {1: 'sequence_length'}}
    )
    print("TinyLlama-1.1B exported.")

def export_vit():
    print("Exporting ViT-Base...")
    model_name = "google/vit-base-patch16-224"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    
    dummy_input = torch.rand(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "vit_base.onnx",
        opset_version=14,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={'pixel_values': {2: 'height', 3: 'width'}}
    )
    print("ViT-Base exported.")

def export_sd_components():
    print("Exporting Stable Diffusion Components...")
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # Text Encoder
    print("  Exporting CLIP Text Encoder...")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder.eval()
    dummy_input = torch.randint(0, 49408, (1, 77))
    torch.onnx.export(
        text_encoder,
        dummy_input,
        "sd_text_encoder.onnx",
        opset_version=14,
        input_names=['input_ids'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={'input_ids': {1: 'sequence_length'}}
    )
    
    # UNet
    print("  Exporting UNet...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    unet.eval()
    latent_model_input = torch.rand(1, 4, 64, 64)
    timestep = torch.tensor([1.0])
    encoder_hidden_states = torch.rand(1, 77, 768)
    torch.onnx.export(
        unet,
        (latent_model_input, timestep, encoder_hidden_states),
        "sd_unet.onnx",
        opset_version=14,
        input_names=['sample', 'timestep', 'encoder_hidden_states'],
        output_names=['out_sample']
    )
    
    # VAE Decoder
    print("  Exporting VAE Decoder...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae.eval()
    class VAEDecodeWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        def forward(self, latents):
            return self.vae.decode(latents).sample
            
    vae_decoder = VAEDecodeWrapper(vae)
    latents = torch.rand(1, 4, 64, 64)
    torch.onnx.export(
        vae_decoder,
        latents,
        "sd_vae_decoder.onnx",
        opset_version=14,
        input_names=['latents'],
        output_names=['sample']
    )
    print("Stable Diffusion Components exported.")

if __name__ == "__main__":
    export_falcon()
    export_gpt2()
    export_graphsage()
    export_mamba()
    export_qwen()
    export_real_esrgan()
    export_swin()
    export_tinyllama()
    export_vit()
    export_sd_components()
    print("\nAll models exported successfully (where possible).")
