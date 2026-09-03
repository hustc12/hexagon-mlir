#!/usr/bin/env python3
"""Run the full published-shape SD/CLIP text encoder as v73 HVX stages."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import CLIPTextModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark_models"))
from hexkl_utils import patch_full_model_dsp_heap  # noqa: E402
from run_sd_text_encoder import (  # noqa: E402
    SD_MODEL_ID,
    _patch_gelu_tanh,
    get_text_inputs,
)
from layered_hvx_options import (  # noqa: E402
    add_layered_hvx_args,
    make_layered_hvx_options,
)


torch.set_num_threads(1)
torch.manual_seed(0)


class CLIPEmbeddingStage(torch.nn.Module):
    def __init__(self, model: CLIPTextModel, seq_len: int):
        super().__init__()
        embeddings = model.text_model.embeddings
        self.token_embedding = embeddings.token_embedding
        self.position_embedding = embeddings.position_embedding
        object.__setattr__(
            self,
            "position_ids",
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids) + self.position_embedding(
            self.position_ids
        )


class CLIPSafeLayerStage(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, seq_len: int, dtype: torch.dtype):
        super().__init__()
        self.layer = layer
        # All-ones tokenizer attention mask contributes zero.  The finite
        # causal sentinel is FP32-softmax equivalent and avoids the HVX
        # -FLT_MAX failure already isolated with full GPT-2.
        causal = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype)
        causal.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
            -1.0e4,
        )
        object.__setattr__(self, "attention_mask", torch.zeros_like(causal))
        object.__setattr__(self, "causal_attention_mask", causal)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer(
            hidden_states,
            attention_mask=self.attention_mask,
            causal_attention_mask=self.causal_attention_mask,
            output_attentions=False,
        )[0]


class CLIPFinalNormStage(torch.nn.Module):
    def __init__(self, model: CLIPTextModel):
        super().__init__()
        self.final_layer_norm = model.text_model.final_layer_norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.final_layer_norm(hidden_states)


def export_stage(stage: torch.nn.Module, example: torch.Tensor, path: Path) -> None:
    module = fx.export_and_import(
        stage,
        example,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=stage.__class__.__name__,
    )
    path.write_bytes(module.operation.get_asm(binary=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/clip-layered"))
    parser.add_argument("--device-iterations", type=int, default=1)
    add_layered_hvx_args(parser)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Uniform model and execution dtype; never enables mixed precision.",
    )
    args = parser.parse_args()

    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
    config = CLIPTextModel.config_class.from_pretrained(
        SD_MODEL_ID, subfolder="text_encoder"
    )
    model_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = CLIPTextModel(config).to(dtype=model_dtype).eval()
    _patch_gelu_tanh(model)
    seq_len = 77
    input_ids = get_text_inputs(
        tokenizer, "A beautiful picture of a Hexagon NPU", max_length=seq_len
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    embedding = CLIPEmbeddingStage(model, seq_len).eval()
    layers = [
        CLIPSafeLayerStage(layer, seq_len, model_dtype).eval()
        for layer in model.text_model.encoder.layers
    ]
    final_norm = CLIPFinalNormStage(model).eval()

    with torch.no_grad():
        reference = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        safe_reference = embedding(input_ids)
        for layer in layers:
            safe_reference = layer(safe_reference)
        safe_reference = final_norm(safe_reference)
    cpu_max_abs = float((safe_reference - reference).abs().max())
    print(
        f"[CLIPLayeredCPU] layers={len(layers)} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads} vocab={config.vocab_size} "
        f"dtype={args.dtype} "
        f"seq_len={seq_len} max_abs={cpu_max_abs:.9g}"
    )
    if cpu_max_abs > 1.0e-4:
        raise AssertionError("safe staged CLIP differs from the formal CPU model")
    if args.cpu_only:
        return

    from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import (
        TorchMLIRHexagonLauncher,
    )

    patch_full_model_dsp_heap()
    options = make_layered_hvx_options(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    launcher = TorchMLIRHexagonLauncher()

    def run_stage(
        stage: torch.nn.Module, stage_input: torch.Tensor, label: str
    ) -> torch.Tensor:
        path = args.output_dir / f"clip_{label}.mlirbc"
        export_stage(stage, stage_input, path)
        with torch.no_grad():
            expected = stage(stage_input)
        print(f"[CLIPDeviceStage] begin={label}")
        actual = launcher.run_torch_mlir(
            str(path),
            [stage_input],
            stage.__class__.__name__,
            iterations=args.device_iterations,
            options=options,
        )[0]
        finite = bool(torch.isfinite(actual).all())
        max_abs = float((actual.float() - expected.float()).abs().max())
        print(
            f"[CLIPDeviceStage] end={label} finite={finite} "
            f"max_abs={max_abs:.9g}"
        )
        if not finite or max_abs > 0.05:
            raise AssertionError(f"CLIP device stage {label} failed correctness")
        return actual.detach()

    hidden = run_stage(embedding, input_ids, "embedding")
    for index, layer in enumerate(layers):
        hidden = run_stage(layer, hidden, f"layer_{index:02d}")
    actual = run_stage(final_norm, hidden, "final_norm")
    final_max_abs = float((actual.float() - reference.float()).abs().max())
    print(
        f"[CLIPDeviceFullCompare] layers={len(layers)} finite="
        f"{bool(torch.isfinite(actual).all())} max_abs={final_max_abs:.9g}"
    )
    if final_max_abs > 0.1:
        raise AssertionError("full staged CLIP failed correctness gate")


if __name__ == "__main__":
    main()
