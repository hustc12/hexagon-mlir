from typing import Optional
import sys, os
import re
import torch
import argparse
import subprocess
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from pathlib import Path
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher
from triton.backends.qcom_hexagon_backend import hexagon_launcher_base as _hlb

# HMX tiles are 32-wide; keep DSP heap inside the mapped TLB window (same as
# verify_omnifetch_Attention.py).
_QURT_HEAP_1GB = "unsigned int _QURT_MAX_HEAP_SIZE = 1073741824; // 1 GB Max Heap Size"
_QURT_HEAP_256MB = "unsigned int _QURT_MAX_HEAP_SIZE = 268435456;  // 256 MB Max Heap Size"


def _patch_dsp_heap_256mb():
    orig_init = _hlb.WrapperGeneratorStrings.__init__

    def _patched_init(self):
        orig_init(self)
        self.code_string = self.code_string.replace(_QURT_HEAP_1GB, _QURT_HEAP_256MB)

    _hlb.WrapperGeneratorStrings.__init__ = _patched_init


def _truncf_generic(ssa: str, src: str, shape: str, indent: str) -> str:
    """Emit a 2D elementwise truncf f32→f16 linalg.generic."""
    init = f"{ssa}_init"
    return (
        f"{indent}{init} = tensor.empty() : tensor<{shape}xf16>\n"
        f"{indent}{ssa} = linalg.generic "
        f"{{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
        f"affine_map<(d0, d1) -> (d0, d1)>], "
        f'iterator_types = ["parallel", "parallel"]}} '
        f"ins({src} : tensor<{shape}xf32>) outs({init} : tensor<{shape}xf16>) {{\n"
        f"{indent}^bb0(%in: f32, %out: f16):\n"
        f"{indent}  %t = arith.truncf %in : f32 to f16\n"
        f"{indent}  linalg.yield %t : f16\n"
        f"{indent}}} -> tensor<{shape}xf16>\n"
    )


def rewrite_matmul_inputs_to_f16(ir: str) -> tuple[str, int]:
    """Make HexKL-ready f16 matmul inputs (f32 accumulators kept).

    Two export styles:
    1) float16 model: extf(f16→f32) + matmul(f32) → undo extf (use f16 SSA).
    2) float32 model: matmul(f32) → insert truncf on both inputs (keeps
       LayerNorm / softmax / residual in f32 — required to avoid f16 NaN
       explosion on full-depth GPT-2).

    Skips oversized lines (dialect dense_resource blobs) — regex on multi‑MB
    hex lines can hang the host for tens of minutes.
    """
    _MAX_LINE = 16384
    lines = ir.splitlines(keepends=True)
    extf = {}
    i = 0
    while i < len(lines):
        if len(lines[i]) > _MAX_LINE:
            i += 1
            continue
        m = re.match(r"(\s*)(%[\w]+)\s*=\s*linalg\.generic\b", lines[i])
        if not m:
            i += 1
            continue
        res = m.group(2)
        block = lines[i]
        j = i
        while j < len(lines) and not re.search(r"\}\s*->\s*tensor<", lines[j]):
            j += 1
            if j < len(lines):
                if len(lines[j]) > _MAX_LINE:
                    break
                block += lines[j]
        if j >= len(lines) or len(lines[j]) > _MAX_LINE:
            i = j + 1
            continue
        if "arith.extf" in block and block.count("arith.") == 1:
            ins = re.search(
                r"ins\((%[\w]+)\s*:\s*(tensor<[^>]+xf16>)\)\s*outs",
                block,
            )
            if ins:
                extf[res] = (ins.group(1), ins.group(2))
        i = j + 1

    out = []
    rewrites = 0
    trunc_id = 0
    for line in lines:
        if len(line) > _MAX_LINE or "linalg.matmul" not in line:
            out.append(line)
            continue
        mm = re.search(
            r"(\s*)(.*\b)?linalg\.matmul\s+ins\((%[\w]+),\s*(%[\w]+)\s*:\s*"
            r"(tensor<([^>]+)xf32>),\s*(tensor<([^>]+)xf32>)\)",
            line,
        )
        if not mm:
            out.append(line)
            continue
        indent = mm.group(1)
        lhs, rhs = mm.group(3), mm.group(4)
        lhs_shape, rhs_shape = mm.group(6), mm.group(8)

        if lhs in extf and rhs in extf:
            lhs_s, lhs_t = extf[lhs]
            rhs_s, rhs_t = extf[rhs]
            line = re.sub(
                r"ins\((%[\w]+),\s*(%[\w]+)\s*:\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)",
                f"ins({lhs_s}, {rhs_s} : {lhs_t}, {rhs_t})",
                line,
                count=1,
            )
            rewrites += 1
            out.append(line)
            continue

        # float32 model: truncf both operands in-place before matmul.
        trunc_id += 1
        lhs16 = f"%hexkl_lhs_f16_{trunc_id}"
        rhs16 = f"%hexkl_rhs_f16_{trunc_id}"
        out.append(_truncf_generic(lhs16, lhs, lhs_shape, indent))
        out.append(_truncf_generic(rhs16, rhs, rhs_shape, indent))
        line = re.sub(
            r"ins\((%[\w]+),\s*(%[\w]+)\s*:\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)",
            f"ins({lhs16}, {rhs16} : tensor<{lhs_shape}xf16>, "
            f"tensor<{rhs_shape}xf16>)",
            line,
            count=1,
        )
        rewrites += 1
        out.append(line)
    return "".join(out), rewrites


def get_encodings(tokenizer, *inputs, max_length: Optional[int] = None):
    if max_length is not None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Left-pad so the last position is a real token (top-5 uses logits[:, -1]).
        tokenizer.padding_side = "left"
        return tokenizer(
            *inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    return tokenizer(*inputs, return_tensors="pt")


def x86_execution(model, encoding):
    if isinstance(encoding, torch.Tensor):
        return model(encoding)
    if isinstance(encoding, dict):
        # Logits-only wrapper takes input_ids; full HF model takes **encoding.
        if "input_ids" in encoding and not any(
            k in encoding for k in ("attention_mask", "position_ids")
        ):
            try:
                return model(encoding["input_ids"])
            except TypeError:
                pass
        return model(**encoding)
    return model(encoding)


def hex_execution(module, func_name, inputs, options: dict = None, mlir_text: Optional[str] = None):
    linalg_filename = Path(__file__).parent / (str(func_name) + ".mlirbc")
    bytecode = module.operation.get_asm(binary=True)
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    launch_path = str(linalg_filename)
    if mlir_text is not None:
        patched = Path(__file__).parent / (str(func_name) + "_f16matmul.mlir")
        patched.write_text(mlir_text)
        launch_path = str(patched)

    hex_outputs = TorchMLIRHexagonLauncher().run_torch_mlir(
        launch_path, inputs, func_name, options=options
    )
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

def compare(hex_outputs, x86_outputs, tokenizer, atol=0.03, fail_on_mismatch: bool=False,
            require_exact_top5: bool = True,
            min_centered_cosine: Optional[float] = None):
    hexagon_logits = hex_outputs[0]
    t_hex, c_hex = get_top_5(hexagon_logits, tokenizer, "hexagon")

    if hasattr(x86_outputs, "logits"):
        x86_logits = x86_outputs.logits
    elif isinstance(x86_outputs, (tuple, list)):
        x86_logits = x86_outputs[0]
    else:
        x86_logits = x86_outputs
    t_x86, c_x86 = get_top_5(x86_logits, tokenizer, "x86")

    if min_centered_cosine is not None:
        # Full-depth GPT-2 accumulates backend numerical drift even when norms,
        # softmax and residuals remain f32.  Do not reduce the correctness gate
        # to top-1 alone: require finite logits, the same prediction, and a
        # minimum centered cosine over the complete last-token vocabulary.
        hex_last = hexagon_logits[0, -1, :].float()
        x86_last = x86_logits[0, -1, :].float()
        finite = bool(torch.isfinite(hex_last).all() and
                      torch.isfinite(x86_last).all())
        hex_centered = hex_last - hex_last.mean()
        x86_centered = x86_last - x86_last.mean()
        denom = torch.linalg.vector_norm(hex_centered) * torch.linalg.vector_norm(
            x86_centered
        )
        centered_cosine = float(
            torch.dot(hex_centered, x86_centered) / denom
        ) if float(denom) != 0.0 else float("nan")
        top5_overlap = len(set(t_hex) & set(t_x86))
        mean_abs_error = float(torch.mean(torch.abs(hex_last - x86_last)))
        print(
            "Full-model numerical gate: "
            f"finite={finite} top1_match={t_x86[0] == t_hex[0]} "
            f"top5_overlap={top5_overlap}/5 "
            f"centered_cosine={centered_cosine:.6f} "
            f"mean_abs_error={mean_abs_error:.6f} "
            f"(required cosine>={min_centered_cosine:.2f})"
        )
        tokens_match = (t_x86[0] == t_hex[0])
        confidences_match = (
            finite and centered_cosine >= min_centered_cosine
        )
    elif require_exact_top5:
        tokens_match = (t_x86 == t_hex)
        confidences_match = torch.allclose(torch.tensor(c_x86), torch.tensor(c_hex), atol)
    else:
        # HexKL/HMX accumulates in a different f16 path; require top-1 agreement.
        tokens_match = (t_x86[0] == t_hex[0])
        confidences_match = abs(c_x86[0] - c_hex[0]) <= max(atol, 1.0)

    if tokens_match and confidences_match:
        if min_centered_cosine is not None:
            print("Full-model numerical gate passed")
        else:
            print("The top5 tokens and their probabilities matched"
                  if require_exact_top5 else
                  "Top-1 token matched (HexKL numerical tolerance)")
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


def customize_gpt2_config(config):
    """Identity hook. Debug scripts may replace this to shrink n_layer."""
    return config


def freeze_gpt2_attn_bias_buffers(model: GPT2LMHeadModel, seq_len: Optional[int] = None):
    """Move causal-mask buffers off the FX/export ABI (same class as Qwen ConstRope).

    GPT-2 `Attention.bias` / `masked_bias` are `register_buffer`s.  torch-mlir
    FX promotes them to **runtime function args** (12× bool 1x1x1024x1024 +
    12× f16 scalars).  WrapperGenerator then emits a starter with only
    `(logits*, input_ids*)` while the compiled `_mlir_ciface_*` still reads
    the extra args from the Hexagon stack → TLBMISS Bad VA **0x0** at
    `_mlir_ciface_*+0x14` (exit 13).

    The old full `GPT2LMHeadModel` export “worked” because it also returned
    24 past_kv tensors; those extra output pointers filled the stack slots
    the iface blindly loads.  Logits-only (1 output) exposes the bug.

    Fix: demote buffers to plain tensor attributes so they fold as constants.
    Optionally slice bias to `seq_len` to shrink the constant footprint.
    """
    n = 0
    for block in model.transformer.h:
        attn = block.attn
        bias = attn._buffers.pop("bias", None)
        masked = attn._buffers.pop("masked_bias", None)
        if bias is None and masked is None:
            continue
        if bias is not None:
            if seq_len is not None and bias.dim() == 4:
                bias = bias[..., :seq_len, :seq_len].contiguous()
            # Plain attribute — not in _buffers / _parameters → FX constant.
            object.__setattr__(attn, "bias", bias)
            n += 1
        if masked is not None:
            object.__setattr__(attn, "masked_bias", masked)
            n += 1
    print(f"[Export] froze {n} attn bias/masked_bias buffers "
          f"(seq_len={seq_len}) — keep masks out of ciface ABI")


class GPT2LogitsWrapper(torch.nn.Module):
    """Export logits only (use_cache=False). Avoids 24 past_kv outputs on 12L.

    Requires `freeze_gpt2_attn_bias_buffers` first or DSP exit 13 (Bad VA 0x0).
    """

    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, use_cache=False).logits


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
    enable_omnifetch_dma_to_vtcm: bool = False,
    enable_omnifetch_weight_prepack: bool = False,
    enable_omnifetch_dual_thread_dae: bool = False,
    enable_omnifetch_inter_layer_prefetch: bool = False,
    enable_omnifetch_attention_hmx: bool = False,
    enable_hexkl_persistent_vtcm: bool = False,
    # Fixed sequence length for fair ablations (HexKL vs HVX vs OmniFetch).
    # None → legacy behaviour (short prompt; HexKL uses a hand-tuned 32-token string).
    seq_len: Optional[int] = None,
):

    model_name = "openai-community/gpt2"
    # Default short prompt (seq≈7).  HexKL needs M multiple of 32, so use a
    # fixed 32-token prompt (no padding / attention-mask) when HexKL is on.
    prompt = "What is nature of our existence?"
    prompt_hexkl = (
        "What is nature of our existence? answer the question carefully using "
        "concise and precise philosophical language today "
        "true true true true true true true true true true true true true true"
    )
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    config = GPT2Config.from_pretrained(model_name)
    config.use_cache = False
    config = customize_gpt2_config(config)
    print(f"[Config] n_layer={config.n_layer} n_embd={config.n_embd} "
          f"n_head={config.n_head} vocab={config.vocab_size}")

    # Keep the published full-depth model in f32.  A fully-f16 GPT-2 overflows
    # after roughly four layers on this path and produces device NaNs even in
    # the HVX baseline.  HexKL mode rewrites only matmul operands to f16 while
    # LayerNorm, softmax, residuals, and the reference remain numerically stable.
    model = GPT2LMHeadModel.from_pretrained(
        model_name, config=config, torch_dtype=torch.float32
    )
    model.eval()

    # Resolve seq_len early so causal-mask constants match the export shape.
    effective_seq = seq_len
    if effective_seq is None and enable_hexkl:
        # HexKL default prompt is length-checked below; freeze with 32 for masks.
        effective_seq = 32
    freeze_gpt2_attn_bias_buffers(model, seq_len=effective_seq)

    wrapped = GPT2LogitsWrapper(model)
    wrapped.eval()
    func_name = wrapped.__class__.__name__
    print(f"[Export] {func_name} dtype=float32 use_cache=False lm_head=full_seq")

    if seq_len is not None:
        if seq_len <= 0:
            raise ValueError(f"--seq-len must be positive, got {seq_len}")
        if enable_hexkl and seq_len % 32 != 0:
            raise ValueError(
                f"--seq-len={seq_len} is not a multiple of 32 (required for HexKL)"
            )
        # Build a fixed-length *content* sequence (no pad_token).  Left-padding
        # without an attention mask makes GPT-2 treat pads as real tokens and
        # breaks fair ablations vs the HexKL hand-tuned prompt.
        base = prompt_hexkl if seq_len >= 32 else prompt
        ids = tokenizer.encode(base, add_special_tokens=False)
        filler = tokenizer.encode(" true", add_special_tokens=False) or [
            tokenizer.eos_token_id
        ]
        while len(ids) < seq_len:
            ids.extend(filler)
        ids = ids[:seq_len]
        encoding = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
        }
        print(f"[Input] fair-compare seq_len={seq_len} (content-filled, "
              f"HexKL={enable_hexkl})")
    elif enable_hexkl:
        encoding = get_encodings(tokenizer, prompt_hexkl)
        got = encoding["input_ids"].shape[-1]
        print(f"[Input] HexKL prompt seq_len={got} (need multiple of 32)")
        assert got % 32 == 0, f"HexKL prompt length {got} not aligned to 32"
    else:
        encoding = get_encodings(tokenizer, prompt)
        print(f"[Input] default prompt seq_len={encoding['input_ids'].shape[-1]}")

    module = compile_to_linalg(wrapped, encoding["input_ids"])

    mlir_text = None
    if enable_hexkl:
        raw = module.operation.get_asm(binary=False)
        mlir_text, n = rewrite_matmul_inputs_to_f16(raw)
        print(f"[HexKL] Rewrote {n} linalg.matmul inputs to f16 for HMX")
        _patch_dsp_heap_256mb()
        # Match Qwen/Falcon HexKL: no vectorization (avoids Bad VA on forall).
        options = HexagonOptions(
            enableHexKL=True,
            enableVectorization=False,
            enableVTCMTiling=enable_vtcm_tiling,
            enableConvertToHexagonmem=True,
            enablePrefetch=enable_omnifetch_vdae,
            enableOmniFetchLayoutAware=enable_omnifetch_layout_aware,
            omniFetchLookahead=omnifetch_lookahead,
            enableOmniFetchVDAE=enable_omnifetch_vdae,
            enableOmniFetchAdaptive=enable_omnifetch_adaptive,
            enableOmniFetchDmaToVtcm=enable_omnifetch_dma_to_vtcm,
            enableOmniFetchWeightPrepack=enable_omnifetch_weight_prepack,
            enableOmniFetchDualThreadDae=enable_omnifetch_dual_thread_dae,
            enableOmniFetchInterLayerPrefetch=enable_omnifetch_inter_layer_prefetch,
            enableOmniFetchAttentionHmx=enable_omnifetch_attention_hmx,
            enableHexKLPersistentVtcm=enable_hexkl_persistent_vtcm,
        ).__dict__
    else:
        options = HexagonOptions().__dict__
        options["enableHexKL"] = False
        options["enableVTCMTiling"] = enable_vtcm_tiling
        options["enableConvertToHexagonmem"] = enable_convert_to_hexagonmem
        options["enablePrefetch"] = enable_omnifetch_vdae
        options["enableOmniFetchLayoutAware"] = enable_omnifetch_layout_aware
        options["omniFetchLookahead"] = omnifetch_lookahead
        options["enableOmniFetchVDAE"] = enable_omnifetch_vdae
        options["enableOmniFetchAdaptive"] = enable_omnifetch_adaptive
        options["enableOmniFetchDmaToVtcm"] = enable_omnifetch_dma_to_vtcm
        options["enableOmniFetchWeightPrepack"] = enable_omnifetch_weight_prepack
        options["enableOmniFetchDualThreadDae"] = enable_omnifetch_dual_thread_dae
        options["enableOmniFetchInterLayerPrefetch"] = (
            enable_omnifetch_inter_layer_prefetch
        )
        options["enableOmniFetchAttentionHmx"] = enable_omnifetch_attention_hmx
        options["enableHexKLPersistentVtcm"] = enable_hexkl_persistent_vtcm

    if enablelwp:
        options["enableLWP"] = True
    inputs = [encoding["input_ids"]]
    hex_outputs = hex_execution(
        module, func_name, inputs, options, mlir_text=mlir_text
    )
    x86_outputs = x86_execution(wrapped, encoding["input_ids"])

    compare(
        hex_outputs,
        x86_outputs,
        tokenizer,
        atol=0.5 if enable_hexkl else 0.03,
        fail_on_mismatch=True,
        require_exact_top5=seq_len is None and not enable_hexkl,
        # Fixed-length runs are the formal full-model ablations.  Their
        # compiler baseline is qualified with a whole-vocabulary numerical
        # gate; short tutorial runs retain the historical exact-top5 check.
        min_centered_cosine=0.80 if seq_len is not None else None,
    )
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
    parser.add_argument(
        "--enable-omnifetch-dma-to-vtcm",
        action="store_true",
        help="DMA-pack OmniFetch weight tiles into VTCM staging "
             "(default: DDR staging).",
    )
    parser.add_argument(
        "--enable-omnifetch-weight-prepack",
        action="store_true",
        help="Hoist HexKL RM->WH per column into VTCM (stream M rows). "
             "Win scales with ceil(M/32) — prefer --seq-len 128/256.",
    )
    parser.add_argument(
        "--enable-omnifetch-dual-thread-dae",
        action="store_true",
        help="Run deferred dma_wait+WH on a scout thread (default off).",
    )
    parser.add_argument(
        "--enable-omnifetch-inter-layer-prefetch",
        action="store_true",
        help="Allow PrefetchInsert on outer HexKL loops (next-layer weights).",
    )
    parser.add_argument(
        "--enable-omnifetch-attention-hmx",
        action="store_true",
        help="Pad attention-like (K==M/N==M) matmuls into HexKL.",
    )
    parser.add_argument(
        "--enable-hexkl-persistent-vtcm",
        action="store_true",
        help="Reuse one max-sized VTCM slab across HexKL matmuls in a function "
             "(default off; gain is noise-level on GPT-2).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Fixed sequence length for fair ablations (content-filled, no "
             "pad_token). Use the same value with/without --enable-hexkl. "
             "Must be a multiple of 32 when HexKL is enabled.",
    )
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
        enable_omnifetch_dma_to_vtcm=args.enable_omnifetch_dma_to_vtcm,
        enable_omnifetch_weight_prepack=args.enable_omnifetch_weight_prepack,
        enable_omnifetch_dual_thread_dae=args.enable_omnifetch_dual_thread_dae,
        enable_omnifetch_inter_layer_prefetch=args.enable_omnifetch_inter_layer_prefetch,
        enable_omnifetch_attention_hmx=args.enable_omnifetch_attention_hmx,
        enable_hexkl_persistent_vtcm=args.enable_hexkl_persistent_vtcm,
        seq_len=args.seq_len,
    )
