from typing import Optional
import sys, os
import re
import torch
import argparse
import subprocess
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher
from triton.backends.qcom_hexagon_backend import hexagon_launcher_base as _hlb

_QURT_HEAP_1GB = "unsigned int _QURT_MAX_HEAP_SIZE = 1073741824; // 1 GB Max Heap Size"
_QURT_HEAP_256MB = "unsigned int _QURT_MAX_HEAP_SIZE = 268435456;  // 256 MB Max Heap Size"


def _patch_dsp_heap_256mb():
    orig_init = _hlb.WrapperGeneratorStrings.__init__

    def _patched_init(self):
        orig_init(self)
        self.code_string = self.code_string.replace(_QURT_HEAP_1GB, _QURT_HEAP_256MB)

    _hlb.WrapperGeneratorStrings.__init__ = _patched_init


def rewrite_matmul_inputs_to_f16(ir: str) -> tuple[str, int]:
    """Rewrite extf(f16→f32) → linalg.matmul(f32) into matmul(f16,f16)→f32."""
    lines = ir.splitlines(keepends=True)
    extf = {}
    i = 0
    while i < len(lines):
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
                block += lines[j]
        if j >= len(lines):
            break
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
    for line in lines:
        mm = re.search(
            r"linalg\.matmul\s+ins\((%[\w]+),\s*(%[\w]+)\s*:\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)",
            line,
        )
        if (
            mm
            and "xf32" in mm.group(3)
            and "xf32" in mm.group(4)
            and mm.group(1) in extf
            and mm.group(2) in extf
        ):
            lhs_s, lhs_t = extf[mm.group(1)]
            rhs_s, rhs_t = extf[mm.group(2)]
            line = re.sub(
                r"ins\((%[\w]+),\s*(%[\w]+)\s*:\s*(tensor<[^>]+>),\s*(tensor<[^>]+>)\)",
                f"ins({lhs_s}, {rhs_s} : {lhs_t}, {rhs_t})",
                line,
                count=1,
            )
            rewrites += 1
        out.append(line)
    return "".join(out), rewrites


def rewrite_batch_matmul_to_matmul(ir: str) -> tuple[str, int]:
    """Collapse batch=1 linalg.batch_matmul into linalg.matmul for HexKL."""
    pat = re.compile(
        r"(?P<indent>\s*)(?P<res>%[\w]+)\s*=\s*linalg\.batch_matmul\s+"
        r"ins\((?P<a>%[\w]+),\s*(?P<b>%[\w]+)\s*:\s*"
        r"tensor<1x(?P<m>\d+)x(?P<k>\d+)x(?P<adt>\w+)>,\s*"
        r"tensor<1x(?P<k2>\d+)x(?P<n>\d+)x(?P<bdt>\w+)>\)\s*"
        r"outs\((?P<c>%[\w]+)\s*:\s*tensor<1x(?P<m2>\d+)x(?P<n2>\d+)x(?P<cdt>\w+)>\)"
        r"(?:\s*->\s*tensor<1x\d+x\d+x\w+>)?"
    )
    n = 0
    out_lines = []
    uid = 0
    for line in ir.splitlines(keepends=True):
        m = pat.search(line)
        if (
            not m
            or m.group("k") != m.group("k2")
            or m.group("m") != m.group("m2")
            or m.group("n") != m.group("n2")
        ):
            out_lines.append(line)
            continue
        M, K, N = int(m.group("m")), int(m.group("k")), int(m.group("n"))
        if (M % 32) != 0 or (K % 32) != 0 or (N % 32) != 0:
            out_lines.append(line)
            continue
        # Keep attention score / context matmuls off HexKL (HMX TLB risk).
        if K == M or N == M:
            out_lines.append(line)
            continue
        indent = m.group("indent")
        a, b, c = m.group("a"), m.group("b"), m.group("c")
        adt, bdt, cdt = m.group("adt"), m.group("bdt"), m.group("cdt")
        res = m.group("res")
        uid += 1
        a2, b2, c2, tmp = f"%_bm_a{uid}", f"%_bm_b{uid}", f"%_bm_c{uid}", f"%_bm_r{uid}"
        out_lines.append(
            f"{indent}{a2} = tensor.collapse_shape {a} [[0, 1], [2]] : "
            f"tensor<1x{M}x{K}x{adt}> into tensor<{M}x{K}x{adt}>\n"
            f"{indent}{b2} = tensor.collapse_shape {b} [[0, 1], [2]] : "
            f"tensor<1x{K}x{N}x{bdt}> into tensor<{K}x{N}x{bdt}>\n"
            f"{indent}{c2} = tensor.collapse_shape {c} [[0, 1], [2]] : "
            f"tensor<1x{M}x{N}x{cdt}> into tensor<{M}x{N}x{cdt}>\n"
            f"{indent}{tmp} = linalg.matmul ins({a2}, {b2} : "
            f"tensor<{M}x{K}x{adt}>, tensor<{K}x{N}x{bdt}>) "
            f"outs({c2} : tensor<{M}x{N}x{cdt}>) -> tensor<{M}x{N}x{cdt}>\n"
            f"{indent}{res} = tensor.expand_shape {tmp} [[0, 1], [2]] "
            f"output_shape [1, {M}, {N}] : "
            f"tensor<{M}x{N}x{cdt}> into tensor<1x{M}x{N}x{cdt}>\n"
        )
        n += 1
    return "".join(out_lines), n


def encode_fixed_seq(tokenizer, prompt: str, seq_len: int):
    """Content-filled fixed length (no pad_token) for fair / HexKL-aligned runs."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    filler = tokenizer.encode(" true", add_special_tokens=False) or [
        tokenizer.eos_token_id
    ]
    while len(ids) < seq_len:
        ids.extend(filler)
    ids = ids[:seq_len]
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float16)
    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
    return input_ids, attention_mask, position_ids


def hex_execution(
    module,
    func_name,
    inputs,
    options: dict = None,
    mlir_text: Optional[str] = None,
    iterations: int = 1,
):
    linalg_filename = Path(__file__).parent / (str(func_name) + ".mlirbc")

    text = mlir_text if mlir_text is not None else str(module)
    text = re.sub(r"[ \t]*cf\.assert[^\n]*\n", "", text)

    from torch_mlir._mlir_libs._mlir.ir import Module as _MLIRModule, Context as _MLIRContext
    from torch_mlir.dialects import torch as _torch_dialect  # noqa: F401

    with _MLIRContext() as _ctx:
        _ctx.allow_unregistered_dialects = True
        clean_module = _MLIRModule.parse(text, _ctx)
        bytecode = clean_module.operation.get_asm(binary=True)

    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    launch_path = str(linalg_filename)
    if mlir_text is not None:
        patched = Path(__file__).parent / (str(func_name) + "_f16matmul.mlir")
        patched.write_text(text)
        launch_path = str(patched)

    options = options or {}
    options["enableVTCMTiling"] = False
    if not options.get("enableHexKL"):
        options["enableConvertToHexagonmem"] = False
    return TorchMLIRHexagonLauncher().run_torch_mlir(
        launch_path, inputs, func_name, iterations=iterations, options=options
    )


def get_top_5(logits: torch.Tensor, tokenizer, run_type: str):
    print(f"\n-------Printing the top5 probable tokens for {run_type}--------\n")
    top_k = 5

    if logits.ndim != 3:
        raise ValueError(f"Expected logits to be a 3D tensor, but got shape {logits.shape}")

    last_row_logits = logits[0, -1, :]
    top_values, top_indices = torch.topk(last_row_logits, top_k)
    top_confidences = top_values.tolist()

    top_tokens = []
    for idx in top_indices:
        try:
            top_tokens.append(tokenizer.decode([idx]))
        except Exception:
            top_tokens.append(f"<id:{idx}>")

    for token, confidence in zip(top_tokens, top_confidences):
        print(f"Token: {[token]}, Confidence: {confidence:.4f}")
    print("---------------------------------------------------\n")
    return top_tokens, top_confidences


def compare(
    hex_outputs,
    x86_outputs,
    tokenizer,
    atol=0.03,
    fail_on_mismatch: bool = False,
    require_exact_top5: bool = True,
):
    hexagon_logits = hex_outputs[0]
    t_hex, c_hex = get_top_5(hexagon_logits, tokenizer, "hexagon")

    if hasattr(x86_outputs, "logits"):
        x86_logits = x86_outputs.logits
    elif isinstance(x86_outputs, torch.Tensor):
        x86_logits = x86_outputs
    else:
        x86_logits = x86_outputs[0]
    t_x86, c_x86 = get_top_5(x86_logits, tokenizer, "x86")

    if require_exact_top5:
        tokens_match = t_x86 == t_hex
        confidences_match = torch.allclose(
            torch.tensor(c_x86), torch.tensor(c_hex), atol
        )
    else:
        tokens_match = t_x86[0] == t_hex[0]
        confidences_match = abs(c_x86[0] - c_hex[0]) <= max(atol, 1.0)

    if tokens_match and confidences_match:
        print(
            "The top5 tokens and their probabilities matched"
            if require_exact_top5
            else "Top-1 token matched (HexKL numerical tolerance)"
        )
    else:
        print("Hexagon and CPU results do not match")
        assert not fail_on_mismatch, (
            "Correctness issue: Hexagon vs x86 results do not match"
        )


def compile_to_linalg(model, input, dump_to_file=None, debug=False) -> str:
    if isinstance(input, torch.Tensor):
        input = (input,)

    import torch._decomp as _decomp

    decomp_table = _decomp.get_decompositions(
        [
            torch.ops.aten.pow.Tensor_Scalar,
            torch.ops.aten.pow.Scalar,
            torch.ops.aten.pow.Tensor_Tensor,
        ]
    )

    linalg = fx.export_and_import(
        model,
        *input,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug,
        decomposition_table=decomp_table,
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
                "/tmp/initial-linalg.mlir",
            ],
            check=True,
            capture_output=True,
            text=True,
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


def customize_model_config(config):
    """Identity hook. Debug scripts may replace this to shrink topology."""
    return config


def load_tinyllama_model(model_name, config):
    """Load published weights. Debug scripts may replace with from_config."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )


def tinyllama_1_1b(
    enable_hexkl: bool = False,
    enable_hvx_vector: bool = False,
    enable_omnifetch_activation_multicast: bool = False,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
    enable_omnifetch_items_1_7: bool = False,
    seq_len: Optional[int] = None,
    device_iterations: int = 1,
):
    _patch_dsp_heap_256mb()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        input_ids, attention_mask, position_ids = encode_fixed_seq(
            tokenizer, prompt, seq_len
        )
        print(
            f"[Input] fair-compare seq_len={seq_len} "
            f"(content-filled, HexKL={enable_hexkl})"
        )
    else:
        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to(torch.int64)
        attention_mask = torch.ones_like(input_ids, dtype=torch.float16)
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long).unsqueeze(
            0
        )
        print(f"[Input] default prompt seq_len={input_ids.shape[-1]}")

    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    config = customize_model_config(config)
    print(
        f"[Config] layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"vocab={config.vocab_size} intermediate={config.intermediate_size} "
        f"heads={config.num_attention_heads}/{config.num_key_value_heads}"
    )

    model = load_tinyllama_model(model_name, config)
    model.eval()

    class LlamaWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, input_ids, attention_mask, position_ids):
            input_ids = torch.clamp(input_ids, 0, self.model.config.vocab_size - 1)
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits

    wrapped_model = LlamaWrapper(model)
    wrapped_model.eval()
    func_name = wrapped_model.__class__.__name__

    # Precompute RoPE cos/sin as closure constants (not buffers / inv_freq ABI).
    rotary_emb = wrapped_model.model.model.rotary_emb
    with torch.no_grad():
        _inv_freq = rotary_emb.inv_freq.float()
        _pos = position_ids[0].float()
        _freqs = torch.outer(_pos, _inv_freq)
        _emb = torch.cat((_freqs, _freqs), dim=-1)
        _scale = getattr(rotary_emb, "attention_scaling", 1.0)
        _cos_cache = (_emb.cos() * _scale).to(torch.float16).unsqueeze(0)
        _sin_cache = (_emb.sin() * _scale).to(torch.float16).unsqueeze(0)

    class _ConstRope(torch.nn.Module):
        def forward(self, x, position_ids):
            return _cos_cache.to(dtype=x.dtype), _sin_cache.to(dtype=x.dtype)

    wrapped_model.model.model.rotary_emb = _ConstRope()

    module = compile_to_linalg(
        wrapped_model, (input_ids, attention_mask, position_ids)
    )
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )

    mlir_text = None
    if enable_hexkl:
        ir2, n_bm = rewrite_batch_matmul_to_matmul(ir)
        ir2, n_f16 = rewrite_matmul_inputs_to_f16(ir2)
        mlir_text = ir2
        print(
            f"[HexKL] batch_matmul→matmul={n_bm}, f16-input rewrite={n_f16}"
        )
        if n_bm == 0 and n_f16 == 0:
            print("[HexKL] WARNING: no matmul rewrite — HexKL may not fire")

    options = HexagonOptions().__dict__
    options["enableLWP"] = False
    options["lowerConstantsInSeparateSharedObjects"] = True
    options["enableVectorization"] = bool(enable_hvx_vector)
    options["enableHexKL"] = bool(enable_hexkl)
    options["enableConvertToHexagonmem"] = bool(enable_hexkl)
    cumulative = bool(enable_omnifetch_items_1_7)
    options["enablePrefetch"] = bool(enable_omnifetch_vdae or cumulative)
    options["enableOmniFetchLayoutAware"] = bool(enable_omnifetch_layout_aware)
    options["omniFetchLookahead"] = int(omnifetch_lookahead)
    options["enableOmniFetchVDAE"] = bool(enable_omnifetch_vdae or cumulative)
    options["enableOmniFetchAdaptive"] = bool(enable_omnifetch_adaptive)
    options["enableOmniFetchPersistentWhCache"] = cumulative
    options["enableOmniFetchTwoDimPipeline"] = cumulative
    options["enableOmniFetchVtcmColoring"] = cumulative
    options["enableOmniFetchKvCachePrefetch"] = cumulative
    options["enableOmniFetchActivationMulticast"] = bool(
        enable_omnifetch_activation_multicast
    )

    inputs = [input_ids, attention_mask, position_ids]

    hex_outputs = hex_execution(
        module,
        func_name,
        inputs,
        options,
        mlir_text=mlir_text,
        iterations=device_iterations,
    )
    print("Successfully ran TinyLlama on Hexagon DSP!")

    x86_logits = x86_execution(
        wrapped_model, [input_ids, attention_mask, position_ids]
    )
    compare(
        hex_outputs,
        x86_logits,
        tokenizer,
        fail_on_mismatch=True,
        require_exact_top5=not enable_hexkl,
    )
    hex_logits = hex_outputs[0]
    max_abs = (hex_logits.float() - x86_logits.float()).abs().max().item()
    print(f"[Compare] max_abs_diff(logits)={max_abs:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TinyLlama-1.1B Hexagon smoke (optional HexKL/OmniFetch)."
    )
    parser.add_argument(
        "--enable-hexkl",
        action="store_true",
        help="Enable HexKL + hexagonmem (vectorization stays off).",
    )
    parser.add_argument("--enable-hvx-vector", action="store_true")
    parser.add_argument("--enable-omnifetch-activation-multicast",
                        action="store_true")
    parser.add_argument(
        "--enable-omnifetch-vdae",
        action="store_true",
        help="Enable Omni-Fetch Prefetch + V-DAE.",
    )
    parser.add_argument(
        "--disable-layout-aware",
        action="store_true",
        help="Disable layout-aware prefetch (L2Hint / linear only).",
    )
    parser.add_argument("--omnifetch-lookahead", type=int, default=2)
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")
    parser.add_argument("--enable-omnifetch-items-1-7", action="store_true")
    parser.add_argument("--device-iterations", type=int, default=1)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Fixed content-filled seq length (HexKL defaults to 32).",
    )
    args = parser.parse_args()
    tinyllama_1_1b(
        enable_hexkl=args.enable_hexkl,
        enable_hvx_vector=args.enable_hvx_vector,
        enable_omnifetch_activation_multicast=(
            args.enable_omnifetch_activation_multicast
        ),
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        enable_omnifetch_items_1_7=args.enable_omnifetch_items_1_7,
        seq_len=args.seq_len,
        device_iterations=args.device_iterations,
    )
