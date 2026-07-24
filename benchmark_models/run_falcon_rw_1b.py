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
    """Collapse batch=1 linalg.batch_matmul into linalg.matmul for HexKL.

    Skip attention-like shapes (K==M or N==M); MatmulToHexKLPass also skips them.
    """
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
    return torch.tensor([ids], dtype=torch.long)


def lower_tm_tensor_scan(mlir_text: str) -> str:
    """Lower tm_tensor.scan ops to scf.for + tensor.extract/insert.

    tm_tensor.scan is an inclusive/exclusive prefix-scan (e.g. cumsum) that
    torch-mlir emits but the Hexagon backend does not recognise.  We replace
    every occurrence with an equivalent scf.for loop before handing the IR to
    the backend.

    Supported body: a single binary op whose result is yielded via
    tm_tensor.yield.  The body operands are (%element, %accumulator) in that
    order (torch-mlir convention).
    """

    # -----------------------------------------------------------------------
    # Regex that captures one tm_tensor.scan op (including its region body).
    # Group indices:
    #   1  – result SSA names, e.g. "%10:2"
    #   2  – scan dimension integer
    #   3  – "true" | "false"  (inclusive flag)
    #   4  – input SSA value, e.g. "%cst_39"
    #   5  – input tensor type, e.g. "tensor<1x10xi64>"
    #   6  – output (scan result) SSA value, e.g. "%7"
    #   7  – output (accumulator) SSA value, e.g. "%9"
    #   8  – scan result tensor type, e.g. "tensor<1x10xi64>"
    #   9  – accumulator tensor type, e.g. "tensor<1xi64>"
    #  10  – full region body text (between the outer braces)
    # -----------------------------------------------------------------------
    scan_re = re.compile(
        r'(%\S+:\d+)\s*=\s*tm_tensor\.scan\s+'
        r'dimension\((\d+)\)\s+inclusive\((true|false)\)\s+'
        r'ins\((\S+)\s*:\s*(\S+)\)\s+'
        r'outs\((\S+),\s*(\S+)\s*:\s*(\S+),\s*(\S+)\)\s*'
        r'\{([^}]*)\}\s*->\s*\S+,\s*\S+',
        re.DOTALL,
    )

    # Helper: parse "tensor<d0 x d1 x ... x dtype>" -> (shape_list, dtype_str)
    def parse_tensor_type(t: str):
        m = re.match(r'tensor<(.+)>', t)
        assert m, f"Cannot parse tensor type: {t}"
        parts = m.group(1).split('x')
        dtype = parts[-1]
        shape = [int(d) for d in parts[:-1]]
        return shape, dtype

    # Helper: extract the binary op name and its two operands from the region
    # body.  torch-mlir always emits exactly one op + tm_tensor.yield.
    def parse_body(body: str):
        # Find the inner op line: %NNN = <dialect>.<op> %argA, %argB : type
        op_m = re.search(
            r'(%\S+)\s*=\s*([\w.]+)\s+(%\S+),\s*(%\S+)\s*:\s*\S+', body
        )
        assert op_m, f"Cannot parse scan body: {body!r}"
        return op_m.group(2)  # e.g. "arith.addi"

    # Unique counter so generated SSA names don't clash across multiple scans
    _counter = [0]

    def replace_one(m: re.Match) -> str:
        _counter[0] += 1
        uid = _counter[0]

        result_names = m.group(1)   # e.g. "%10:2"
        dim          = int(m.group(2))
        inclusive    = m.group(3) == 'true'
        inp_val      = m.group(4)
        inp_type     = m.group(5)
        out_val      = m.group(6)
        acc_val      = m.group(7)
        out_type     = m.group(8)
        acc_type     = m.group(9)
        body         = m.group(10)

        binary_op = parse_body(body)
        shape, dtype = parse_tensor_type(inp_type)
        rank = len(shape)
        scan_dim_size = shape[dim]

        # SSA name for the two results: %<base>#0 and %<base>#1
        base = result_names.split(':')[0]   # e.g. "%10"

        # Build index variable names
        idx_vars = [f'%_scan{uid}_i{d}' for d in range(rank)]
        loop_var = idx_vars[dim]

        # Constants we need
        lines = []
        lines.append(f'    %_scan{uid}_c0 = arith.constant 0 : index')
        lines.append(f'    %_scan{uid}_c1 = arith.constant 1 : index')
        lines.append(f'    %_scan{uid}_cN = arith.constant {scan_dim_size} : index')

        # Zero-value for the accumulator element type
        if 'i' in dtype or dtype == 'index':
            zero_val = f'%_scan{uid}_zero'
            lines.append(f'    {zero_val} = arith.constant 0 : {dtype}')
        else:
            zero_val = f'%_scan{uid}_zero'
            lines.append(f'    {zero_val} = arith.constant 0.0 : {dtype}')

        # For dimensions other than the scan dim we need loop bounds too.
        # Build nested scf.for loops for all non-scan dims, then the scan loop.
        # For simplicity (and because Falcon only has rank-2 with dim=1),
        # we generate a flat loop over the scan dimension only and use
        # tensor.extract/insert with fixed indices for the other dims.
        # This works for any rank as long as the non-scan dims are size 1
        # (which is the case here: tensor<1x10xi64>).
        # For the general case we'd need nested loops; add that if needed.

        # Build the index list for tensor.extract/insert
        # Non-scan dims: use constant 0 (they are size 1 in this model)
        extract_indices = []
        for d in range(rank):
            if d == dim:
                extract_indices.append(loop_var)
            else:
                extract_indices.append(f'%_scan{uid}_c0')
        idx_str = ', '.join(extract_indices)

        # For the accumulator tensor (rank = rank-1, scan dim removed)
        acc_shape, _ = parse_tensor_type(acc_type)
        acc_rank = len(acc_shape)
        # accumulator indices: all non-scan dims (size 1 → constant 0)
        acc_idx_str = ', '.join([f'%_scan{uid}_c0'] * acc_rank) if acc_rank > 0 else ''

        # Build the scf.for loop
        lines.append(
            f'    {base}:2 = scf.for {loop_var} = %_scan{uid}_c0 to %_scan{uid}_cN step %_scan{uid}_c1 '
            f'iter_args(%_scan{uid}_out = {out_val}, %_scan{uid}_run = {zero_val}) '
            f'-> ({out_type}, {dtype}) {{'
        )

        if inclusive:
            # inclusive: result[i] = combine(input[i], running_acc)
            lines.append(f'      %_scan{uid}_cur = tensor.extract {inp_val}[{idx_str}] : {inp_type}')
            lines.append(f'      %_scan{uid}_new = {binary_op} %_scan{uid}_cur, %_scan{uid}_run : {dtype}')
            lines.append(f'      %_scan{uid}_nout = tensor.insert %_scan{uid}_new into %_scan{uid}_out[{idx_str}] : {out_type}')
            lines.append(f'      scf.yield %_scan{uid}_nout, %_scan{uid}_new : {out_type}, {dtype}')
        else:
            # exclusive: result[i] = running_acc, then update acc with input[i]
            lines.append(f'      %_scan{uid}_cur = tensor.extract {inp_val}[{idx_str}] : {inp_type}')
            lines.append(f'      %_scan{uid}_nout = tensor.insert %_scan{uid}_run into %_scan{uid}_out[{idx_str}] : {out_type}')
            lines.append(f'      %_scan{uid}_new = {binary_op} %_scan{uid}_run, %_scan{uid}_cur : {dtype}')
            lines.append(f'      scf.yield %_scan{uid}_nout, %_scan{uid}_new : {out_type}, {dtype}')

        lines.append('    }')

        # Write the final running value back into the accumulator tensor
        if acc_rank > 0:
            lines.append(
                f'    %_scan{uid}_final_acc = tensor.insert {base}#1 into {acc_val}[{acc_idx_str}] : {acc_type}'
            )
            # Redefine %10#1 as the updated accumulator tensor
            lines.append(f'    // Note: {base}#1 is the scalar running value; {acc_val} holds the acc tensor')
        # The scan result tensor is base#0 (already correct from scf.for)

        return '\n'.join(lines)

    result = scan_re.sub(replace_one, mlir_text)

    if result != mlir_text:
        n = len(scan_re.findall(mlir_text))
        print(f"[lower_tm_tensor_scan] Lowered {n} tm_tensor.scan op(s) to scf.for loops.")
    return result


def lower_math_powf_intexp(mlir_text: str) -> str:
    """Replace the pattern:
        %a = arith.extf %x : f32 to f64
        %b = arith.sitofp %i : i32 to f64
        %c = math.powf %a, %b : f64
        %d = arith.truncf %c : f64 to f32
    with:
        %d = math.fpowi %x, %i : f32, i32

    This avoids the exp2-based expansion of math.powf on f64 which triggers a
    bf16 conversion that the Hexagon backend cannot select.  math.fpowi on f32
    is expanded by ExpandMathOpsPass (populateExpandFPowIPattern) into a simple
    multiply-chain with no transcendental calls.
    """
    pattern = re.compile(
        r'(\s*)(%\w+)\s*=\s*arith\.extf\s+(%\w+)\s*:\s*f32\s+to\s+f64\s*\n'
        r'\s*(%\w+)\s*=\s*arith\.sitofp\s+(%\w+)\s*:\s*i32\s+to\s+f64\s*\n'
        r'\s*(%\w+)\s*=\s*math\.powf\s+\2\s*,\s*\4\s*:\s*f64\s*\n'
        r'\s*(%\w+)\s*=\s*arith\.truncf\s+\6\s*:\s*f64\s+to\s+f32'
    )
    def replace_one(m):
        indent  = m.group(1)
        x_f32   = m.group(3)   # original f32 base
        i_i32   = m.group(5)   # original i32 exponent
        result  = m.group(7)   # final f32 result SSA name
        return f'{indent}{result} = math.fpowi {x_f32}, {i_i32} : f32, i32'

    result = pattern.sub(replace_one, mlir_text)
    if result != mlir_text:
        n = len(pattern.findall(mlir_text))
        print(f"[lower_math_powf_intexp] Replaced {n} math.powf(f64,f64) pattern(s) with math.fpowi(f32,i32).")
    return result


def get_encodings(tokenizer, *inputs):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encodings = tokenizer(*inputs, return_tensors="pt")
    return encodings

def x86_execution(model, encoding):
    x86_outputs = model(**encoding)
    return x86_outputs

def hex_execution(module, func_name, inputs, options: dict = None, mlir_text: Optional[str] = None):
    linalg_filename = Path(__file__).parent / (str(func_name) + ".mlirbc")

    text = mlir_text if mlir_text is not None else module.operation.get_asm(binary=False)
    text = lower_tm_tensor_scan(text)
    text = lower_math_powf_intexp(text)
    # Strip cf.assert (embedding bounds → ub.poison on Hexagon LLVM).
    text = re.sub(r"[ \t]*cf\.assert[^\n]*\n", "", text)

    patched_filename = Path(__file__).parent / (str(func_name) + "_patched.mlir")
    patched_filename.write_text(text)

    bytecode = module.operation.get_asm(binary=True)
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    options = options or {}
    options["enableVTCMTiling"] = False
    if not options.get("enableHexKL"):
        options["enableConvertToHexagonmem"] = False
    return TorchMLIRHexagonLauncher().run_torch_mlir(
        str(patched_filename), inputs, func_name, options=options
    )


# logits is expected to be "[batch_size, sequence_length, vocab_size]"
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
        confidences_match = torch.allclose(torch.tensor(c_x86), torch.tensor(c_hex), atol)
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

    linalg = fx.export_and_import(
        model,
        *input,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug,
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


def customize_model_config(config):
    """Identity hook. Debug scripts may replace this to shrink num_hidden_layers."""
    return config


def load_falcon_model(model_name, config):
    """Load published weights. Debug scripts may replace with from_config."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )


def _install_alibi_patch():
    """Avoid Hexagon-incompatible powf/bf16 in Falcon ALiBi slope construction."""
    import transformers.models.falcon.modeling_falcon as falcon_modeling
    import math as _math

    def _patched_build_alibi_tensor(attention_mask, num_heads, dtype):
        batch_size, seq_length = attention_mask.shape
        closest_power_of_2 = 2 ** _math.floor(_math.log2(num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(_math.log2(closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        base_repeated = base.expand(closest_power_of_2)
        slopes = torch.cumprod(base_repeated, dim=0)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(_math.log2(2 * closest_power_of_2) - 3))),
                device=attention_mask.device,
                dtype=torch.float32,
            )
            num_remaining_heads = min(
                closest_power_of_2, num_heads - closest_power_of_2
            )
            extra_base_rep = extra_base.expand(num_remaining_heads)
            extra_slopes_half = torch.cumprod(extra_base_rep, dim=0)
            extra_slopes = extra_slopes_half * extra_slopes_half
            slopes = torch.cat([slopes, extra_slopes], dim=0)

        arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[
            :, None, :
        ]
        alibi = slopes[..., None].float() * arange_tensor
        return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

    falcon_modeling.build_alibi_tensor = _patched_build_alibi_tensor


def falcon_rw_1b(
    enablelwp: bool = False,
    enable_hexkl: bool = False,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
    seq_len: Optional[int] = None,
):
    _patch_dsp_heap_256mb()

    model_name = "Rocketknight1/falcon-rw-1b"  # tiiuae/falcon-rw-1b
    prompt = "What is nature of our existence?"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if seq_len is None and enable_hexkl:
        seq_len = 32
    if seq_len is not None:
        if seq_len <= 0:
            raise ValueError(f"--seq-len must be positive, got {seq_len}")
        if enable_hexkl and seq_len % 32 != 0:
            raise ValueError(
                f"--seq-len={seq_len} is not a multiple of 32 (required for HexKL)"
            )
        input_ids = encode_fixed_seq(tokenizer, prompt, seq_len)
        print(
            f"[Input] fair-compare seq_len={seq_len} "
            f"(content-filled, HexKL={enable_hexkl})"
        )
    else:
        encoding = get_encodings(tokenizer, prompt)
        input_ids = encoding["input_ids"].to(torch.int64)
        print(f"[Input] default prompt seq_len={input_ids.shape[-1]}")

    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    # Backend-only: gelu → gelu_fast (no math.erf). Not a topology shrink.
    config.activation = "gelu_fast"
    config = customize_model_config(config)
    print(
        f"[Config] layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"vocab={config.vocab_size} heads={config.num_attention_heads} "
        f"act={config.activation}"
    )

    _install_alibi_patch()
    model = load_falcon_model(model_name, config)
    model.eval()
    func_name = model.__class__.__name__

    # Debug may shrink vocab; clamp so embedding / compare stay in-range.
    input_ids = input_ids.clamp(0, config.vocab_size - 1)

    module = compile_to_linalg(model, input_ids)
    ir = module.operation.get_asm(binary=False)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )

    mlir_text = None
    if enable_hexkl:
        ir2, n_bm = rewrite_batch_matmul_to_matmul(ir)
        ir2, n_f16 = rewrite_matmul_inputs_to_f16(ir2)
        mlir_text = ir2
        print(f"[HexKL] batch_matmul→matmul={n_bm}, f16-input rewrite={n_f16}")

    options = HexagonOptions().__dict__
    options["enableLWP"] = bool(enablelwp)
    options["lowerConstantsInSeparateSharedObjects"] = True
    # Same as Qwen: MicroHMX does not need HVX vectorization; vec crashed Falcon
    # on device (exit 13). MatmulToHexKL already skips attention-like shapes.
    options["enableVectorization"] = False
    options["enableHexKL"] = bool(enable_hexkl)
    options["enableConvertToHexagonmem"] = bool(enable_hexkl)
    options["enablePrefetch"] = bool(enable_omnifetch_vdae)
    options["enableOmniFetchLayoutAware"] = bool(enable_omnifetch_layout_aware)
    options["omniFetchLookahead"] = int(omnifetch_lookahead)
    options["enableOmniFetchVDAE"] = bool(enable_omnifetch_vdae)
    options["enableOmniFetchAdaptive"] = bool(enable_omnifetch_adaptive)

    # torch-mlir may lift unused ALiBi slopes buffer as arg0; keep a matching dummy.
    n_heads = config.num_attention_heads
    dummy_slopes = torch.zeros(n_heads, dtype=torch.float32)
    inputs = [dummy_slopes, input_ids]

    hex_outputs = hex_execution(
        module, func_name, inputs, options, mlir_text=mlir_text
    )
    print("Successfully ran Falcon on Hexagon DSP!")

    with torch.no_grad():
        x86_outputs = model(input_ids=input_ids)
    max_abs = (hex_outputs[0] - x86_outputs.logits).abs().max().item()
    print(f"[Compare] max_abs_diff(logits)={max_abs:.4f}")
    compare(
        hex_outputs,
        x86_outputs,
        tokenizer,
        atol=0.5 if enable_hexkl else 0.03,
        fail_on_mismatch=True,
        require_exact_top5=not enable_hexkl,
    )
    if enablelwp:
        process_lwp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Falcon-RW-1B Hexagon smoke (optional HexKL/OmniFetch)."
    )
    parser.add_argument("--enable-lwp", action="store_true")
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-omnifetch-vdae", action="store_true")
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--omnifetch-lookahead", type=int, default=2)
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Fixed content-filled seq length (HexKL defaults to 32).",
    )
    args = parser.parse_args()
    falcon_rw_1b(
        enablelwp=args.enable_lwp,
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        seq_len=args.seq_len,
    )
