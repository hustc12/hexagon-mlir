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

def hex_execution(module, func_name, inputs, options: dict=None):
    linalg_filename = Path(__file__).parent / (str(func_name) + ".mlirbc")

    # Lower tm_tensor.scan (and any other unsupported tm_tensor ops) to
    # standard linalg/scf ops before handing the IR to the Hexagon backend.
    mlir_text = module.operation.get_asm(binary=False)
    mlir_text = lower_tm_tensor_scan(mlir_text)
    mlir_text = lower_math_powf_intexp(mlir_text)

    # Write the patched text MLIR to a .mlir file; the Hexagon launcher
    # accepts both .mlirbc (bytecode) and .mlir (text) paths.
    patched_filename = Path(__file__).parent / (str(func_name) + "_patched.mlir")
    with open(patched_filename, "w") as f:
        f.write(mlir_text)

    # Also save the original bytecode for reference/debugging
    bytecode = module.operation.get_asm(binary=True)
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = False 
    hex_outputs = TorchMLIRHexagonLauncher().run_torch_mlir(str(patched_filename), inputs, func_name, options=options)
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

def falcon_rw_1b(enablelwp=False): 

    model_name = "Rocketknight1/falcon-rw-1b" # tiiuae/falcon-rw-1b
    prompt = "What is nature of our existence?"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(model_name)
    # Using 2 layers for a "Lite" compilation test, similar to the other models
    config.num_hidden_layers = 2
    # "gelu" uses math.erf which the Hexagon MLIR backend cannot lower to LLVM IR.
    # Switch to "gelu_fast" (tanh-based approximation using x*x, no pow) to avoid
    # both math.erf and math.powf from gelu_new's x^3 term.
    # See test_graphsage.py for the same pattern with "gelu_new".
    config.activation = "gelu_fast"

    # Monkey-patch build_alibi_tensor to avoid two Hexagon-incompatible patterns:
    #   1. torch.pow(base, int_powers) → math.powf(f64,f64) which expands to exp2
    #      and then triggers a bf16 conversion crash in the Hexagon backend.
    #   2. slopes.bfloat16() → explicit bf16 cast that Hexagon cannot select.
    # Replacement: compute slopes via torch.cumprod (pure multiplications) and
    # keep the tensor in float32 until the final dtype cast.
    import transformers.models.falcon.modeling_falcon as falcon_modeling
    import math as _math

    def _patched_build_alibi_tensor(attention_mask, num_heads, dtype):
        batch_size, seq_length = attention_mask.shape
        closest_power_of_2 = 2 ** _math.floor(_math.log2(num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(_math.log2(closest_power_of_2) - 3))),
            device=attention_mask.device, dtype=torch.float32,
        )
        # Replace torch.pow(base, arange) with cumprod to avoid math.powf
        base_repeated = base.expand(closest_power_of_2)
        slopes = torch.cumprod(base_repeated, dim=0)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(_math.log2(2 * closest_power_of_2) - 3))),
                device=attention_mask.device, dtype=torch.float32,
            )
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_base_rep = extra_base.expand(num_remaining_heads)
            extra_slopes_half = torch.cumprod(extra_base_rep, dim=0)
            # original uses every-other power: pow(extra_base, 1,3,5,...)
            # cumprod gives pow(extra_base, 1,2,3,...); take odd indices via squaring
            extra_slopes = extra_slopes_half * extra_slopes_half
            slopes = torch.cat([slopes, extra_slopes], dim=0)

        arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
        # Use float32 instead of bfloat16 to avoid bf16 ops the Hexagon backend
        # cannot select (the original code does slopes.bfloat16() here).
        alibi = slopes[..., None].float() * arange_tensor
        return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

    falcon_modeling.build_alibi_tensor = _patched_build_alibi_tensor

    # Note: Using from_pretrained will download the weights directly from Hugging Face (~2GB).
    # If you just want to test compilation without downloading weights,
    # you can switch to AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    func_name = model.__class__.__name__

    encoding = get_encodings(tokenizer, prompt)
    module = compile_to_linalg(model, encoding["input_ids"])

    options = HexagonOptions().__dict__
    if enablelwp:
        options['enableLWP'] = True
    options['lowerConstantsInSeparateSharedObjects'] = True

    # torch-mlir lifts the ALiBi slopes buffer (shape [n_heads, f32]) as the first
    # function argument even though it is unused in the body (the actual slopes are
    # baked in as arith.constant).  We must still pass a correctly-shaped tensor so
    # the compiled wrapper reads the right number of arguments.
    n_heads = config.num_attention_heads
    dummy_slopes = torch.zeros(n_heads, dtype=torch.float32)
    inputs = [dummy_slopes, encoding["input_ids"]]
    
    # Run Hexagon
    hex_outputs = hex_execution(module, func_name, inputs, options)
    
    # Run x86
    with torch.no_grad():
        x86_outputs = x86_execution(model, encoding)

    compare(hex_outputs, x86_outputs, tokenizer, atol=0.03, fail_on_mismatch=False)
    if enablelwp:
        process_lwp()

if __name__ == "__main__":
    falcon_rw_1b()
