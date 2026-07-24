"""Shared Phase-4 HexKL / OmniFetch harness helpers for benchmark_models."""
from __future__ import annotations

from typing import Optional
import re
from pathlib import Path

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import (
    TorchMLIRHexagonLauncher,
)
from triton.backends.qcom_hexagon_backend import hexagon_launcher_base as _hlb

_QURT_HEAP_1GB = "unsigned int _QURT_MAX_HEAP_SIZE = 1073741824; // 1 GB Max Heap Size"
_QURT_HEAP_256MB = "unsigned int _QURT_MAX_HEAP_SIZE = 268435456;  // 256 MB Max Heap Size"


def patch_dsp_heap_256mb():
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
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    filler = tokenizer.encode(" true", add_special_tokens=False) or [
        tokenizer.eos_token_id or 0
    ]
    while len(ids) < seq_len:
        ids.extend(filler)
    ids = ids[:seq_len]
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float16)
    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
    return input_ids, attention_mask, position_ids


def compile_to_linalg(model, input, dump_to_file=None, debug=False, decomp_pow=True):
    if isinstance(input, torch.Tensor):
        input = (input,)
    kwargs = dict(
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug,
    )
    if decomp_pow:
        import torch._decomp as _decomp

        ops = [
            torch.ops.aten.pow.Tensor_Scalar,
            torch.ops.aten.pow.Scalar,
            torch.ops.aten.pow.Tensor_Tensor,
        ]
        # gelu_new / some transformers paths emit prims.pow.
        for name in ("pow", "pow_tensor_scalar", "pow_tensor_tensor"):
            op = getattr(torch.ops.prims, name, None)
            if op is not None:
                ops.append(op)
        kwargs["decomposition_table"] = _decomp.get_decompositions(ops)
    linalg = fx.export_and_import(model, *input, **kwargs)
    if dump_to_file:
        with open(dump_to_file, "w") as f:
            f.write(str(linalg))
    return linalg


def hex_execution(
    module,
    func_name,
    inputs,
    options: dict = None,
    mlir_text: Optional[str] = None,
    out_dir: Optional[Path] = None,
):
    out_dir = out_dir or Path(__file__).parent
    linalg_filename = out_dir / (str(func_name) + ".mlirbc")

    if mlir_text is not None:
        text = re.sub(r"[ \t]*cf\.assert[^\n]*\n", "", mlir_text)
        from torch_mlir._mlir_libs._mlir.ir import (
            Module as _MLIRModule,
            Context as _MLIRContext,
        )
        from torch_mlir.dialects import torch as _torch_dialect  # noqa: F401

        with _MLIRContext() as _ctx:
            _ctx.allow_unregistered_dialects = True
            clean_module = _MLIRModule.parse(text, _ctx)
            bytecode = clean_module.operation.get_asm(binary=True)
        patched = out_dir / (str(func_name) + "_f16matmul.mlir")
        patched.write_text(text)
        launch_path = str(patched)
    else:
        # Keep original module bytecode — re-parsing str(module) drops dialects
        # like tm_tensor (BERT/GraphSAGE attention) and fails MLIR parse.
        bytecode = module.operation.get_asm(binary=True)
        launch_path = str(linalg_filename)

    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    options = options or {}
    options["enableVTCMTiling"] = False
    if not options.get("enableHexKL"):
        options["enableConvertToHexagonmem"] = False
    return TorchMLIRHexagonLauncher().run_torch_mlir(
        launch_path, inputs, func_name, options=options
    )


def apply_hexkl_ir_rewrites(ir: str) -> tuple[str, int, int]:
    ir2, n_bm = rewrite_batch_matmul_to_matmul(ir)
    ir2, n_f16 = rewrite_matmul_inputs_to_f16(ir2)
    return ir2, n_bm, n_f16


def hexagon_options_phase4(
    enable_hexkl: bool,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
    lower_constants_separate: bool = True,
):
    from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions

    options = HexagonOptions().__dict__
    options["enableLWP"] = False
    options["lowerConstantsInSeparateSharedObjects"] = bool(lower_constants_separate)
    options["enableVectorization"] = False
    options["enableHexKL"] = bool(enable_hexkl)
    options["enableConvertToHexagonmem"] = bool(enable_hexkl)
    options["enablePrefetch"] = bool(enable_omnifetch_vdae)
    options["enableOmniFetchLayoutAware"] = bool(enable_omnifetch_layout_aware)
    options["omniFetchLookahead"] = int(omnifetch_lookahead)
    options["enableOmniFetchVDAE"] = bool(enable_omnifetch_vdae)
    options["enableOmniFetchAdaptive"] = bool(enable_omnifetch_adaptive)
    return options


def add_phase4_args(parser):
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-omnifetch-vdae", action="store_true")
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--omnifetch-lookahead", type=int, default=2)
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None)
    return parser


def phase4_kwargs_from_args(args):
    return dict(
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        seq_len=getattr(args, "seq_len", None),
    )
