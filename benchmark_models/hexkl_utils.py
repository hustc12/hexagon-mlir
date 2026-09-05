"""Shared Phase-4 HexKL / Alps harness helpers for benchmark_models."""
from __future__ import annotations

from typing import Optional
import argparse
import os
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


def patch_dsp_heap_mb(heap_size_mb: int):
    if heap_size_mb <= 0 or heap_size_mb >= 1024:
        raise ValueError("DSP heap must be between 1 and 1023 MiB")
    orig_init = _hlb.WrapperGeneratorStrings.__init__
    replacement = (
        "unsigned int _QURT_MAX_HEAP_SIZE = "
        f"{heap_size_mb * 1024 * 1024};  // {heap_size_mb} MB Max Heap Size"
    )

    def _patched_init(self):
        orig_init(self)
        self.code_string = self.code_string.replace(_QURT_HEAP_1GB, replacement)

    _hlb.WrapperGeneratorStrings.__init__ = _patched_init


def patch_dsp_heap_256mb():
    patch_dsp_heap_mb(256)


def patch_full_model_dsp_heap(default_mb: int = 512):
    """Patch the full-model DSP heap, with an auditable environment override."""
    heap_mb = int(os.environ.get("ALPS_DSP_HEAP_MB", str(default_mb)))
    print(f"[DSPHeap] full_model_heap_mb={heap_mb}")
    patch_dsp_heap_mb(heap_mb)


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


def rewrite_batch_matmul_to_matmul(ir: str, enable_m_pad: bool = False) -> tuple[str, int]:
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
        # K and N must be tile-aligned.  M (rows / tokens) may be unaligned when
        # M-pad is enabled: the C++ MatmulToHexKL pass pads M up to the next tile
        # multiple and discards the padded output rows (safest pad direction).
        if (K % 32) != 0 or (N % 32) != 0:
            out_lines.append(line)
            continue
        if (M % 32) != 0 and not enable_m_pad:
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


def _prepare_launch_path(
    module,
    func_name,
    mlir_text: Optional[str] = None,
    out_dir: Optional[Path] = None,
):
    """Dump the MLIR to disk and return the path to compile.

    When mlir_text is provided (e.g. HexKL-rewritten f16 IR) it is cleaned of
    cf.assert lines, re-parsed to bytecode for validation, and the cleaned text
    is written to <func>_f16matmul.mlir. Otherwise the module's original
    bytecode is preserved (re-parsing str(module) would drop dialects like
    tm_tensor)."""
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
        bytecode = module.operation.get_asm(binary=True)
        launch_path = str(linalg_filename)

    with open(linalg_filename, "wb") as f:
        f.write(bytecode)
    return launch_path


def hex_execution(
    module,
    func_name,
    inputs,
    options: dict = None,
    mlir_text: Optional[str] = None,
    out_dir: Optional[Path] = None,
    iterations: int = 1,
):
    launch_path = _prepare_launch_path(module, func_name, mlir_text, out_dir)
    options = options or {}
    return TorchMLIRHexagonLauncher().run_torch_mlir(
        launch_path, inputs, func_name, iterations=iterations, options=options
    )


def hex_execution_interleaved(
    module,
    func_name,
    inputs,
    configs_by_profile: dict,
    out_dir: Optional[Path] = None,
    iterations: int = 1,
    rounds: int = 1,
):
    """Round-robin interleaved measurement across profiles.

    configs_by_profile maps a profile label to a dict with keys "options"
    (HexagonOptions dict) and "mlir_text" (HexKL-rewritten IR text, or None to
    compile the original module bytecode). Each profile's launch path is
    prepared here so HexKL profiles compile the rewritten IR while HVX/scalar
    profiles compile the original module."""
    prepared = {}
    for profile, cfg in configs_by_profile.items():
        launch_path = _prepare_launch_path(
            module, func_name, cfg.get("mlir_text"), out_dir
        )
        prepared[profile] = {
            "launch_path": launch_path,
            "options": cfg.get("options") or {},
        }
    return TorchMLIRHexagonLauncher().run_torch_mlir_interleaved(
        prepared,
        inputs,
        func_name,
        iterations=iterations,
        rounds=rounds,
    )


def apply_hexkl_ir_rewrites(ir: str, enable_m_pad: bool = False) -> tuple[str, int, int]:
    if os.environ.get("HEXAGON_BASELINE_MODE") == "upstream-strict":
        print("[UpstreamStrict] host-side HexKL IR rewrites disabled")
        return ir, 0, 0
    ir2, n_bm = rewrite_batch_matmul_to_matmul(ir, enable_m_pad=enable_m_pad)
    ir2, n_f16 = rewrite_matmul_inputs_to_f16(ir2)
    return ir2, n_bm, n_f16


def hexagon_options_phase4(
    enable_hexkl: bool,
    enable_alps_vdae: bool = False,
    enable_alps_layout_aware: bool = True,
    alps_lookahead: int = 2,
    enable_alps_adaptive: bool = True,
    enable_alps_items_1_7: bool = False,
    lower_constants_separate: bool = True,
    backend_profile: Optional[str] = None,
    enable_lwp: bool = False,
    lwp_loop_depth: int = 1,
    disable_lwp_loop: bool = False,
    instrument_lwp_hexkl_phases: bool = False,
    alps_items_through: int = 0,
    enable_alps_kv_vtcm: bool = False,
    enable_alps_activation_multicast: bool = False,
    enable_alps_m_pad_hmx: bool = False,
    enable_out_params: bool = False,
    prefetch_baseline: str = "none",
    prefetch_baseline_distance: int = 1,
    apt_get_hx_manual_candidate_ids: str = "",
    enable_alps_kv_cache_prefetch: bool = False,
    disable_alps_persistent_wh_cache: bool = False,
    alps_p0_mode: str = "none",
    enable_alps_fp16_hvx_arithmetic: Optional[bool] = None,
    enable_alps_hvx_widening_conv: Optional[bool] = None,
):
    from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions

    options = HexagonOptions().__dict__
    profile = backend_profile or os.environ.get(
        "ALPS_BACKEND_PROFILE", "legacy-scalar"
    )
    valid_profiles = {
        "legacy-scalar",
        "hvx-vector",
        "hvx-vector-vtcm",
    }
    if profile not in valid_profiles:
        raise ValueError(
            f"unknown backend profile {profile!r}; "
            f"expected one of {sorted(valid_profiles)}"
        )
    if lwp_loop_depth < 0:
        raise ValueError("LWP loop depth must be non-negative")
    if not 0 <= alps_items_through <= 7:
        raise ValueError("Alps cumulative item level must be in [0, 7]")
    if prefetch_baseline not in ("none", "prefetch-kernel-hx", "apt-get-hx"):
        raise ValueError(f"unknown prefetch baseline {prefetch_baseline!r}")
    if prefetch_baseline_distance <= 0:
        raise ValueError("prefetch baseline distance must be positive")
    valid_alps_p0_modes = {
        "none", "semantic", "fusion", "elementwise-fusion",
        "multi-use-fusion", "split-reduction", "slicing", "runtime",
        "legacy-all",
    }
    if alps_p0_mode not in valid_alps_p0_modes:
        raise ValueError(
            f"unknown ALPS P0 mode {alps_p0_mode!r}; "
            f"expected one of {sorted(valid_alps_p0_modes)}"
        )
    if enable_alps_kv_cache_prefetch and alps_p0_mode != "none":
        raise ValueError("legacy item7 and ALPS P0 mode are mutually exclusive")
    alps_p0_enabled = alps_p0_mode != "none"
    if enable_alps_fp16_hvx_arithmetic is None:
        enable_alps_fp16_hvx_arithmetic = (
            os.environ.get("ALPS_ENABLE_FP16_HVX_ARITHMETIC", "0") == "1"
        )
    if enable_alps_hvx_widening_conv is None:
        enable_alps_hvx_widening_conv = (
            os.environ.get("ALPS_ENABLE_HVX_WIDENING_CONV", "0") == "1"
        )

    options["enableLWP"] = bool(enable_lwp)
    options["disableLWPLoop"] = bool(disable_lwp_loop)
    options["LWPloopDepth"] = int(lwp_loop_depth)
    options["instrumentLWPHexKLPhases"] = bool(instrument_lwp_hexkl_phases)
    options["lowerConstantsInSeparateSharedObjects"] = bool(lower_constants_separate)
    options["enableHexKL"] = bool(enable_hexkl)
    if profile == "legacy-scalar":
        options["enableVectorization"] = False
        options["enableVTCMTiling"] = False
        options["enableConvertToHexagonmem"] = bool(enable_hexkl)
    elif profile == "hvx-vector":
        options["enableVectorization"] = True
        options["enableVTCMTiling"] = False
        options["enableConvertToHexagonmem"] = True
    else:
        options["enableVectorization"] = True
        options["enableVTCMTiling"] = True
        options["enableConvertToHexagonmem"] = True
    cumulative_level = 7 if enable_alps_items_1_7 else alps_items_through
    if os.environ.get("HEXAGON_BASELINE_MODE") == "upstream-strict":
        if (
            enable_alps_vdae
            or cumulative_level
            or enable_alps_kv_vtcm
            or enable_alps_activation_multicast
            or enable_alps_m_pad_hmx
            or enable_alps_kv_cache_prefetch
            or alps_p0_enabled
            or enable_alps_fp16_hvx_arithmetic
            or enable_alps_hvx_widening_conv
            or prefetch_baseline != "none"
        ):
            raise ValueError("upstream-strict baseline cannot enable downstream passes")
        # Reconstruct from the installed runtime's native dataclass and change
        # only explicit baseline knobs. This prevents downstream-only option
        # keys (including the runners' default-on out-param workaround) from
        # leaking into the clean upstream plugin.
        native_options = HexagonOptions().__dict__.copy()
        native_options["enableHexKL"] = bool(enable_hexkl)
        if "enableBufferResultsToOutParams" in native_options:
            native_options["enableBufferResultsToOutParams"] = bool(
                enable_out_params
            )
        # Upstream VTCM tiling is kernel-oriented: on a full linear graph its
        # one-shot bufferization currently keeps all per-op staging buffers
        # live until function exit. DINOv2-small alone accumulates 822 VTCM
        # allocations (~190 MB). Disable this optional staging optimization for
        # both native baselines while retaining real HVX vectorization. HexKL's
        # own local VTCM allocations are inserted later and explicitly freed.
        native_options["enableVTCMTiling"] = False
        native_options["lowerConstantsInSeparateSharedObjects"] = bool(
            lower_constants_separate
        )
        native_options["enableLWP"] = bool(enable_lwp)
        native_options["disableLWPLoop"] = bool(disable_lwp_loop)
        native_options["LWPloopDepth"] = int(lwp_loop_depth)
        native_options["instrumentLWPHexKLPhases"] = bool(
            instrument_lwp_hexkl_phases
        )
        print(
            "[BackendConfig] profile=upstream-native "
            f"vectorization={int(native_options['enableVectorization'])} "
            f"vtcm_tiling={int(native_options['enableVTCMTiling'])} "
            f"hexagonmem={int(native_options['enableConvertToHexagonmem'])} "
            f"hexkl={int(native_options['enableHexKL'])} host_rewrites=0"
        )
        return native_options
    if prefetch_baseline != "none" and (
        enable_alps_vdae or cumulative_level > 0 or alps_p0_enabled
    ):
        raise ValueError("external prefetch baselines cannot be combined with ALPS")
    options["enablePrefetchKernelHX"] = prefetch_baseline == "prefetch-kernel-hx"
    options["enableAlpsFP16HVXArithmetic"] = bool(
        enable_alps_fp16_hvx_arithmetic
    )
    options["enableAlpsHVXWideningConv"] = bool(
        enable_alps_hvx_widening_conv
    )
    options["prefetchKernelHxDistance"] = int(prefetch_baseline_distance)
    options["enableAPTGetHX"] = prefetch_baseline == "apt-get-hx"
    options["aptGetHxDistance"] = int(prefetch_baseline_distance)
    options["aptGetHxManualCandidateIds"] = apt_get_hx_manual_candidate_ids
    options["enablePrefetch"] = bool(
        enable_alps_vdae
        or cumulative_level >= 1
        or enable_alps_kv_cache_prefetch
        or alps_p0_mode in ("runtime", "legacy-all")
    )
    options["enableAlpsLayoutAware"] = bool(enable_alps_layout_aware)
    options["alpsLookahead"] = int(alps_lookahead)
    options["alpsVdaePanelTiles"] = max(
        1, min(8, int(os.environ.get("ALPS_VDAE_PANEL_TILES", "1")))
    )
    options["enableAlpsVDAE"] = bool(
        enable_alps_vdae or cumulative_level >= 3
    )
    options["enableAlpsAdaptive"] = bool(enable_alps_adaptive)
    options["enableAlpsPersistentWhCache"] = bool(
        cumulative_level >= 4 and not disable_alps_persistent_wh_cache
    )
    options["enableAlpsTwoDimPipeline"] = cumulative_level >= 5
    options["enableAlpsVtcmColoring"] = cumulative_level >= 6
    # Item 7 covers page-aware prefetch of compiler-identified attention K/V
    # streams.  For autoregressive decoders those streams may be persistent
    # K/V cache pages; for encoders they are the current invocation's ordinary
    # attention K/V tensors.  Both are valid early-data-movement opportunities.
    options["enableAlpsKvCachePrefetch"] = bool(
        cumulative_level >= 7 or enable_alps_kv_cache_prefetch
    )
    options["enableAlpsKvSemanticTracking"] = alps_p0_enabled
    options["enableAlpsKvFusionPolicy"] = alps_p0_mode in ("fusion", "legacy-all")
    options["enableAlpsKvElementwiseFusionPolicy"] = (
        alps_p0_mode == "elementwise-fusion"
    )
    options["enableAlpsKvMultiUseFusionPolicy"] = (
        alps_p0_mode == "multi-use-fusion"
    )
    options["enableAlpsKvSplitReductionPolicy"] = (
        alps_p0_mode == "split-reduction"
    )
    options["enableAlpsKvSlicingPolicy"] = alps_p0_mode in ("slicing", "legacy-all")
    options["enableAlpsKvRuntimePrefetch"] = alps_p0_mode in (
        "runtime", "legacy-all"
    )
    options["enableAlpsMovementLedger"] = (
        os.environ.get("ALPS_ENABLE_MOVEMENT_LEDGER", "0") == "1"
    )
    options["enableAlpsZeroCopyAttention"] = (
        os.environ.get("ALPS_ENABLE_ZERO_COPY_ATTENTION", "0") == "1"
    )
    options["enableAlpsProducerDirectAttention"] = (
        os.environ.get("ALPS_ENABLE_PRODUCER_DIRECT_ATTENTION", "0") == "1"
    )
    options["enableAlpsConsumerDrivenLayout"] = (
        os.environ.get("ALPS_ENABLE_CONSUMER_DRIVEN_LAYOUT", "0") == "1"
    )
    options["enableAlpsConsumerLayoutPropagation"] = (
        os.environ.get("ALPS_ENABLE_CONSUMER_LAYOUT_PROPAGATION", "0") == "1"
    )
    options["enableAlpsContinuityAudit"] = (
        os.environ.get("ALPS_ENABLE_CONTINUITY_AUDIT", "0") == "1"
    )
    options["enableAlpsLoopInterchangedDirectFormation"] = (
        os.environ.get("ALPS_ENABLE_LOOP_INTERCHANGED_DIRECT", "0") == "1"
    )
    options["enableAlpsRegisterTileFormation"] = (
        os.environ.get("ALPS_ENABLE_REGISTER_TILE_FORMATION", "0") == "1"
    )
    options["alpsRegisterTileDemandBegin"] = int(
        os.environ.get("ALPS_REGISTER_TILE_DEMAND_BEGIN", "0")
    )
    options["alpsRegisterTileDemandEnd"] = int(
        os.environ.get("ALPS_REGISTER_TILE_DEMAND_END", "-1")
    )
    options["enableAlpsContractDischargeLedger"] = (
        os.environ.get("ALPS_ENABLE_CONTRACT_DISCHARGE_LEDGER", "0") == "1"
    )
    options["enableAlpsRepresentationSupplyAnalysis"] = (
        os.environ.get("ALPS_ENABLE_REPRESENTATION_SUPPLY_ANALYSIS", "0") == "1"
    )
    options["enableAlpsLayoutSupplyPrefetch"] = (
        os.environ.get("ALPS_ENABLE_LAYOUT_SUPPLY_PREFETCH", "0") == "1"
    )
    options["enableAlpsCrpSupplyAnalysis"] = (
        os.environ.get("ALPS_ENABLE_CRP_SUPPLY_ANALYSIS", "0") == "1"
    )
    options["enableAlpsCrpSupplyPrefetch"] = (
        os.environ.get("ALPS_ENABLE_CRP_SUPPLY_PREFETCH", "0") == "1"
    )
    options["enableAlpsCrpSegmentedSupply"] = (
        os.environ.get("ALPS_ENABLE_CRP_SEGMENTED_SUPPLY", "0") == "1"
    )
    options["enableAlpsCrpVtcmFormation"] = (
        os.environ.get("ALPS_ENABLE_CRP_VTCM_FORMATION", "0") == "1"
    )
    options["enableAlpsCrpVtcmWindow"] = (
        os.environ.get("ALPS_ENABLE_CRP_VTCM_WINDOW", "0") == "1"
    )
    options["enableAlpsCrpVtcmAsyncWindow"] = (
        os.environ.get("ALPS_ENABLE_CRP_VTCM_ASYNC_WINDOW", "0") == "1"
    )
    options["enableAlpsCrpProducerDirectAnalysis"] = (
        os.environ.get("ALPS_ENABLE_CRP_PRODUCER_DIRECT_ANALYSIS", "0") == "1"
    )
    options["enableAlpsCrpProducerDirectVtcm"] = (
        os.environ.get("ALPS_ENABLE_CRP_PRODUCER_DIRECT_VTCM", "0") == "1"
    )
    options["enableAlpsCrpProducerDirectHeadMajor"] = (
        os.environ.get("ALPS_ENABLE_CRP_PRODUCER_DIRECT_HEAD_MAJOR", "0") == "1"
    )
    options["enableAlpsCrpProducerLoopFormation"] = (
        os.environ.get("ALPS_ENABLE_CRP_PRODUCER_LOOP_FORMATION", "0") == "1"
    )
    options["enableAlpsAttentionDestinationFormation"] = (
        os.environ.get("ALPS_ENABLE_ATTENTION_DESTINATION_FORMATION", "0") == "1"
    )
    options["enableAlpsPatchConvFormation"] = (
        os.environ.get("ALPS_ENABLE_PATCH_CONV_FORMATION", "0") == "1"
    )
    options["enableAlpsHmxF16EpilogueFormation"] = (
        os.environ.get("ALPS_ENABLE_HMX_F16_EPILOGUE_FORMATION", "0") == "1"
    )
    options["enableAlpsHmxDirectOutputFormation"] = (
        os.environ.get("ALPS_ENABLE_HMX_DIRECT_OUTPUT_FORMATION", "0") == "1"
    )
    options["enableAlpsHmxF16BiasEpilogueFormation"] = (
        os.environ.get("ALPS_ENABLE_HMX_F16_BIAS_EPILOGUE_FORMATION", "0") == "1"
    )
    options["enableAlpsHmxAsyncDrainAnalysis"] = (
        os.environ.get("ALPS_ENABLE_HMX_ASYNC_DRAIN_ANALYSIS", "0") == "1"
    )
    options["enableAlpsHmxAsyncDrain"] = (
        os.environ.get("ALPS_ENABLE_HMX_ASYNC_DRAIN", "0") == "1"
    )
    options["enableAlpsFusedTransformTransfer"] = (
        os.environ.get("ALPS_ENABLE_FUSED_TRANSFORM_TRANSFER", "0") == "1"
    )
    options["enableAlpsMinimalStaticAdmission"] = (
        os.environ.get("ALPS_ENABLE_MINIMAL_STATIC_ADMISSION", "0") == "1"
    )
    options["enableAlpsExactReadiness"] = (
        os.environ.get("ALPS_ENABLE_EXACT_READINESS", "0") == "1"
    )
    options["enableAlpsExactOverlap"] = (
        os.environ.get("ALPS_ENABLE_EXACT_OVERLAP", "0") == "1"
    )
    options["enableAlpsTrafficControl"] = (
        os.environ.get("ALPS_ENABLE_TRAFFIC_CONTROL", "0") == "1"
    )
    options["enableAlpsDualThreadDae"] = (
        os.environ.get("ALPS_ENABLE_VECTOR_DAE", "0") == "1"
    )
    if options["enableAlpsDualThreadDae"] and not options["enableAlpsExactOverlap"]:
        raise ValueError("vectorized DAE requires an exact-overlap residual supply")
    # P3b starts and retires UserDMA on the issuing Hexagon hardware thread.
    # UserDMA completion state is thread-contextual, so exact overlap must not
    # implicitly move completion polling to the optional scout thread.  The
    # DMA still overlaps the intervening HMX work between kick and consume.
    options["enableAlpsActivationMulticast"] = bool(
        enable_alps_activation_multicast
    )
    options["enableAlpsMPadHmx"] = bool(enable_alps_m_pad_hmx)
    options["enableBufferResultsToOutParams"] = bool(enable_out_params)
    options["enableAlpsDmaToVtcm"] = bool(enable_alps_kv_vtcm)
    options["enableHexagonmemCopyToDMA"] = bool(
        enable_alps_kv_vtcm
        or options["enableAlpsCrpVtcmFormation"]
        or options["enableAlpsCrpVtcmWindow"]
        or options["enableAlpsCrpVtcmAsyncWindow"]
    )
    if cumulative_level:
        print(
            f"[AlpsCumulative] items_through={cumulative_level}: "
            "prefetch/layout/V-DAE + persistent-WH + "
            "two-dimensional-pipeline + VTCM-coloring + "
            "attention-K/V-stream-prefetch (features gated by level)"
            + (" + K/V DMA-to-VTCM staging" if enable_alps_kv_vtcm else "")
        )
    print(
        "[BackendConfig] "
        f"profile={profile} "
        f"vectorization={int(options['enableVectorization'])} "
        f"vtcm_tiling={int(options['enableVTCMTiling'])} "
        f"hexagonmem={int(options['enableConvertToHexagonmem'])} "
        f"hexkl={int(options['enableHexKL'])} "
        f"prefetch_baseline={prefetch_baseline} "
        f"prefetch_distance={prefetch_baseline_distance} "
        f"lwp={int(options['enableLWP'])} "
        f"lwp_loop_depth={options['LWPloopDepth']} "
        f"lwp_loops_disabled={int(options['disableLWPLoop'])} "
        f"lwp_hexkl_phases={int(options['instrumentLWPHexKLPhases'])} "
        f"alps_p2e_consumer_layout={int(options['enableAlpsConsumerDrivenLayout'])} "
        f"alps_p2f_layout_propagation={int(options['enableAlpsConsumerLayoutPropagation'])} "
        f"alps_p2g_continuity_audit={int(options['enableAlpsContinuityAudit'])} "
        f"alps_p2g_loop_interchanged={int(options['enableAlpsLoopInterchangedDirectFormation'])} "
        f"alps_p2g_register_tile={int(options['enableAlpsRegisterTileFormation'])} "
        f"alps_p2g_demand_window={options['alpsRegisterTileDemandBegin']}:{options['alpsRegisterTileDemandEnd']} "
        f"alps_p5a_discharge_ledger={int(options['enableAlpsContractDischargeLedger'])} "
        f"alps_p5b_supply_analysis={int(options['enableAlpsRepresentationSupplyAnalysis'])} "
        f"alps_p5c_layout_supply={int(options['enableAlpsLayoutSupplyPrefetch'])} "
        f"alps_p5f_a_crp_supply={int(options['enableAlpsCrpSupplyAnalysis'])} "
        f"alps_p5f_b_crp_prefetch={int(options['enableAlpsCrpSupplyPrefetch'])} "
        f"alps_p5f_c_segmented={int(options['enableAlpsCrpSegmentedSupply'])} "
        f"alps_p5g_a_vtcm={int(options['enableAlpsCrpVtcmFormation'])} "
        f"alps_p5g_b_window={int(options['enableAlpsCrpVtcmWindow'])} "
        f"alps_p5g_c_async={int(options['enableAlpsCrpVtcmAsyncWindow'])} "
        f"alps_p5g_d_producer_analysis={int(options['enableAlpsCrpProducerDirectAnalysis'])} "
        f"alps_p5g_e_producer_vtcm={int(options['enableAlpsCrpProducerDirectVtcm'])} "
        f"alps_p5g_f_head_major={int(options['enableAlpsCrpProducerDirectHeadMajor'])} "
        f"alps_p5g_g_producer_loop={int(options['enableAlpsCrpProducerLoopFormation'])} "
        f"alps_p5h_attention_destination={int(options['enableAlpsAttentionDestinationFormation'])} "
        f"alps_p5i_patch_conv={int(options['enableAlpsPatchConvFormation'])} "
        f"alps_p5j_hmx_f16_epilogue={int(options['enableAlpsHmxF16EpilogueFormation'])} "
        f"alps_p5k_hmx_direct_output={int(options['enableAlpsHmxDirectOutputFormation'])} "
        f"alps_p5l_hmx_f16_bias_epilogue={int(options['enableAlpsHmxF16BiasEpilogueFormation'])} "
        f"alps_p5m_hmx_async_drain_analysis={int(options['enableAlpsHmxAsyncDrainAnalysis'])} "
        f"alps_p5n_hmx_async_drain={int(options['enableAlpsHmxAsyncDrain'])} "
        f"alps_p4a_traffic={int(options['enableAlpsTrafficControl'])} "
        f"alps_vector_dae={int(options['enableAlpsDualThreadDae'])} "
        f"alps_vdae_panel_tiles={options['alpsVdaePanelTiles']} "
        f"alps_fp16_hvx={int(options['enableAlpsFP16HVXArithmetic'])} "
        f"alps_hvx_widening_conv={int(options['enableAlpsHVXWideningConv'])}"
    )
    return options


def add_phase4_args(parser):
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-alps-vdae", action="store_true")
    parser.add_argument(
        "--enable-alps-fp16-hvx-arithmetic",
        action="store_true",
        help=(
            "Default-off negative ablation: lower eligible FP32 convolution/"
            "elementwise islands in this FP16 model to FP16 arithmetic. V73 "
            "admission additionally requires the object half-helper audit."
        ),
    )
    parser.add_argument(
        "--enable-alps-hvx-widening-conv",
        action="store_true",
        help=(
            "ALPS C: retain F16 convolution operands, widen 32 output lanes "
            "once, and accumulate in a native 128-byte F32 HVX vector."
        ),
    )
    parser.add_argument(
        "--prefetch-baseline",
        choices=("none", "prefetch-kernel-hx", "apt-get-hx"),
        default="none",
        help="Run one isolated external prefetch baseline.",
    )
    parser.add_argument(
        "--prefetch-baseline-distance",
        type=int,
        default=1,
        help="Static or APT-profile-selected future iteration distance.",
    )
    parser.add_argument(
        "--apt-get-hx-manual-candidate-ids",
        default="",
        help="Comma-separated manually qualified stable candidate IDs.",
    )
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--alps-lookahead", type=int, default=2)
    parser.add_argument("--disable-alps-adaptive", action="store_true")
    parser.add_argument(
        "--enable-alps-items-1-7",
        action="store_true",
        help="Enable the cumulative innovation items 1 through 7.",
    )
    parser.add_argument(
        "--alps-items-through",
        type=int,
        choices=range(0, 8),
        default=0,
        metavar="N",
        help="Ablation: enable cumulative Alps items through N (0-7).",
    )
    parser.add_argument(
        "--enable-alps-kv-cache-prefetch",
        action="store_true",
        help="Legacy umbrella switch reproducing the complete historical item 7.",
    )
    parser.add_argument(
        "--alps-p0-mode",
        choices=(
            "none", "semantic", "fusion", "elementwise-fusion",
            "multi-use-fusion", "split-reduction", "slicing", "runtime",
            "legacy-all",
        ),
        default="none",
        help=(
            "ALPS P0 causal K/V mode. fusion, slicing, and runtime each imply "
            "semantic tracking; legacy-all reproduces historical item 7."
        ),
    )
    parser.add_argument(
        "--disable-alps-persistent-wh-cache",
        action="store_true",
        help=(
            "Ablation: keep later cumulative items enabled but disable item 4 "
            "persistent-WH caching and its cold/warm/invalidated protocol."
        ),
    )
    parser.add_argument(
        "--enable-alps-kv-vtcm",
        action="store_true",
        help=(
            "Stage item-7 K/V streams into VTCM through synchronous DMA "
            "instead of issuing L2-only hints."
        ),
    )
    parser.add_argument(
        "--enable-alps-activation-multicast",
        action="store_true",
        help="Enable Alps N2 activation multicast for sibling projections.",
    )
    parser.add_argument(
        "--enable-alps-m-pad-hmx",
        action="store_true",
        help=(
            "Pad the M (rows/tokens) dimension up to a multiple of 32 so "
            "unaligned-token encoders (DINOv2/BEiT/DeiT/Whisper) lower matmuls "
            "through HexKL/HMX."
        ),
    )
    parser.add_argument(
        "--enable-out-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Convert memref function results into trailing out-param arguments "
            "(function returns void). Avoids the by-value memref (sret) return "
            "that the Hexagon backend miscompiles for large monolithic full "
            "models. Use --no-enable-out-params to reproduce the legacy ABI."
        ),
    )
    parser.add_argument(
        "--backend-profile",
        choices=("legacy-scalar", "hvx-vector", "hvx-vector-vtcm"),
        default=os.environ.get("ALPS_BACKEND_PROFILE", "legacy-scalar"),
        help=(
            "Backend codegen profile. legacy-scalar reproduces historical "
            "results; hvx-vector enables vectorization; hvx-vector-vtcm also "
            "enables VTCM tiling."
        ),
    )
    parser.add_argument(
        "--enable-lwp",
        action="store_true",
        help="Enable function/loop Lightweight Profiling instrumentation.",
    )
    parser.add_argument(
        "--lwp-loop-depth",
        type=int,
        default=1,
        help="Maximum sibling loop depth instrumented by LWP (default: 1).",
    )
    parser.add_argument(
        "--disable-lwp-loop",
        action="store_true",
        help="Instrument only the function body, not individual loops.",
    )
    parser.add_argument(
        "--lwp-hexkl-phases",
        action="store_true",
        help="Profile individual HexKL/HMX compute and representation phases.",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument(
        "--interleave-profiles",
        type=str,
        default=None,
        help=(
            "Comma-separated profiles to measure interleaved (compile once, "
            "round-robin execute). Valid labels: legacy-scalar, hvx-vector, "
            "hvx-vector-vtcm, hexkl (hvx-vector-vtcm + HexKL overlay), "
            "hexkl-items17 (hexkl + Alps items 1-7)."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of round-robin rounds for --interleave-profiles.",
    )
    return parser


# Profile label -> hexagon_options_phase4 overrides for interleaved runs.
# Each entry sets backend_profile plus the HexKL/items overlays; profiles whose
# "hexkl" flag is True compile the HexKL-rewritten IR (mlir_text).
INTERLEAVE_PROFILE_SPECS = {
    "legacy-scalar": dict(
        backend_profile="legacy-scalar", enable_hexkl=False, items_1_7=False
    ),
    "hvx-vector": dict(
        backend_profile="hvx-vector", enable_hexkl=False, items_1_7=False
    ),
    "hvx-vector-vtcm": dict(
        backend_profile="hvx-vector-vtcm", enable_hexkl=False, items_1_7=False
    ),
    "hexkl": dict(
        backend_profile="hvx-vector-vtcm", enable_hexkl=True, items_1_7=False
    ),
    "hexkl-items17": dict(
        backend_profile="hvx-vector-vtcm", enable_hexkl=True, items_1_7=True
    ),
}


def build_interleave_configs(args, ir: str):
    """Build {profile: {"options", "mlir_text"}} for --interleave-profiles.

    ir is str(module); HexKL profiles use the rewritten f16 IR as mlir_text.
    Raises ValueError on unknown profile labels."""
    labels = [p.strip() for p in args.interleave_profiles.split(",") if p.strip()]
    hexkl_text = None
    configs = {}
    for label in labels:
        if label not in INTERLEAVE_PROFILE_SPECS:
            raise ValueError(
                f"unknown interleave profile {label!r}; expected one of "
                f"{sorted(INTERLEAVE_PROFILE_SPECS)}"
            )
        spec = INTERLEAVE_PROFILE_SPECS[label]
        options = hexagon_options_phase4(
            spec["enable_hexkl"],
            args.enable_alps_vdae,
            not args.disable_layout_aware,
            args.alps_lookahead,
            not args.disable_alps_adaptive,
            spec["items_1_7"],
            lower_constants_separate=True,
            backend_profile=spec["backend_profile"],
            enable_lwp=args.enable_lwp,
            lwp_loop_depth=args.lwp_loop_depth,
            disable_lwp_loop=args.disable_lwp_loop,
            instrument_lwp_hexkl_phases=args.lwp_hexkl_phases,
            alps_items_through=args.alps_items_through,
            enable_alps_m_pad_hmx=(
                spec["enable_hexkl"]
                and getattr(args, "enable_alps_m_pad_hmx", False)
            ),
        )
        mlir_text = None
        if spec["enable_hexkl"]:
            if hexkl_text is None:
                candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(
                    ir,
                    enable_m_pad=getattr(
                        args, "enable_alps_m_pad_hmx", False
                    ),
                )
                if n_batch or n_f16:
                    hexkl_text = candidate
                print(
                    f"[HexKL] batch_matmul→matmul={n_batch}, "
                    f"f16-input rewrite={n_f16}"
                )
                if not (n_batch or n_f16):
                    print(
                        f"[HexKL] WARNING: profile {label!r} yielded 0 rewrites "
                        "— no HMX coverage for this model shape"
                    )
            mlir_text = hexkl_text
        configs[label] = {"options": options, "mlir_text": mlir_text}
    return configs


def phase4_kwargs_from_args(args):
    return dict(
        enable_hexkl=args.enable_hexkl,
        enable_alps_vdae=args.enable_alps_vdae,
        enable_alps_layout_aware=not args.disable_layout_aware,
        alps_lookahead=args.alps_lookahead,
        enable_alps_adaptive=not args.disable_alps_adaptive,
        enable_alps_items_1_7=args.enable_alps_items_1_7,
        seq_len=getattr(args, "seq_len", None),
    )
