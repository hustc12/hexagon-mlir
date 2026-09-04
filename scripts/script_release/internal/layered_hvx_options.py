"""Matched backend options for complete-model layered HVX experiments."""

from __future__ import annotations

import argparse
import os


def add_layered_hvx_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument(
        "--enable-lwp",
        action="store_true",
        help="Enable function/loop Lightweight Profiling for every stage.",
    )
    parser.add_argument("--lwp-loop-depth", type=int, default=1)
    parser.add_argument("--disable-lwp-loop", action="store_true")
    parser.add_argument("--lwp-hexkl-phases", action="store_true")
    parser.add_argument(
        "--enable-alps-fp16-hvx-arithmetic",
        action="store_true",
        help=(
            "Default-off negative ablation: lower eligible FP32 convolution/"
            "elementwise islands in an FP16 model to FP16 arithmetic. V73 "
            "admission additionally requires the object half-helper audit."
        ),
    )
    parser.add_argument(
        "--enable-alps-hvx-widening-conv",
        action="store_true",
        help="ALPS C mixed F16/F32 NCHW convolution HVX schedule.",
    )
    parser.add_argument(
        "--prefetch-baseline",
        choices=("none", "prefetch-kernel-hx", "apt-get-hx"),
        default="none",
    )
    parser.add_argument("--prefetch-baseline-distance", type=int, default=1)
    parser.add_argument("--apt-get-hx-manual-candidate-ids", default="")
    parser.add_argument(
        "--enable-alps-kv-cache-prefetch",
        action="store_true",
        help="Legacy umbrella alias for the complete historical item-7 policy.",
    )
    parser.add_argument(
        "--alps-p0-mode",
        choices=(
            "none", "semantic", "fusion", "elementwise-fusion",
            "multi-use-fusion", "split-reduction", "slicing", "runtime",
            "legacy-all",
        ),
        default="none",
        help="ALPS P0 causal K/V policy; non-none modes imply semantic tracking.",
    )
    # Accepted explicitly so the unified driver uses exactly the same item-7
    # command line for monolithic and layered model families.
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--disable-alps-adaptive", action="store_true")
    parser.add_argument(
        "--alps-lookahead",
        type=int,
        default=2,
        help="Bounded ALPS access lead in consumer tiles (1..7).",
    )
    parser.add_argument(
        "--enable-alps-vector-dae",
        action="store_true",
        help=(
            "Run exact residual supply on the scout-owned vectorized "
            "Decoupled Access--Execute path."
        ),
    )


def make_layered_hvx_options(args: argparse.Namespace) -> dict:
    from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions

    if args.prefetch_baseline_distance <= 0:
        raise ValueError("prefetch baseline distance must be positive")
    if args.lwp_loop_depth < 0:
        raise ValueError("LWP loop depth must be non-negative")
    if not 1 <= args.alps_lookahead <= 7:
        raise ValueError("ALPS lookahead must be in [1, 7]")
    item7 = bool(args.enable_alps_kv_cache_prefetch)
    alps_mode = str(args.alps_p0_mode)
    if item7 and alps_mode != "none":
        raise ValueError("legacy item7 and --alps-p0-mode are mutually exclusive")
    alps_enabled = alps_mode != "none"
    if args.prefetch_baseline != "none" and (item7 or alps_enabled):
        raise ValueError("external prefetch baselines cannot be combined with ALPS")
    if (item7 or alps_enabled) and not (
        args.disable_layout_aware and args.disable_alps_adaptive
    ):
        raise ValueError(
            "ALPS P0 requires --disable-layout-aware and "
            "--disable-alps-adaptive"
        )

    options = HexagonOptions().__dict__.copy()
    options["enableVectorization"] = True
    options["enableHexKL"] = bool(args.enable_hexkl)
    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = True
    # Every staged language model is already uniformly f16. Do not insert a
    # mixed-precision conversion pass.
    options["enableConversionToFp16"] = False
    options["enableAlpsFP16HVXArithmetic"] = bool(
        args.enable_alps_fp16_hvx_arithmetic
    )
    options["enableAlpsHVXWideningConv"] = bool(
        args.enable_alps_hvx_widening_conv
    )
    options["lowerConstantsInSeparateSharedObjects"] = True
    options["enableLWP"] = bool(args.enable_lwp)
    options["disableLWPLoop"] = bool(args.disable_lwp_loop)
    options["LWPloopDepth"] = int(args.lwp_loop_depth)
    options["instrumentLWPHexKLPhases"] = bool(args.lwp_hexkl_phases)
    if "enableBufferResultsToOutParams" in options:
        options["enableBufferResultsToOutParams"] = True

    options["enablePrefetchKernelHX"] = (
        args.prefetch_baseline == "prefetch-kernel-hx"
    )
    options["prefetchKernelHxDistance"] = int(args.prefetch_baseline_distance)
    options["enableAPTGetHX"] = args.prefetch_baseline == "apt-get-hx"
    options["aptGetHxDistance"] = int(args.prefetch_baseline_distance)
    options["aptGetHxManualCandidateIds"] = args.apt_get_hx_manual_candidate_ids

    runtime_prefetch = alps_mode in ("runtime", "legacy-all")
    options["enablePrefetch"] = item7 or runtime_prefetch
    options["enableAlpsLayoutAware"] = False
    options["enableAlpsVDAE"] = False
    options["enableAlpsAdaptive"] = False
    options["enableAlpsPersistentWhCache"] = False
    options["enableAlpsTwoDimPipeline"] = False
    options["enableAlpsVtcmColoring"] = False
    options["enableAlpsKvCachePrefetch"] = item7
    options["enableAlpsKvSemanticTracking"] = alps_enabled
    options["enableAlpsKvFusionPolicy"] = alps_mode in ("fusion", "legacy-all")
    options["enableAlpsKvElementwiseFusionPolicy"] = (
        alps_mode == "elementwise-fusion"
    )
    options["enableAlpsKvMultiUseFusionPolicy"] = (
        alps_mode == "multi-use-fusion"
    )
    options["enableAlpsKvSplitReductionPolicy"] = (
        alps_mode == "split-reduction"
    )
    options["enableAlpsKvSlicingPolicy"] = alps_mode in ("slicing", "legacy-all")
    options["enableAlpsKvRuntimePrefetch"] = runtime_prefetch
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
    options["enableAlpsDualThreadDae"] = bool(
        args.enable_alps_vector_dae
        or os.environ.get("ALPS_ENABLE_VECTOR_DAE", "0") == "1"
    )
    options["alpsLookahead"] = int(args.alps_lookahead)
    if options["enableAlpsDualThreadDae"] and not options["enableAlpsExactOverlap"]:
        raise ValueError("vectorized DAE requires an exact-overlap residual supply")
    options["enableAlpsDmaToVtcm"] = False
    options["enableHexagonmemCopyToDMA"] = (
        options["enableAlpsCrpVtcmFormation"]
        or options["enableAlpsCrpVtcmWindow"]
        or options["enableAlpsCrpVtcmAsyncWindow"]
    )

    print(
        "[LayeredBackendConfig] "
        f"vectorization=1 hexkl={int(options['enableHexKL'])} "
        f"lwp={int(options['enableLWP'])} "
        f"lwp_hexkl_phases={int(options['instrumentLWPHexKLPhases'])} "
        "fp16_model_storage=1 kernel_precision=mixed_f16_hmx_f32_hvx "
        "vtcm_tiling=0 runtime_conversion_to_fp16=0 "
        f"prefetch_baseline={args.prefetch_baseline} "
        f"legacy_item7={int(item7)} alps_p0_mode={alps_mode} "
        f"alps_p1_ledger={int(options['enableAlpsMovementLedger'])} "
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
        f"alps_p2d_admission={int(options['enableAlpsMinimalStaticAdmission'])} "
        f"alps_p3a_exact={int(options['enableAlpsExactReadiness'])}"
        f" alps_p3b_overlap={int(options['enableAlpsExactOverlap'])}"
        f" alps_p4a_traffic={int(options['enableAlpsTrafficControl'])}"
        f" alps_vector_dae={int(options['enableAlpsDualThreadDae'])}"
        f" alps_fp16_hvx={int(options['enableAlpsFP16HVXArithmetic'])}"
    )
    return options
