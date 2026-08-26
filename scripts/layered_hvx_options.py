"""Matched backend options for complete-model layered HVX experiments."""

from __future__ import annotations

import argparse
import os


def add_layered_hvx_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument(
        "--prefetch-baseline",
        choices=("none", "prefetch-kernel-hx", "apt-get-hx"),
        default="none",
    )
    parser.add_argument("--prefetch-baseline-distance", type=int, default=1)
    parser.add_argument("--apt-get-hx-manual-candidate-ids", default="")
    parser.add_argument(
        "--enable-omnifetch-kv-cache-prefetch",
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
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")


def make_layered_hvx_options(args: argparse.Namespace) -> dict:
    from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions

    if args.prefetch_baseline_distance <= 0:
        raise ValueError("prefetch baseline distance must be positive")
    item7 = bool(args.enable_omnifetch_kv_cache_prefetch)
    alps_mode = str(args.alps_p0_mode)
    if item7 and alps_mode != "none":
        raise ValueError("legacy item7 and --alps-p0-mode are mutually exclusive")
    alps_enabled = alps_mode != "none"
    if args.prefetch_baseline != "none" and (item7 or alps_enabled):
        raise ValueError("external prefetch baselines cannot be combined with ALPS")
    if (item7 or alps_enabled) and not (
        args.disable_layout_aware and args.disable_omnifetch_adaptive
    ):
        raise ValueError(
            "ALPS P0 requires --disable-layout-aware and "
            "--disable-omnifetch-adaptive"
        )

    options = HexagonOptions().__dict__.copy()
    options["enableVectorization"] = True
    options["enableHexKL"] = bool(args.enable_hexkl)
    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = True
    # Every staged language model is already uniformly f16. Do not insert a
    # mixed-precision conversion pass.
    options["enableConversionToFp16"] = False
    options["lowerConstantsInSeparateSharedObjects"] = True
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
    options["enableOmniFetchLayoutAware"] = False
    options["enableOmniFetchVDAE"] = False
    options["enableOmniFetchAdaptive"] = False
    options["enableOmniFetchPersistentWhCache"] = False
    options["enableOmniFetchTwoDimPipeline"] = False
    options["enableOmniFetchVtcmColoring"] = False
    options["enableOmniFetchKvCachePrefetch"] = item7
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
    options["enableAlpsContractDischargeLedger"] = (
        os.environ.get("ALPS_ENABLE_CONTRACT_DISCHARGE_LEDGER", "0") == "1"
    )
    options["enableAlpsRepresentationSupplyAnalysis"] = (
        os.environ.get("ALPS_ENABLE_REPRESENTATION_SUPPLY_ANALYSIS", "0") == "1"
    )
    options["enableAlpsLayoutSupplyPrefetch"] = (
        os.environ.get("ALPS_ENABLE_LAYOUT_SUPPLY_PREFETCH", "0") == "1"
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
    # Keep P3b issuer-owned: UserDMA start/poll must remain on the same
    # Hexagon hardware thread.  Dual-thread DAE remains an independent switch
    # for schemes whose completion work is safe to execute on the scout.
    options["enableOmniFetchDmaToVtcm"] = False
    options["enableHexagonmemCopyToDMA"] = False

    print(
        "[LayeredBackendConfig] "
        f"vectorization=1 hexkl={int(options['enableHexKL'])} "
        "uniform_fp16=1 vtcm_tiling=0 conversion_to_fp16=0 "
        f"prefetch_baseline={args.prefetch_baseline} "
        f"legacy_item7={int(item7)} alps_p0_mode={alps_mode} "
        f"alps_p1_ledger={int(options['enableAlpsMovementLedger'])} "
        f"alps_p2e_consumer_layout={int(options['enableAlpsConsumerDrivenLayout'])} "
        f"alps_p2f_layout_propagation={int(options['enableAlpsConsumerLayoutPropagation'])} "
        f"alps_p5a_discharge_ledger={int(options['enableAlpsContractDischargeLedger'])} "
        f"alps_p5b_supply_analysis={int(options['enableAlpsRepresentationSupplyAnalysis'])} "
        f"alps_p5c_layout_supply={int(options['enableAlpsLayoutSupplyPrefetch'])} "
        f"alps_p2d_admission={int(options['enableAlpsMinimalStaticAdmission'])} "
        f"alps_p3a_exact={int(options['enableAlpsExactReadiness'])}"
        f" alps_p3b_overlap={int(options['enableAlpsExactOverlap'])}"
        f" alps_p4a_traffic={int(options['enableAlpsTrafficControl'])}"
    )
    return options
