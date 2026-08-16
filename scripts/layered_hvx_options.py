"""Matched backend options for complete-model layered HVX experiments."""

from __future__ import annotations

import argparse


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
        help="Enable isolated OmniFetch item 7; items 1-6 remain disabled.",
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
    if args.prefetch_baseline != "none" and item7:
        raise ValueError("external prefetch baselines cannot be combined with item7")
    if item7 and not (args.disable_layout_aware and args.disable_omnifetch_adaptive):
        raise ValueError(
            "item7-only requires --disable-layout-aware and "
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

    options["enablePrefetch"] = item7
    options["enableOmniFetchLayoutAware"] = False
    options["enableOmniFetchVDAE"] = False
    options["enableOmniFetchAdaptive"] = False
    options["enableOmniFetchPersistentWhCache"] = False
    options["enableOmniFetchTwoDimPipeline"] = False
    options["enableOmniFetchVtcmColoring"] = False
    options["enableOmniFetchKvCachePrefetch"] = item7
    options["enableOmniFetchDmaToVtcm"] = False
    options["enableHexagonmemCopyToDMA"] = False

    print(
        "[LayeredBackendConfig] "
        f"vectorization=1 hexkl={int(options['enableHexKL'])} "
        "uniform_fp16=1 vtcm_tiling=0 conversion_to_fp16=0 "
        f"prefetch_baseline={args.prefetch_baseline} "
        f"item7_only={int(item7)}"
    )
    return options
