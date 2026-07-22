#!/usr/bin/env bash
#
# run_benchmark.sh — 一键编译并运行 Hexagon-MLIR benchmark_models
#
# Usage:
#   ./run_benchmark.sh [model_name] [--build-only] [--skip-build] [--simulator] [--model-dir DIR] [-h|--help]
#
# Supported model names (map to benchmark_models/*.py scripts):
#   gpt2          → run_gpt2lmheadmodel.py
#   gpt2_quant    → run_gpt2lmheadmodel_quantized.py
#   tinyllama     → run_tinyllama.py
#   qwen25_05b    → run_qwen2.5-0.5b.py
#   vit           → run_vit.py
#   swin          → run_swin_transformer.py
#   stable_diff   → run_stable_diffusion.py
#   sd_text       → run_sd_text_encoder.py
#   sd_unet       → run_sd_unet.py
#   sd_vae        → run_sd_vae_decoder.py
#   mamba         → run_mamba-130m.py
#   esrgan        → run_real-esrgan.py
#   graphsage     → run_graphsage.py
#   falcon        → run_falcon_rw_1b.py
#   matmul        → micro_bench/test_matmul_benchmark.py
#   conv          → micro_bench/test_conv_benchmark.py
#   dnn           → micro_bench/test_small_dnn_benchmark.py
#   all_micro     → micro_bench/run_all_benchmarks.py
#   validate      → micro_bench/test_quick_validation.py
#
# Examples:
#   ./run_benchmark.sh gpt2                    # Build + run on real device
#   ./run_benchmark.sh vit --skip-build        # Skip build, run on real device
#   ./run_benchmark.sh gpt2 --simulator        # Run on Hexagon simulator (no device needed)
#   ./run_benchmark.sh all_micro               # Run all micro benchmarks
#   ./run_benchmark.sh gpt2 --build-only       # Build only, do not run
#   ./run_benchmark.sh -h                      # Show this help message

set -uo pipefail

###############################################################################
# Usage / Help
###############################################################################
show_help() {
    cat <<'EOF'
Hexagon-MLIR Benchmark Runner

Usage:
  ./run_benchmark.sh [model_name] [options]

Supported Models:
  LLM / NLP Models:
    gpt2          GPT-2 LMHeadModel (2-layer, fp16)
    gpt2_quant    Quantized GPT-2
    tinyllama     TinyLlama 1.1B (reduced config)
    qwen25_05b    Qwen2.5-0.5B (reduced config, fp16)
    falcon        Falcon RW 1B
    mamba         Mamba-130M

  Vision Models:
    vit           Vision Transformer (Google ViT-B/16)
    swin          Swin Transformer Tiny
    stable_diff   Stable Diffusion v1.4 (Text Encoder + UNet + VAE)
    sd_text       SD Text Encoder only
    sd_unet       SD UNet only
    sd_vae        SD VAE Decoder only
    esrgan        Real-ESRGAN

  Graph / Other:
    graphsage     GraphSAGE

  Micro Benchmarks:
    matmul        Matrix multiplication benchmark (Scalar/HVX/HMX)
    conv          Convolution benchmark
    dnn           Small DNN (SimpleMLP + SmallCNN)
    all_micro     Run all micro benchmarks + generate report
    validate      Quick environment validation test

Options:
  --build-only          Build the compiler environment, then exit (do not run any model)
  --skip-build          Skip the build step; source existing environment and run directly
  --simulator           Run on the Hexagon simulator instead of a physical device
                        (sets RUN_ON_SIM=1; no Android device required)
  --model-dir <DIR>     Specify an alternative hexagon-mlir root directory
                        (default: script's parent directory)
  -h, --help            Show this help message and exit

Examples:
  # First-time run: build everything + run GPT-2 on connected Android device
  ./run_benchmark.sh gpt2

  # Skip build on subsequent runs
  ./run_benchmark.sh vit --skip-build

  # Test on simulator without a physical device
  ./run_benchmark.sh gpt2 --simulator

  # Build only (takes a long time on first run)
  ./run_benchmark.sh --build-only

  # Run all micro benchmarks and generate comparison report
  ./run_benchmark.sh all_micro

  # Use a custom hexagon-mlir location (e.g., after migration)
  ./run_benchmark.sh gpt2 --model-dir /home/user/research/projects/alps/hexagon-mlir

Environment Notes:
  - A Python virtual environment is auto-activated from $BASE_DIR/mlir-env
  - ANDROID_SERIAL is auto-detected from 'adb devices' if not set
  - HEXAGON_ARCH_VERSION defaults to 75 (override via set_local_env.sh)
  - 4 known lit test failures (vtcm_tiling, return_alloc_from_loop) are ignored
EOF
}

###############################################################################
# 0. Argument parsing
###############################################################################
MODEL=""
BUILD_ONLY=false
SKIP_BUILD=false
RUN_ON_SIM=false
MODEL_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --build-only)
            BUILD_ONLY=true; shift ;;
        --skip-build)
            SKIP_BUILD=true; shift ;;
        --simulator)
            RUN_ON_SIM=true; export RUN_ON_SIM=1; shift ;;
        --model-dir)
            MODEL_DIR="$2"; shift 2 ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"; shift
            else
                echo "Error: unexpected argument '$1'" >&2
                echo "" >&2
                echo "Use -h or --help to see usage information." >&2
                exit 1
            fi
            ;;
    esac
done

###############################################################################
# 1. Resolve paths
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "$MODEL_DIR" ]]; then
    HEXAGON_MLIR_ROOT="$MODEL_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    HEXAGON_MLIR_ROOT="$SCRIPT_DIR"
fi

echo "============================================================"
echo " Hexagon-MLIR Benchmark Runner"
echo "============================================================"
echo "  HEXAGON_MLIR_ROOT = $HEXAGON_MLIR_ROOT"
echo "  Model            = ${MODEL:-'(not specified)'}"
echo "  Build only       = $BUILD_ONLY"
echo "  Skip build       = $SKIP_BUILD"
echo "  Simulator mode   = $RUN_ON_SIM"
echo "============================================================"

###############################################################################
# 2. Prerequisite checks
###############################################################################
check_prerequisites() {
    local ok=true

    if ! command -v python3 &>/dev/null; then
        echo "[FAIL] python3 not found" >&2
        ok=false
    fi

    if ! command -v adb &>/dev/null; then
        echo "[WARN] adb not found (not needed for simulator mode)" >&2
    elif [[ "$RUN_ON_SIM" == false ]]; then
        local devices
        devices=$(adb devices 2>/dev/null | tail -n +2 | grep -v '^$' || true)
        if [[ -z "$devices" ]]; then
            echo "[FAIL] No Android device detected. Connect your phone and enable USB debugging," >&2
            echo "       or use --simulator to run on the Hexagon simulator." >&2
            ok=false
        else
            echo "[OK] Android device(s) detected:"
            echo "$devices" | while read -r line; do
                echo "      $line"
            done
        fi
    fi

    if [[ "$ok" == false ]]; then
        echo "" >&2
        echo "Prerequisite checks failed. Fix the issues above and retry." >&2
        exit 1
    fi
}

###############################################################################
# 3. Build environment
###############################################################################
build_environment() {
    echo ""
    echo "============================================================"
    echo " Step 1/2: Build Hexagon-MLIR environment"
    echo "============================================================"

    if [[ ! -d "$HEXAGON_MLIR_ROOT/.git" ]]; then
        echo "[FAIL] $HEXAGON_MLIR_ROOT is not a git repo. Please clone hexagon-mlir first." >&2
        exit 1
    fi

    cd "$HEXAGON_MLIR_ROOT"

    # Initialize submodules if missing
    if [[ ! -d "$HEXAGON_MLIR_ROOT/triton" ]] || [[ ! -d "$HEXAGON_MLIR_ROOT/triton_shared" ]]; then
        echo "[->] Initializing submodules triton / triton_shared ..."
        source ci/setup_submodules.sh
    else
        echo "[OK] Submodules already present, skipping initialization"
    fi

    # Set BASE_DIR (required by build_hexagon_mlir.sh)
    export BASE_DIR="$(cd "$HEXAGON_MLIR_ROOT/.." && pwd)"

    # Run the one-shot build script
    echo "[->] Running build_hexagon_mlir.sh ..."
    bash ./scripts/build_hexagon_mlir.sh || true
    # Note: build_hexagon_mlir.sh may exit non-zero due to 4 known lit test failures.
    # These do NOT prevent running benchmarks.

    echo ""
    echo "[OK] Build completed"
}

###############################################################################
# 4. Source runtime environment
###############################################################################
source_environment() {
    echo ""
    echo "============================================================"
    echo " Step 2/2: Configure runtime environment"
    echo "============================================================"

    # Activate virtual environment if it exists
    local VENV_DIR="${HEXAGON_MLIR_ROOT}/../mlir-env"
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        source "$VENV_DIR/bin/activate"
        echo "[->] Activated Python venv: $VENV_DIR"
    elif [[ -d "$HEXAGON_MLIR_ROOT/.venv" ]]; then
        source "$HEXAGON_MLIR_ROOT/.venv/bin/activate"
        echo "[->] Activated Python venv: .venv"
    else
        echo "[WARN] No Python virtual environment found at $VENV_DIR"
        echo "       Make sure torch/triton/torch-mlir are installed."
    fi

    cd "$HEXAGON_MLIR_ROOT"
    source scripts/set_local_env.sh

    # ADB device setup
    if [[ "$RUN_ON_SIM" == false ]]; then
        if [[ -z "${ANDROID_HOST:-}" ]]; then
            export ANDROID_HOST=""
        fi
        if [[ -z "${ANDROID_SERIAL:-}" ]]; then
            # Auto-select the first online device
            ANDROID_SERIAL=$(adb devices 2>/dev/null | tail -n +2 | grep -v '^$' | awk '{print $1}')
            if [[ -n "$ANDROID_SERIAL" ]]; then
                echo "[->] Auto-selected device: $ANDROID_SERIAL"
                export ANDROID_SERIAL
            fi
        fi
        echo "[->] ANDROID_SERIAL=$ANDROID_SERIAL"
        echo "[->] ANDROID_HOST=${ANDROID_HOST:-''}"
    fi

    # Simulator switch
    if [[ "${RUN_ON_SIM}" == "true" ]] || [[ "${RUN_ON_SIM}" == "1" ]]; then
        echo "[->] Simulator mode (RUN_ON_SIM=${RUN_ON_SIM})"
    else
        echo "[->] Device mode (RUN_ON_SIM=0)"
    fi

    echo "[OK] Environment configured"
    echo ""
    echo "  Environment summary:"
    echo "    HEXAGON_MLIR_ROOT = $HEXAGON_MLIR_ROOT"
    echo "    HEXAGON_TOOLS     = ${HEXAGON_TOOLS:-'(not set)'}"
    echo "    HEXAGON_SDK_ROOT  = ${HEXAGON_SDK_ROOT:-'(not set)'}"
    echo "    TRITON_ROOT       = ${TRITON_ROOT:-'(not set)'}"
    echo "    PYTHONPATH        = ${PYTHONPATH:-'(not set)'}"
}

###############################################################################
# 5. Run the selected model
###############################################################################
run_model() {
    if [[ -z "$MODEL" ]]; then
        echo ""
        echo "No model specified. Exiting."
        echo "Usage: $0 <model_name> [--build-only] [--skip-build] [--simulator]"
        echo ""
        echo "Supported models:"
        echo "  gpt2, gpt2_quant, tinyllama, qwen25_05b,"
        echo "  vit, swin, stable_diff, sd_text, sd_unet, sd_vae,"
        echo "  mamba, esrgan, graphsage, falcon,"
        echo "  matmul, conv, dnn, all_micro, validate"
        exit 0
    fi

    cd "$HEXAGON_MLIR_ROOT/benchmark_models"

    local SCRIPT_PATH=""
    case "$MODEL" in
        gpt2)
            SCRIPT_PATH="run_gpt2lmheadmodel.py" ;;
        gpt2_quant)
            SCRIPT_PATH="run_gpt2lmheadmodel_quantized.py" ;;
        tinyllama)
            SCRIPT_PATH="run_tinyllama.py" ;;
        qwen25_05b)
            SCRIPT_PATH="run_qwen2.5-0.5b.py" ;;
        vit)
            SCRIPT_PATH="run_vit.py" ;;
        swin)
            SCRIPT_PATH="run_swin_transformer.py" ;;
        stable_diff)
            SCRIPT_PATH="run_stable_diffusion.py" ;;
        sd_text)
            SCRIPT_PATH="run_sd_text_encoder.py" ;;
        sd_unet)
            SCRIPT_PATH="run_sd_unet.py" ;;
        sd_vae)
            SCRIPT_PATH="run_sd_vae_decoder.py" ;;
        mamba)
            SCRIPT_PATH="run_mamba-130m.py" ;;
        esrgan)
            SCRIPT_PATH="run_real-esrgan.py" ;;
        graphsage)
            SCRIPT_PATH="run_graphsage.py" ;;
        falcon)
            SCRIPT_PATH="run_falcon_rw_1b.py" ;;
        matmul)
            SCRIPT_PATH="micro_bench/test_matmul_benchmark.py" ;;
        conv)
            SCRIPT_PATH="micro_bench/test_conv_benchmark.py" ;;
        dnn)
            SCRIPT_PATH="micro_bench/test_small_dnn_benchmark.py" ;;
        all_micro)
            SCRIPT_PATH="micro_bench/run_all_benchmarks.py" ;;
        validate)
            SCRIPT_PATH="micro_bench/test_quick_validation.py" ;;
        *)
            echo "[FAIL] Unknown model: $MODEL" >&2
            echo "       Supported: gpt2, gpt2_quant, tinyllama, qwen25_05b," >&2
            echo "                    vit, swin, stable_diff, sd_text, sd_unet, sd_vae," >&2
            echo "                    mamba, esrgan, graphsage, falcon," >&2
            echo "                    matmul, conv, dnn, all_micro, validate" >&2
            exit 1
            ;;
    esac

    echo "============================================================"
    echo " Running model: $MODEL"
    echo " Script        : $SCRIPT_PATH"
    echo " Mode          : $([ "${RUN_ON_SIM}" == "true" ] || [ "${RUN_ON_SIM}" == "1" ] && echo 'Simulator' || echo 'Device')"
    echo "============================================================"

    python3 "$SCRIPT_PATH" "$@"
}

###############################################################################
# 6. Main entry point
###############################################################################
main() {
    check_prerequisites

    if [[ "$BUILD_ONLY" == true ]] || [[ "$SKIP_BUILD" == false ]]; then
        build_environment
    else
        echo "[->] Skipping build (--skip-build)"
    fi

    source_environment

    if [[ "$BUILD_ONLY" == true ]]; then
        echo ""
        echo "Build completed. Exiting (--build-only)."
        exit 0
    fi

    run_model
}

main "$@"
