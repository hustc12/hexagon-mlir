#!/bin/bash
# Systematically test all full models baseline (no HMX, no M-padding)
# Goal: establish which models can run successfully

set +e  # Continue on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source /home/huzq85/2-working/hexagon_npu/mlir-env/bin/activate

RESULTS_FILE="full_model_baseline_results.txt"
echo "=== Full Model Baseline Testing ($(date)) ===" | tee "$RESULTS_FILE"
echo "Profile: legacy-scalar (no HMX, no M-padding)" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# List of full models to test (vision transformers first, then LLMs)
declare -a MODELS=(
    # Vision Transformers (faster to compile)
    "run_dinov2-small.py --backend-profile legacy-scalar --device-iterations 1"
    "run_deit-small.py --backend-profile legacy-scalar --device-iterations 1"
    "run_beit-base.py --backend-profile legacy-scalar --device-iterations 1"
    "run_swin_transformer.py --backend-profile legacy-scalar --device-iterations 1"
    "run_segformer-mit-b0.py --backend-profile legacy-scalar --device-iterations 1"
    
    # LLMs (slower to compile, may timeout)
    "run_gpt2lmheadmodel.py --seq-len 128 --device-iterations 1"
    "run_tinyllama.py --seq-len 128 --device-iterations 1"
    "run_mamba-130m.py --seq-len 128 --device-iterations 1"
    "run_qwen2.5-0.5b.py --seq-len 128 --device-iterations 1"
)

for model_cmd in "${MODELS[@]}"; do
    model_name=$(echo "$model_cmd" | awk '{print $1}' | sed 's/run_//; s/.py//')
    echo "======================================" | tee -a "$RESULTS_FILE"
    echo "Testing: $model_name" | tee -a "$RESULTS_FILE"
    echo "Command: python benchmark_models/$model_cmd" | tee -a "$RESULTS_FILE"
    echo "Start: $(date +%H:%M:%S)" | tee -a "$RESULTS_FILE"
    
    # Timeout after 15 minutes
    eval timeout 900 python "benchmark_models/$model_cmd" 2>&1 | tee /tmp/model_test_output.txt
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "❌ TIMEOUT (>15 min)" | tee -a "$RESULTS_FILE"
    elif [ $exit_code -ne 0 ]; then
        # Check for specific errors
        if grep -q "exit 13" /tmp/model_test_output.txt; then
            echo "❌ FAILED: Exit 13 (AEE_EBADSTATE)" | tee -a "$RESULTS_FILE"
        elif grep -q "Error" /tmp/model_test_output.txt; then
            echo "❌ FAILED: $(grep -m1 'Error' /tmp/model_test_output.txt)" | tee -a "$RESULTS_FILE"
        else
            echo "❌ FAILED: exit code $exit_code" | tee -a "$RESULTS_FILE"
        fi
    else
        # Success - extract performance
        perf=$(grep "PerfP50" /tmp/model_test_output.txt | tail -1)
        compare=$(grep "Compare" /tmp/model_test_output.txt | tail -1)
        if [ -n "$perf" ]; then
            echo "✅ SUCCESS: $perf" | tee -a "$RESULTS_FILE"
            echo "   $compare" | tee -a "$RESULTS_FILE"
        else
            echo "⚠️  COMPILED but no performance data" | tee -a "$RESULTS_FILE"
        fi
    fi
    
    echo "End: $(date +%H:%M:%S)" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
done

echo "======================================" | tee -a "$RESULTS_FILE"
echo "Testing complete: $(date)" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
