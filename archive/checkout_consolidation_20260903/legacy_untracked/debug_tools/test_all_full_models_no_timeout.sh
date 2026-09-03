#!/bin/bash
# Test all full models with NO TIMEOUT
# Goal: Establish baseline for which models can complete successfully

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source /home/huzq85/2-working/hexagon_npu/mlir-env/bin/activate

RESULTS_FILE="full_model_results_no_timeout.txt"
echo "=== Full Model Testing - No Timeout ($(date)) ===" | tee "$RESULTS_FILE"
echo "Profile: legacy-scalar (baseline, no HMX)" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Test models one by one
declare -a MODELS=(
    "DINOv2-small:run_dinov2-small.py --backend-profile legacy-scalar --device-iterations 1"
    "DEiT-small:run_deit-small.py --backend-profile legacy-scalar --device-iterations 1"
    "BEiT-base:run_beit-base.py --backend-profile legacy-scalar --device-iterations 1"
    "Swin-Trans:run_swin_transformer.py --backend-profile legacy-scalar --device-iterations 1"
    "Segformer:run_segformer-mit-b0.py --backend-profile legacy-scalar --device-iterations 1"
    "ViT:run_vit.py --backend-profile legacy-scalar --device-iterations 1"
    "Real-ESRGAN:run_real-esrgan.py --backend-profile legacy-scalar --device-iterations 1"
)

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_cmd <<< "$model_spec"
    
    echo "======================================" | tee -a "$RESULTS_FILE"
    echo "Testing: $model_name" | tee -a "$RESULTS_FILE"
    echo "Command: python benchmark_models/$model_cmd" | tee -a "$RESULTS_FILE"
    echo "Start: $(date +%Y-%m-%d_%H:%M:%S)" | tee -a "$RESULTS_FILE"
    
    # NO TIMEOUT - let it run as long as needed
    python "benchmark_models/$model_cmd" > /tmp/model_test_${model_name}.log 2>&1
    exit_code=$?
    
    echo "End: $(date +%Y-%m-%d_%H:%M:%S)" | tee -a "$RESULTS_FILE"
    
    if [ $exit_code -ne 0 ]; then
        if grep -q "exit 13" /tmp/model_test_${model_name}.log; then
            echo "❌ FAILED: Exit 13 (AEE_EBADSTATE)" | tee -a "$RESULTS_FILE"
        elif grep -q "Error" /tmp/model_test_${model_name}.log; then
            error_line=$(grep -m1 "Error" /tmp/model_test_${model_name}.log)
            echo "❌ FAILED: $error_line" | tee -a "$RESULTS_FILE"
        else
            echo "❌ FAILED: exit code $exit_code" | tee -a "$RESULTS_FILE"
        fi
    else
        perf=$(grep "PerfP50" /tmp/model_test_${model_name}.log | tail -1)
        compare=$(grep "Compare" /tmp/model_test_${model_name}.log | tail -1)
        if [ -n "$perf" ]; then
            echo "✅ SUCCESS: $perf" | tee -a "$RESULTS_FILE"
            echo "   $compare" | tee -a "$RESULTS_FILE"
        else
            echo "⚠️  COMPILED but no performance data" | tee -a "$RESULTS_FILE"
        fi
    fi
    
    # Save full log
    tail -50 /tmp/model_test_${model_name}.log >> "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
done

echo "======================================" | tee -a "$RESULTS_FILE"
echo "Testing complete: $(date)" | tee -a "$RESULTS_FILE"
