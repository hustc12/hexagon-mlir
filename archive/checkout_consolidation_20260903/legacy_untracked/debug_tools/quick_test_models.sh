#!/bin/bash
# Quick test key models with minimal iterations

echo "=== Testing Key Full Models (Baseline) ==="
echo ""

# Test 1: DINOv2-small
echo "[1/5] DINOv2-small (ViT, image=518, M=1370)"
timeout 300 python benchmark_models/run_dinov2-small.py --backend-profile legacy-scalar --device-iterations 1 2>&1 | grep -E "PerfP50|Compare|Error|exit 13" || echo "FAILED/TIMEOUT"
echo ""

# Test 2: DEiT-small  
echo "[2/5] DEiT-small (ViT)"
timeout 300 python benchmark_models/run_deit-small.py --backend-profile legacy-scalar --device-iterations 1 2>&1 | grep -E "PerfP50|Compare|Error|exit 13" || echo "FAILED/TIMEOUT"
echo ""

# Test 3: GPT-2 (if it finishes compiling)
echo "[3/5] GPT-2 (LLM, seq_len=128)"
timeout 600 python benchmark_models/run_gpt2lmheadmodel.py --seq-len 128 --device-iterations 1 2>&1 | grep -E "PerfP50|Compare|Error|exit 13" || echo "FAILED/TIMEOUT"
echo ""

# Test 4: Segformer
echo "[4/5] Segformer-MIT-B0 (Segmentation)"
timeout 300 python benchmark_models/run_segformer-mit-b0.py --backend-profile legacy-scalar --device-iterations 1 2>&1 | grep -E "PerfP50|Compare|Error|exit 13" || echo "FAILED/TIMEOUT"
echo ""

# Test 5: Real-ESRGAN  
echo "[5/5] Real-ESRGAN (Image enhancement)"
timeout 300 python benchmark_models/run_real-esrgan.py --backend-profile legacy-scalar --device-iterations 1 2>&1 | grep -E "PerfP50|Compare|Error|exit 13" || echo "FAILED/TIMEOUT"
echo ""

echo "=== Quick Test Complete ==="
