#!/bin/bash
# ===- run_benchmarks.sh ----------------------------------------------------===
#
# Convenience script to run Hexagon NPU benchmarks
#
# ===------------------------------------------------------------------------===

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Hexagon NPU Benchmark Suite${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Function to run a benchmark
run_benchmark() {
    local name=$1
    local script=$2
    
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Running: $name${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    
    if python3 "$SCRIPT_DIR/$script"; then
        echo -e "${GREEN}✓ $name completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ $name failed${NC}"
        return 1
    fi
}

# Parse command line arguments
MODE=${1:-all}

case $MODE in
    validate|v)
        echo "Running validation test..."
        run_benchmark "Validation Test" "test_quick_validation.py"
        ;;
    
    matmul|m)
        echo "Running matrix multiplication benchmark..."
        run_benchmark "Matrix Multiplication Benchmark" "test_matmul_benchmark.py"
        ;;
    
    conv|c)
        echo "Running convolution benchmark..."
        run_benchmark "Convolution Benchmark" "test_conv_benchmark.py"
        ;;
    
    dnn|d)
        echo "Running DNN model benchmark..."
        run_benchmark "DNN Model Benchmark" "test_small_dnn_benchmark.py"
        ;;
    
    all|a)
        echo "Running all benchmarks..."
        echo ""
        echo -e "${YELLOW}This may take several minutes to complete.${NC}"
        echo ""
        
        # First run validation
        if ! run_benchmark "Validation Test" "test_quick_validation.py"; then
            echo ""
            echo -e "${YELLOW}Warning: Validation test failed.${NC}"
            echo -e "${YELLOW}Continuing with benchmarks anyway...${NC}"
        fi
        
        # Run all benchmarks
        run_benchmark "All Benchmarks" "run_all_benchmarks.py"
        ;;
    
    help|h|--help|-h)
        echo "Usage: $0 [MODE]"
        echo ""
        echo "Available modes:"
        echo "  validate, v    - Run quick validation test"
        echo "  matmul, m      - Run matrix multiplication benchmark"
        echo "  conv, c        - Run convolution benchmark"
        echo "  dnn, d         - Run DNN model benchmark"
        echo "  all, a         - Run all benchmarks (default)"
        echo "  help, h        - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 validate    # Quick validation"
        echo "  $0 matmul      # Only matrix multiplication"
        echo "  $0 all         # Run everything"
        echo ""
        exit 0
        ;;
    
    *)
        echo -e "${RED}Error: Unknown mode '$MODE'${NC}"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Benchmark complete!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Results available in:"
echo "  - matmul_benchmark_results.json"
echo "  - conv_benchmark_results.json"
echo "  - dnn_benchmark_results.json"
echo "  - benchmark_comparison_report.md"
echo ""
