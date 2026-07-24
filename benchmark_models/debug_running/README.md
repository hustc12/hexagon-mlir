# Debug runners (reduced topology)

These scripts override the parent harness `customize_*` / `load_*` hooks for
fast DSP smoke and Phase-4 A/B. **Do not use them for fair full-structure numbers.**

```bash
python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32
python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32 --enable-hexkl
python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32 --enable-hexkl --enable-omnifetch-vdae
```

Full published topology: `benchmark_models/run_*.py` (parent directory).
