# Hexagon-MLIR

Hexagon‑MLIR is an open‑source AI compiler stack that lets you easily compile and run Triton kernels and PyTorch models on Qualcomm Hexagon Neural Processing Units (NPUs).

This initiative complements our commercial toolchains by exploring an open‑source MLIR‑based compilation stack, giving developers a path to advance AI compilation capabilities through a more flexible and transparent approach.

## Table of Contents

- [Features](#features)
- [Documentation](#documentation)
- [License](#license)
- [Support](#support)

## Features
- **Triton Kernel Compilation & Execution**: Compile Triton kernels and execute on Hexagon NPU targets
- **PyTorch Model Compilation & Execution**: Compile PyTorch models and execute on Hexagon NPU targets
- **Performance Optimization**: Leverage Hexagon-specific features for maximum performance, including:
  - **Multi-threading**: Hexagon-optimized parallel execution of operations for improved performance
  - **Vector Processing**: Optimized code generation targeting Hexagon Vector eXtensions (HVX) units
  - **TCM Utilization**: Leverage Tightly Coupled Memory (TCM) for reduced memory latency
  - **DMA Optimization**: Efficient DMA transfers between DDR and TCM memory spaces
  - **Matrix Processing (experimental)**: Leverage Qualcomm's Hexagon Kernel Library for matrix multiplication
- **IR Inspection**: Inspect and analyze IR lowering passes, helping you understand how your code is optimized

## Documentation
- 📖 [User Guide](docs/user-guide.md) - For instructions on how to download, setup our compiler and start running Triton kernels or PyTorch models on Hexagon NPUs
- 🎓 [Tutorials](docs/tutorials/README.md) - A set of tutorials on Triton kernels and PyTorch models
- ❓ [FAQ](docs/faq.md) - Frequently asked questions
- 🏗️ [Developer Guide](docs/developer-guide.md) - Insights on how to develop, debug and profile using our compiler toolchain for Triton kernels and PyTorch models
- 🧪 [ALPS Reproduction](scripts/script_release/README.md) - One-command complete-model evaluation, ablation, traffic audit, and V75/V79 portability workflow
- 📊 [ALPS Results and Analysis](docs/alps/README.md) - Canonical complete-model data, portability evidence, and bottleneck analysis

## ALPS evaluation

After provisioning the dependencies described in the user guide, the complete
release evaluation has a single zero-argument entry point:

```bash
./run_alps.sh
```

It builds the V73 toolchain once, then runs the 15-model end-to-end matrix, the
selected-model ablation, the movement/traffic audit, and the V75/V79
portability study strictly serially.

## License

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

---

<div align="center">
  <sub>Built with ❤️ by the  Hexagon-MLIR Team</sub>
</div>
