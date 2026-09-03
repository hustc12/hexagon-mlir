# OmniFetch overlay for Qualcomm Hexagon-MLIR

The `baseline_5_upstream_v73` stabilization branch is rooted directly at Qualcomm upstream commit
`9b4b8fcea2b93c801b5de784ee750ca9350d504f`. The branch
`upstream-snapshot-20260808` preserves that unmodified snapshot.

The OmniFetch implementation is maintained as an overlay rather than a forked
copy of Triton or triton-shared:

- custom dialect, lowering, runtime, passes, tests, model runners, and scripts
  are owned by this repository;
- Triton and triton-shared are cloned at the revisions selected by Qualcomm's
  `ci/setup_submodules.sh`, are intentionally untracked by the parent repo, and
  receive Qualcomm's official patches plus a tracked, reversible Hexagon-only
  build patch from `omnifetch/patches/triton`;
- changes to upstream files are restricted to dialect/pass registration,
  pipeline option plumbing, semantic K/V metadata preservation, HexKL tile
  decomposition hooks, and runtime linking;
- the legacy repository remains at `../hexagon-mlir-legacy` and is never used
  as a build dependency.

## Updating upstream

1. Fetch Qualcomm `main` and advance a new clean snapshot branch.
2. Rebase `omnifetch-overlay` onto that snapshot.
3. Resolve only the small integration surface recorded in `manifest.txt`.
4. Run `scripts/script_legacy/verify_omnifetch_overlay.sh` to detect accidental changes to
   Qualcomm-owned files.
5. Run `scripts/script_legacy/build_omnifetch_upstream.sh`.
6. Run the serial no-timeout model matrix.

Do not copy the legacy `triton`, `triton_shared`, build trees, MLIR bytecode,
shared objects, logs, or model outputs into this repository.

The pinned LLVM revision also needs three later official Hexagon fixes for
large v73 full models. Exact upstream patches are stored in `patches/llvm/`
and applied idempotently by `scripts/script_legacy/apply_llvm_hexagon_fixes.sh`: truncating
DoubleRegs/IntRegs COPY support, aligned-frame AP prologue ordering, and AP
live-in tracking. These are compiler correctness prerequisites, not OmniFetch
performance improvements.

The Hexagon-only dependency patch makes Triton's built-in NVIDIA and AMD
backends optional. OmniFetch sets `TRITON_IN_TREE_BACKENDS` to empty and builds
LLVM targets `X86;Hexagon`; this avoids compiling AMDGPU/NVPTX while preserving
the external `triton_shared` and Qualcomm Hexagon plugins. The patch is applied
idempotently by `scripts/script_legacy/apply_hexagon_only_triton_patch.sh` and is deliberately
kept outside the untracked Triton checkout so a future upstream refresh can
fail cleanly instead of silently carrying stale dependency edits.

Triton is still a build-time/compiler-host dependency: the existing model
runners import `triton._C.libtriton.qcom_hexagon_backend`, and Qualcomm's C++
bindings are registered with `add_triton_plugin`. The models do not use the
NVIDIA/AMD code generators, so those in-tree backends are disabled. Removing
Triton entirely would require a separate standalone pybind/launcher project and
is intentionally outside this migration.

## Reproducible full-model run

After the SDK/tool paths documented by the upstream project are installed, the
entire compile, test, and 15-model HVX/HexKL/HexKL+OmniFetch matrix is launched
with one command:

```bash
BUILD_JOBS=4 scripts/script_legacy/run_all_models_no_timeout.sh
```

Model/configuration pairs run strictly serially and without a host-side
deadline. Logs and a resumable CSV are written under
`/tmp/omnifetch-upstream-full-models` by default. Re-running the command skips
rows already recorded as `PASS`; `--force` deliberately repeats them.
