# Checkout consolidation record (2026-09-03)

This directory records the one-time consolidation of three similarly named
source trees into the sole active checkout `hexagon-mlir`.

## Source identities

| Former path | Git identity at consolidation | Disposition |
|---|---|---|
| `hexagon-mlir` | `alps_release_cleanup`, `e9c21aa31ffe4a0b62ce0c03449ac4d149547437` | Retained as the canonical checkout |
| `hexagon-mlir-legacy` | `baseline_5`, `eabb4a4749865af8e104a856d4189749bbfbe4e7` | Branch remains on `hustc12`; dirty tracked changes and unique small artifacts are archived here |
| `hexagon-mlir-native` | detached upstream `9b4b8fcea2b93c801b5de784ee750ca9350d504f` | Temporary linked worktree removed after its dirty patch was archived |

The canonical repository retains remotes `origin` (Qualcomm upstream) and
`hustc12` (project fork). The local-path remote named `legacy` was removed
because its target checkout no longer exists.

## Preserved material

- `patches/hexagon-mlir-legacy-tracked.patch` is the complete tracked diff from
  the legacy checkout, excluding dirty submodule worktrees.
- `patches/hexagon-mlir-native-tracked.patch` is the complete tracked diff from
  the temporary native worktree.
- `legacy_untracked/engineering_notes` contains only notes that did not already
  exist in the canonical `docs/alps` or `archive/engineering_notes` trees.
- `legacy_untracked/debug_tools` retains the unique source-level probes.

Generated logs, plots, model IR, and the tagged items-1--7 bundle were moved,
not copied, to
`nano:/home/huzq85/2-working/working_set/checkout_consolidation_20260903`.
They are deliberately outside Git.

The native patch's buffer-results-to-out-parameters support is already present
in the canonical implementation (`enableBufferResultsToOutParams`, return/arg
shape inspection, launcher wiring, conversion, and translation). It was not
reapplied. The legacy patch contains earlier Debug-only probes, an old stack
size adjustment, and a global mutable HexKL padding-pressure heuristic. These
are retained for reference but were not allowed to overwrite the newer full
model and admission-control implementation.

## Intentionally excluded generated data

The legacy checkout contained roughly 1.6 GiB of generated
`benchmark_models/*_f16matmul.mlir` files plus reproducible `triton` and
`triton_shared` worktrees. These are neither source-of-truth inputs nor release
artifacts and were deliberately excluded rather than transferred. Their associated historical branch
commits remain recoverable from `hustc12/baseline_5`,
`hustc12/hvx_test_4`, and `hustc12/alps_improve_3`.

## LLVM directory convention

The repository's original and now canonical convention is
`../LLVM_DIR/llvm-project`. During upstream-v73 migration the active checkout
was temporarily named `LLVM_DIR_upstream` to distinguish it from an older
patched tree. The old tree is gone, so the active 14 GiB source/build directory
was renamed back to `LLVM_DIR`. A relative compatibility symlink
`LLVM_DIR_upstream -> LLVM_DIR` is retained for the current build cache because
CMake and Ninja files embed the old absolute path. New scripts and clean builds
use `LLVM_DIR`; the symlink can be removed after the next clean LLVM rebuild.
