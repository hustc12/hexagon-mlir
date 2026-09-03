# Reproducing ALPS results

This directory contains the public, complete-model reproduction workflow. No
script requires a positional argument. Models and configurations run strictly
serially, have no timeout, and are not automatically retried after a failed
case.

Run the complete workflow from any directory:

```bash
/path/to/hexagon-mlir/run_alps.sh
```

The workflow builds the V73 compiler/runtime once and then executes these
stages in order:

| Script | Experiment | Workload |
|---|---|---|
| `00_build_alps.sh` | Toolchain build | Incremental V73 ALPS/Hexagon-MLIR build |
| `01_end_to_end_15_models.sh` | End-to-end | Five configurations, 15 complete FP16 models |
| `02_ablation_selected_models.sh` | Ablation | A0--A4, five complete FP16 models |
| `03_movement_and_traffic.sh` | Data movement | Compiler materialization ledger, DMA telemetry, and sysMon AXI traffic |
| `04_portability_v75_v79.sh` | Portability | V75/V79 dynamic proxies plus complete-graph lowering/codegen/link |

“Build once” means that the compiler and runtime are built once. Each model
and configuration must still undergo DSP code generation because it produces a
different executable graph. Stage 03 reuses Stage 01's raw measurements and
does not compile or execute those 75 cases again.

Results are written under `/tmp/alps_reproduce_<git-sha>` while compact logs,
summaries, and generated model artifacts are moved or synchronized to
`nano:/home/huzq85/2-working/working_set/alps_reproduce_<git-sha>`. A stopped
run can be resumed with the same source commit: passing cases are reused.

The only prerequisites are the provisioned paths documented in
`docs/user-guide.md`, an attached V73 Android device, SSH access to host
`nano`, and the host libraries required by Qualcomm `hexagon-sim`.

Optional environment variables are available for constrained machines or
nonstandard installations, but are not needed in the configured lab setup:

```bash
ALPS_BUILD_JOBS=4 ALPS_RUN_ID=my_run ./run_alps.sh
```
