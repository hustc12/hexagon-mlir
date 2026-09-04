//===- Passes.td - Alps to LLVM passes ----------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Converts alps dialect ops to LLVM external function calls.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_ALPSTOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_ALPSTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def AlpsToLLVM : Pass<"alps-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower alps dialect ops to LLVM runtime calls";
  let description = [{
    Lowers all `alps` dialect operations to calls to the
    Hexagon device runtime (`__alps_*` symbols).

    Mapping:
    - `alps.create_sem`         → `__alps_create_sem() -> i32`
    - `alps.signal(%sem)`       → `__alps_signal(sem: i32)`
    - `alps.wait(%sem)`         → `__alps_wait(sem: i32)`
    - `alps.invocation_*`       → generation-safe P3 context calls
    - `alps.descriptor_*`       → exact-readiness descriptor calls
    - `alps.prefetch_in_situ`   → `__alps_prefetch_insitu(
                                            src_ptr, dest_ptr, elem_bytes,
                                            num_elems, layout_kind,
                                            lookahead, index_map_ptr)`
    - `alps.adaptive_control`   → `__alps_update_distance(cur: i32) -> i32`
  }];

  let options = [
    Option<"enableDualThreadDae", "enable-dual-thread-dae", "bool",
           /*default=*/"false",
           "Emit __alps_set_dual_thread_dae(1) so deferred WH runs on scout.">,
  ];

  let constructor = "mlir::alps::createAlpsToLLVMPass()";

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // HEXAGON_CONVERSION_ALPSTOLLVM_PASSES_TD
