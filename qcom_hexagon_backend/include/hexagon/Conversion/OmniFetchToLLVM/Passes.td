//===- Passes.td - OmniFetch to LLVM passes ----------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Converts omni_fetch dialect ops to LLVM external function calls.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_OMNIFETCHTOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_OMNIFETCHTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def OmniFetchToLLVM : Pass<"omni-fetch-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower omni_fetch dialect ops to LLVM runtime calls";
  let description = [{
    Lowers all `omni_fetch` dialect operations to calls to the
    Hexagon device runtime (`__omni_fetch_*` symbols).

    Mapping:
    - `omni_fetch.create_sem`         → `__omni_fetch_create_sem() -> i32`
    - `omni_fetch.signal(%sem)`       → `__omni_fetch_signal(sem: i32)`
    - `omni_fetch.wait(%sem)`         → `__omni_fetch_wait(sem: i32)`
    - `omni_fetch.prefetch_in_situ`   → `__omni_fetch_prefetch_insitu(
                                            src_ptr, dest_ptr, elem_bytes,
                                            num_elems, layout_kind,
                                            lookahead, index_map_ptr)`
    - `omni_fetch.adaptive_control`   → `__omni_fetch_update_distance(cur: i32) -> i32`
  }];

  let constructor = "mlir::omni_fetch::createOmniFetchToLLVMPass()";

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // HEXAGON_CONVERSION_OMNIFETCHTOLLVM_PASSES_TD
