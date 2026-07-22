//===- OmniFetchDialect.td - OmniFetch Dialect -------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Describes the omni_fetch dialect — the IR backbone of the Omni-Fetch
// prefetching system for Hexagon NPU inference acceleration.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_OMNIFETCH_IR_OMNIFETCH_DIALECT_TD
#define HEXAGON_DIALECT_OMNIFETCH_IR_OMNIFETCH_DIALECT_TD

include "mlir/IR/OpBase.td"

def OmniFetch_Dialect : Dialect {
  let name = "omni_fetch";

  let cppNamespace = "::mlir::omni_fetch";

  let summary = "OmniFetch prefetching dialect for Hexagon NPU";

  let description = [{
    The `omni_fetch` dialect provides the IR primitives for the
    Omni-Fetch prefetching system.  It models three concerns that
    are cleanly separated from each other and from the surrounding
    compute IR:

      1. **Semaphore lifecycle** – `create_sem`, `signal`, `wait` ops
         that map to Hexagon hardware semaphore instructions and enable
         the Virtual Decoupled Access-Execute (V-DAE) execution model.

      2. **Layout-aware prefetch** – `prefetch_in_situ` carries a
         `layout_transform` attribute that instructs the backend to
         perform an in-situ reshape (using `vgather`) while moving data
         from DDR to VTCM, eliminating a separate reshape step.

      3. **Adaptive control hints** – `adaptive_control` allows the
         runtime to tune the prefetch look-ahead window based on PMU
         feedback (AXI stall counters).

    The dialect is designed to lower cleanly to external C runtime
    calls via the `OmniFetchToLLVM` conversion pass, keeping compiler
    and runtime fully decoupled.
  }];

  let extraClassDeclaration = [{
    void registerOperations();
    void registerTypes();
  }];

  let usePropertiesForAttributes = 1;
}

#endif // HEXAGON_DIALECT_OMNIFETCH_IR_OMNIFETCH_DIALECT_TD
