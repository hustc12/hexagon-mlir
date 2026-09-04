//===- AlpsDialect.td - Alps Dialect -------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Describes the alps dialect — the IR backbone of the ALPS
// prefetching system for Hexagon NPU inference acceleration.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_ALPS_IR_ALPS_DIALECT_TD
#define HEXAGON_DIALECT_ALPS_IR_ALPS_DIALECT_TD

include "mlir/IR/OpBase.td"

def Alps_Dialect : Dialect {
  let name = "alps";

  let cppNamespace = "::mlir::alps";

  let summary = "Alps prefetching dialect for Hexagon NPU";

  let description = [{
    The `alps` dialect provides the IR primitives for the
    ALPS prefetching system.  It models three concerns that
    are cleanly separated from each other and from the surrounding
    compute IR:

      1. **Semaphore lifecycle** – `create_sem`, `signal`, `wait` ops
         that map to Hexagon hardware semaphore instructions and enable
         the Vectorized Decoupled Access-Execute (V-DAE) execution model.

      2. **Layout-aware prefetch** – `prefetch_in_situ` carries a
         `layout_transform` attribute that instructs the backend to
         perform an in-situ reshape (using `vgather`) while moving data
         from DDR to VTCM, eliminating a separate reshape step.

      3. **Adaptive control hints** – `adaptive_control` allows the
         runtime to tune the prefetch look-ahead window based on PMU
         feedback (AXI stall counters).

    The dialect is designed to lower cleanly to external C runtime
    calls via the `AlpsToLLVM` conversion pass, keeping compiler
    and runtime fully decoupled.
  }];

  let extraClassDeclaration = [{
    void registerOperations();
    void registerTypes();
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif // HEXAGON_DIALECT_ALPS_IR_ALPS_DIALECT_TD
