# TableGen targets for the Alps dialect.

set(LLVM_TARGET_DEFINITIONS AlpsDialect.td)
mlir_tablegen(AlpsDialect.h.inc -gen-dialect-decls -dialect=alps)
mlir_tablegen(AlpsDialect.cpp.inc -gen-dialect-defs -dialect=alps)

set(LLVM_TARGET_DEFINITIONS AlpsOps.td)
mlir_tablegen(AlpsOps.h.inc -gen-op-decls)
mlir_tablegen(AlpsOps.cpp.inc -gen-op-defs)
mlir_tablegen(AlpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(AlpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(AlpsAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AlpsAttrs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(AlpsTableGen)
