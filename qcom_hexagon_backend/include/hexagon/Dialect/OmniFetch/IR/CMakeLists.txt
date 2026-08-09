# TableGen targets for the OmniFetch dialect.

set(LLVM_TARGET_DEFINITIONS OmniFetchDialect.td)
mlir_tablegen(OmniFetchDialect.h.inc -gen-dialect-decls -dialect=omni_fetch)
mlir_tablegen(OmniFetchDialect.cpp.inc -gen-dialect-defs -dialect=omni_fetch)

set(LLVM_TARGET_DEFINITIONS OmniFetchOps.td)
mlir_tablegen(OmniFetchOps.h.inc -gen-op-decls)
mlir_tablegen(OmniFetchOps.cpp.inc -gen-op-defs)
mlir_tablegen(OmniFetchEnums.h.inc -gen-enum-decls)
mlir_tablegen(OmniFetchEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(OmniFetchAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(OmniFetchAttrs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(OmniFetchTableGen)
