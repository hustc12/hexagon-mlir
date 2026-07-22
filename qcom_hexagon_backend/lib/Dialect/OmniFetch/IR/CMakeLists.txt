add_mlir_dialect_library(OmniFetch
  OmniFetchDialect.cpp
  OmniFetchOps.cpp
  LayoutAwareMapping.cpp

  DEPENDS
  OmniFetchTableGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
)
