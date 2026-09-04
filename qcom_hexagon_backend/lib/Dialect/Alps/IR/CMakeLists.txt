add_mlir_dialect_library(Alps
  AlpsDialect.cpp
  AlpsOps.cpp
  LayoutAwareMapping.cpp

  DEPENDS
  AlpsTableGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
)
