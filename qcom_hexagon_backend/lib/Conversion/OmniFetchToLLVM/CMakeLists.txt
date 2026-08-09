add_mlir_conversion_library(OmniFetchToLLVM
  OmniFetchToLLVMPass.cpp

  DEPENDS
  OmniFetchToLLVMConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
  MLIRTransforms
  OmniFetch
)
