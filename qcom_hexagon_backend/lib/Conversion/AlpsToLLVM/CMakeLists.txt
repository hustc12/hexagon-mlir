add_mlir_conversion_library(AlpsToLLVM
  AlpsToLLVMPass.cpp

  DEPENDS
  AlpsToLLVMConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
  MLIRTransforms
  Alps
)
