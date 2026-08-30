// RUN: linalg-hexagon-opt %s --pass-pipeline='builtin.module(func.func(alps-hvx-widening-conv))' | FileCheck %s

module {
  func.func @mixed_f16_f32_conv(
      %input: memref<1x3x230x230xf16>,
      %filter: memref<32x3x7x7xf16>,
      %output: memref<1x32x56x56xf32>) {
    linalg.conv_2d_nchw_fchw
        {dilations = dense<1> : vector<2xi64>,
         strides = dense<4> : vector<2xi64>}
        ins(%input, %filter : memref<1x3x230x230xf16>,
                              memref<32x3x7x7xf16>)
        outs(%output : memref<1x32x56x56xf32>)
    return
  }

  func.func @mixed_f16_f32_conv1d(
      %input: memref<1x80x3002xf16>,
      %filter: memref<384x80x3xf16>,
      %output: memref<1x384x1500xf32>) {
    linalg.conv_1d_ncw_fcw
        {dilations = dense<1> : vector<1xi64>,
         strides = dense<2> : vector<1xi64>}
        ins(%input, %filter : memref<1x80x3002xf16>,
                              memref<384x80x3xf16>)
        outs(%output : memref<1x384x1500xf32>)
    return
  }

  // Small non-overlapping patchification belongs to direct patch formation,
  // not the 64-lane sliding-convolution schedule.
  func.func @small_nonoverlap_patchify(
      %input: memref<1x3x224x224xf16>,
      %filter: memref<384x3x14x14xf16>,
      %output: memref<1x384x16x16xf32>) {
    linalg.conv_2d_nchw_fchw
        {dilations = dense<1> : vector<2xi64>,
         strides = dense<14> : vector<2xi64>}
        ins(%input, %filter : memref<1x3x224x224xf16>,
                              memref<384x3x14x14xf16>)
        outs(%output : memref<1x384x16x16xf32>)
    return
  }
}

// CHECK-LABEL: func.func @mixed_f16_f32_conv
// CHECK-SAME: alps.c.hvx_widening_convs = 1
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: scf.for {{.*}} step %[[C64:.*]]
// CHECK: %[[MASK:.*]] = vector.create_mask {{.*}} : vector<64xi1>
// CHECK: %[[ACC:.*]] = vector.maskedload {{.*}}, %[[MASK]], {{.*}} : {{.*}} vector<64xi1>, vector<64xf32> into vector<64xf32>
// CHECK: %[[ACT:.*]] = vector.gather {{.*}}, %[[MASK]], {{.*}} : {{.*}} vector<64xi32>, vector<64xi1>, vector<64xf16> into vector<64xf16>
// CHECK: %[[WEIGHT:.*]] = vector.broadcast {{.*}} : f16 to vector<64xf16>
// CHECK: %[[ACT32:.*]] = arith.extf %[[ACT]] : vector<64xf16> to vector<64xf32>
// CHECK: %[[WEIGHT32:.*]] = arith.extf %[[WEIGHT]] : vector<64xf16> to vector<64xf32>
// CHECK: %[[PRODUCT:.*]] = arith.mulf %[[ACT32]], %[[WEIGHT32]] : vector<64xf32>
// CHECK: %[[SUM:.*]] = arith.addf {{.*}}, %[[PRODUCT]] : vector<64xf32>
// CHECK: vector.maskedstore {{.*}}, %[[MASK]], %{{.*}} : {{.*}} vector<64xi1>, vector<64xf32>

// CHECK-LABEL: func.func @mixed_f16_f32_conv1d
// CHECK-SAME: alps.c.hvx_widening_convs = 1
// CHECK-NOT: linalg.conv_1d_ncw_fcw
// CHECK: scf.for {{.*}} step %[[C64_1D:.*]]
// CHECK: %[[MASK_1D:.*]] = vector.create_mask {{.*}} : vector<64xi1>
// CHECK: %[[ACC_1D:.*]] = vector.maskedload {{.*}}, %[[MASK_1D]], {{.*}} : {{.*}} vector<64xi1>, vector<64xf32> into vector<64xf32>
// CHECK: %[[ACT_1D:.*]] = vector.gather {{.*}}, %[[MASK_1D]], {{.*}} : {{.*}} vector<64xi32>, vector<64xi1>, vector<64xf16> into vector<64xf16>
// CHECK: %[[WEIGHT_1D:.*]] = vector.broadcast {{.*}} : f16 to vector<64xf16>
// CHECK: %[[ACT32_1D:.*]] = arith.extf %[[ACT_1D]] : vector<64xf16> to vector<64xf32>
// CHECK: %[[WEIGHT32_1D:.*]] = arith.extf %[[WEIGHT_1D]] : vector<64xf16> to vector<64xf32>
// CHECK: %[[PRODUCT_1D:.*]] = arith.mulf %[[ACT32_1D]], %[[WEIGHT32_1D]] : vector<64xf32>
// CHECK: %[[SUM_1D:.*]] = arith.addf {{.*}}, %[[PRODUCT_1D]] : vector<64xf32>
// CHECK: vector.maskedstore {{.*}}, %[[MASK_1D]], %{{.*}} : {{.*}} vector<64xi1>, vector<64xf32>

// CHECK-LABEL: func.func @small_nonoverlap_patchify
// CHECK-NOT: alps.c.hvx_widening_convs
// CHECK: linalg.conv_2d_nchw_fchw
