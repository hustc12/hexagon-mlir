// RUN: linalg-hexagon-opt %s --hexagon-rewrite-ub-poison-to-zero | FileCheck %s

module {
  func.func @scalar_poison() -> f32 {
    // CHECK-LABEL: func.func @scalar_poison
    // CHECK-NOT: ub.poison
    // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: return %[[ZERO]] : f32
    %0 = ub.poison : f32
    return %0 : f32
  }

  func.func @vector_poison() -> vector<4xf16> {
    // CHECK-LABEL: func.func @vector_poison
    // CHECK-NOT: ub.poison
    // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<4xf16>
    // CHECK: return %[[ZERO]] : vector<4xf16>
    %0 = ub.poison : vector<4xf16>
    return %0 : vector<4xf16>
  }
}
