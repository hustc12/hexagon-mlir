// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(prefetch-kernel-hx{distance=2 max-command-bytes=8191 enable-two-dimensional=true}))' \
// RUN:   | FileCheck %s
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(prefetch-kernel-hx{distance=2 baseline-kind=apt-get-hx require-manual-safe=true}))' \
// RUN:   | FileCheck %s --check-prefix=APT
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(prefetch-kernel-hx{distance=2 baseline-kind=apt-get-hx require-manual-safe=true manual-candidate-ids=apt_unmarked:loop0:view0}))' \
// RUN:   | FileCheck %s --check-prefix=ALLOW

// CHECK-LABEL: func.func @affine_1d
// CHECK-SAME: prefetch_kernel_hx.admitted_1d = 1
// CHECK-SAME: prefetch_kernel_hx.admitted_2d = 0
// CHECK-SAME: prefetch_kernel_hx.candidates = 1
// CHECK-SAME: prefetch_kernel_hx.hints = 1
// CHECK-SAME: prefetch_kernel_hx.requested_bytes = 128
// CHECK: %[[FUTURE:.+]] = arith.addi %arg1, %{{.+}} : index
// CHECK: scf.if
// CHECK: %[[VIEW:.+]] = memref.subview %arg0[%[[FUTURE]]] [64] [1]
// CHECK: alps.l2_hint %[[VIEW]]
// CHECK-SAME: prefetch_baseline.address_class = "affine_1d"
// CHECK-SAME: prefetch_baseline.candidate_id = "affine_1d:loop0:view0"
// CHECK-SAME: prefetch_baseline.distance = 2
// CHECK-SAME: prefetch_baseline.kind = "prefetch-kernel-hx"
// CHECK-SAME: prefetch_baseline.page_policy = "runtime_clip_v1"
// CHECK-SAME: prefetch_baseline.requested_bytes = 128
func.func @affine_1d(%src: memref<256xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %i = %c0 to %c256 step %c64 {
    %view = memref.subview %src[%i] [64] [1]
      : memref<256xf16> to memref<64xf16, strided<[1], offset: ?>>
    %value = memref.load %view[%c0] : memref<64xf16, strided<[1], offset: ?>>
  }
  return
}

// CHECK-LABEL: func.func @affine_2d
// CHECK-SAME: prefetch_kernel_hx.admitted_1d = 0
// CHECK-SAME: prefetch_kernel_hx.admitted_2d = 1
// CHECK-SAME: prefetch_kernel_hx.hints = 1
// CHECK-SAME: prefetch_kernel_hx.requested_bytes = 512
// CHECK: memref.subview %arg0[0, %{{.+}}] [8, 32] [1, 1]
// CHECK: alps.l2_hint
// CHECK-SAME: prefetch_baseline.address_class = "affine_2d"
func.func @affine_2d(%src: memref<8x128xf16>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c32 {
    %view = memref.subview %src[0, %i] [8, 32] [1, 1]
      : memref<8x128xf16> to memref<8x32xf16, strided<[128, 1], offset: ?>>
    %value = memref.load %view[%c0, %c0]
      : memref<8x32xf16, strided<[128, 1], offset: ?>>
  }
  return
}

// A destination/write view must never be treated as a prefetch-only source.
// CHECK-LABEL: func.func @reject_write
// CHECK-SAME: prefetch_kernel_hx.hints = 0
// CHECK-SAME: prefetch_kernel_hx.rejected_write = 1
// CHECK-NOT: alps.l2_hint
func.func @reject_write(%dst: memref<256xf16>, %value: f16) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %i = %c0 to %c256 step %c64 {
    %view = memref.subview %dst[%i] [64] [1]
      : memref<256xf16> to memref<64xf16, strided<[1], offset: ?>>
    memref.store %value, %view[%c0]
      : memref<64xf16, strided<[1], offset: ?>>
  }
  return
}

// One logical request above the configured budget is rejected rather than
// silently claiming full Prefetch-Kernel coverage after runtime clipping.
// CHECK-LABEL: func.func @reject_oversize
// CHECK-SAME: prefetch_kernel_hx.hints = 0
// CHECK-SAME: prefetch_kernel_hx.rejected_oversize = 1
// CHECK-NOT: alps.l2_hint
func.func @reject_oversize(%src: memref<128x128xf16>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c64 {
    %view = memref.subview %src[%i, 0] [64, 128] [1, 1]
      : memref<128x128xf16>
        to memref<64x128xf16, strided<[128, 1], offset: ?>>
    %value = memref.load %view[%c0, %c0]
      : memref<64x128xf16, strided<[128, 1], offset: ?>>
  }
  return
}

// APT-only mode is intentionally unable to turn an automatically inferred
// address into a standalone APT baseline. It consumes an explicit manual-safe
// marker and is evaluated independently from Prefetch-Kernel-HX.
// CHECK-LABEL: func.func @apt_manual_safe
// CHECK-SAME: prefetch_kernel_hx.hints = 1
// CHECK: alps.l2_hint
// APT-LABEL: func.func @apt_manual_safe
// APT-SAME: prefetch_kernel_hx.hints = 1
// APT-SAME: prefetch_kernel_hx.policy = "apt-get-hx"
// APT-SAME: prefetch_kernel_hx.rejected_unmarked = 0
// APT: alps.l2_hint
// APT-SAME: prefetch_baseline.candidate_id = "apt_manual_safe:loop0:view0"
// APT-SAME: prefetch_baseline.kind = "apt-get-hx"
func.func @apt_manual_safe(%src: memref<256xf16>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %i = %c0 to %c256 step %c64 {
    %view = memref.subview %src[%i] [64] [1]
      {prefetch_baseline.manual_safe}
      : memref<256xf16> to memref<64xf16, strided<[1], offset: ?>>
    %value = memref.load %view[%c0]
      : memref<64xf16, strided<[1], offset: ?>>
  }
  return
}

// APT-LABEL: func.func @apt_unmarked
// CHECK-LABEL: func.func @apt_unmarked
// CHECK-SAME: prefetch_kernel_hx.hints = 1
// CHECK: alps.l2_hint
// APT-SAME: prefetch_kernel_hx.hints = 0
// APT-SAME: prefetch_kernel_hx.rejected_unmarked = 1
// APT-NOT: alps.l2_hint
// ALLOW-LABEL: func.func @apt_unmarked
// ALLOW-SAME: prefetch_kernel_hx.hints = 1
// ALLOW: alps.l2_hint
// ALLOW-SAME: prefetch_baseline.candidate_id = "apt_unmarked:loop0:view0"
// ALLOW-SAME: prefetch_baseline.kind = "apt-get-hx"
func.func @apt_unmarked(%src: memref<256xf16>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %i = %c0 to %c256 step %c64 {
    %view = memref.subview %src[%i] [64] [1]
      : memref<256xf16> to memref<64xf16, strided<[1], offset: ?>>
    %value = memref.load %view[%c0]
      : memref<64xf16, strided<[1], offset: ?>>
  }
  return
}
