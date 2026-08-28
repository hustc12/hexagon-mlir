// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(alps-attention-destination-formation))' 2>&1 | FileCheck %s

func.func @attention_destination(%source: memref<2x2x4xf32>) {
  %source_active = memref.subview %source[0, 0, 0] [2, 2, 3] [1, 1, 1]
    : memref<2x2x4xf32> to memref<2x2x3xf32, strided<[8, 4, 1]>>
  %temporary = memref.alloc() : memref<2x2x3xf32>
  memref.copy %source_active, %temporary
    : memref<2x2x3xf32, strided<[8, 4, 1]>> to memref<2x2x3xf32>
  %temporary_row = memref.subview %temporary[0, 0, 0] [1, 1, 3] [1, 1, 1]
    : memref<2x2x3xf32> to memref<3xf32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %value = memref.load %temporary_row[%c0] : memref<3xf32, strided<[1]>>
  memref.store %value, %temporary_row[%c1] : memref<3xf32, strided<[1]>>
  %destination = memref.alloc() : memref<2x2x4xf32>
  memref.copy %source, %destination
    : memref<2x2x4xf32> to memref<2x2x4xf32>
  %destination_active = memref.subview %destination[0, 0, 0] [2, 2, 3] [1, 1, 1]
    : memref<2x2x4xf32> to memref<2x2x3xf32, strided<[8, 4, 1]>>
  memref.copy %temporary, %destination_active
    : memref<2x2x3xf32> to memref<2x2x3xf32, strided<[8, 4, 1]>>
  memref.dealloc %temporary : memref<2x2x3xf32>
  memref.dealloc %destination : memref<2x2x4xf32>
  return
}

// CHECK: [ALPS-P5H] function=attention_destination matched=1 rewritten=1 eliminated_copy_bytes=96 residual_tail_copy_bytes=16
// CHECK-LABEL: func.func @attention_destination
// CHECK-SAME: (%[[SOURCE:.*]]: memref<2x2x4xf32>)
// CHECK: %[[DEST:.*]] = memref.alloc() : memref<2x2x4xf32>
// CHECK: %[[DACTIVE:.*]] = memref.subview %[[DEST]][0, 0, 0] [2, 2, 3] [1, 1, 1]
// CHECK: memref.copy %{{.*}}, %[[DACTIVE]]
// CHECK: %[[ROW:.*]] = memref.subview %[[DACTIVE]][0, 0, 0] [1, 1, 3] [1, 1, 1]
// CHECK: memref.load %[[ROW]]
// CHECK: memref.store {{.*}}, %[[ROW]]
// CHECK: %[[STAIL:.*]] = memref.subview %[[SOURCE]][0, 0, 3] [2, 2, 1] [1, 1, 1]
// CHECK: %[[DTAIL:.*]] = memref.subview %[[DEST]][0, 0, 3] [2, 2, 1] [1, 1, 1]
// CHECK: memref.copy %[[STAIL]], %[[DTAIL]]
// CHECK-NOT: memref.alloc() : memref<2x2x3xf32>

// An existing destination read before the writeback makes early destination
// formation observable, so the chain must remain untouched.
func.func @reject_early_destination_read(%source: memref<2x2x4xf32>) {
  %source_active = memref.subview %source[0, 0, 0] [2, 2, 3] [1, 1, 1]
    : memref<2x2x4xf32> to memref<2x2x3xf32, strided<[8, 4, 1]>>
  %temporary = memref.alloc() : memref<2x2x3xf32>
  memref.copy %source_active, %temporary
    : memref<2x2x3xf32, strided<[8, 4, 1]>> to memref<2x2x3xf32>
  %temporary_row = memref.subview %temporary[0, 0, 0] [1, 1, 3] [1, 1, 1]
    : memref<2x2x3xf32> to memref<3xf32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %value = memref.load %temporary_row[%c0] : memref<3xf32, strided<[1]>>
  memref.store %value, %temporary_row[%c0] : memref<3xf32, strided<[1]>>
  %destination = memref.alloc() : memref<2x2x4xf32>
  %early = memref.load %destination[%c0, %c0, %c0] : memref<2x2x4xf32>
  memref.copy %source, %destination
    : memref<2x2x4xf32> to memref<2x2x4xf32>
  %destination_active = memref.subview %destination[0, 0, 0] [2, 2, 3] [1, 1, 1]
    : memref<2x2x4xf32> to memref<2x2x3xf32, strided<[8, 4, 1]>>
  memref.copy %temporary, %destination_active
    : memref<2x2x3xf32> to memref<2x2x3xf32, strided<[8, 4, 1]>>
  memref.dealloc %temporary : memref<2x2x3xf32>
  memref.dealloc %destination : memref<2x2x4xf32>
  return
}

// CHECK-LABEL: func.func @reject_early_destination_read
// CHECK-SAME: alps.p5h.matched_chains = 0
// CHECK-SAME: alps.p5h.rewritten_chains = 0
// CHECK: memref.copy %{{.*}}, %[[TMP:[a-zA-Z0-9_]+]]
// CHECK: memref.copy %{{.*}}, %[[DST:[a-zA-Z0-9_]+]]
// CHECK: memref.copy %[[TMP]], %{{.*}}

// A temporary-derived view used after the whole-root copy is outside the
// proven in-place interval and must also be rejected.
func.func @reject_late_temporary_use(%source: memref<2x2x4xf32>) {
  %source_active = memref.subview %source[0, 0, 0] [2, 2, 3] [1, 1, 1]
    : memref<2x2x4xf32> to memref<2x2x3xf32, strided<[8, 4, 1]>>
  %temporary = memref.alloc() : memref<2x2x3xf32>
  memref.copy %source_active, %temporary
    : memref<2x2x3xf32, strided<[8, 4, 1]>> to memref<2x2x3xf32>
  %temporary_row = memref.subview %temporary[0, 0, 0] [1, 1, 3] [1, 1, 1]
    : memref<2x2x3xf32> to memref<3xf32, strided<[1]>>
  %destination = memref.alloc() : memref<2x2x4xf32>
  memref.copy %source, %destination
    : memref<2x2x4xf32> to memref<2x2x4xf32>
  %c0 = arith.constant 0 : index
  %late = memref.load %temporary_row[%c0] : memref<3xf32, strided<[1]>>
  memref.store %late, %temporary_row[%c0] : memref<3xf32, strided<[1]>>
  %destination_active = memref.subview %destination[0, 0, 0] [2, 2, 3] [1, 1, 1]
    : memref<2x2x4xf32> to memref<2x2x3xf32, strided<[8, 4, 1]>>
  memref.copy %temporary, %destination_active
    : memref<2x2x3xf32> to memref<2x2x3xf32, strided<[8, 4, 1]>>
  memref.dealloc %temporary : memref<2x2x3xf32>
  memref.dealloc %destination : memref<2x2x4xf32>
  return
}

// CHECK-LABEL: func.func @reject_late_temporary_use
// CHECK-SAME: alps.p5h.matched_chains = 0
// CHECK-SAME: alps.p5h.rewritten_chains = 0
// CHECK: memref.copy %{{.*}}, %[[TMP2:[a-zA-Z0-9_]+]]
// CHECK: memref.copy %{{.*}}, %[[DST2:[a-zA-Z0-9_]+]]
// CHECK: memref.copy %[[TMP2]], %{{.*}}
