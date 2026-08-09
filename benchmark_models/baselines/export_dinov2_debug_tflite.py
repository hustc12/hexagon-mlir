#!/usr/bin/env python3
"""Build the exact DINOv2 Debug graph as a float32-container TFLite model.

The model parameters retain their FP16-rounded values.  Float32 containers are
used because LiteRT's QAIRT 2.26 compiler accepts this graph and lowers it to
FP16 HTP execution, while TensorFlow Lite does not expose a native FP16 input.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from onnx import numpy_helper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_module(onnx_path):
    model = onnx.load(str(onnx_path))
    weights = {
        init.name: numpy_helper.to_array(init).astype(np.float32)
        for init in model.graph.initializer
    }

    def c(name):
        return tf.constant(
            weights[name], dtype=tf.float32, name=name.replace(".", "_")
        )

    conv_b = c("model.dinov2.embeddings.patch_embeddings.projection.bias")
    norm_w = c("model.dinov2.encoder.layer.0.norm1.weight")
    conv_w = tf.transpose(
        c("model.dinov2.embeddings.patch_embeddings.projection.weight"),
        [2, 3, 1, 0],
    )
    cls_token = c("model.dinov2.embeddings.cls_token")
    position = c("model.dinov2.embeddings.position_embeddings")
    q_w = c("onnx::MatMul_193")
    k_w = c("onnx::MatMul_181")
    v_w = c("onnx::MatMul_187")
    attn_out_w = c("onnx::MatMul_207")
    fc1_w = c("onnx::MatMul_208")
    fc1_b = c("model.dinov2.encoder.layer.0.mlp.fc1.bias")
    fc2_w = c("onnx::MatMul_209")
    classifier_w = c("model.classifier.weight")
    classifier_b = c("model.classifier.bias")

    def layer_norm(x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) * tf.math.rsqrt(variance + 1.0e-6) * norm_w + conv_b

    class DinoDebug(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec([1, 32, 32, 3], tf.float32, name="pixels")
            ]
        )
        def __call__(self, pixels):
            patches = tf.nn.conv2d(
                pixels, conv_w, strides=[1, 8, 8, 1], padding="VALID"
            )
            patches = tf.nn.bias_add(patches, conv_b)
            patches = tf.reshape(patches, [1, 16, 64])
            x = tf.concat([cls_token, patches], axis=1) + position

            n = layer_norm(x)
            q = tf.matmul(n, q_w) + conv_b
            k = tf.matmul(n, k_w) + conv_b
            v = tf.matmul(n, v_w) + conv_b
            q = tf.transpose(tf.reshape(q, [1, 17, 2, 32]), [0, 2, 1, 3])
            k = tf.transpose(tf.reshape(k, [1, 17, 2, 32]), [0, 2, 3, 1])
            v = tf.transpose(tf.reshape(v, [1, 17, 2, 32]), [0, 2, 1, 3])
            scale = tf.constant(math.sqrt(1.0 / math.sqrt(32.0)), tf.float32)
            probs = tf.nn.softmax(tf.matmul(q * scale, k * scale), axis=-1)
            context = tf.matmul(probs, v)
            context = tf.reshape(
                tf.transpose(context, [0, 2, 1, 3]), [1, 17, 64]
            )
            x = x + (tf.matmul(context, attn_out_w) + conv_b) * norm_w

            hidden = tf.matmul(layer_norm(x), fc1_w) + fc1_b
            gelu = 0.5 * hidden * (
                1.0
                + tf.math.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (hidden + 0.044715 * tf.pow(hidden, 3.0))
                )
            )
            x = x + (tf.matmul(gelu, fc2_w) + conv_b) * norm_w
            x = layer_norm(x)
            pooled = tf.concat(
                [x[:, 0, :], tf.reduce_mean(x[:, 1:, :], axis=1)], axis=1
            )
            logits = (
                tf.matmul(pooled, classifier_w, transpose_b=True) + classifier_b
            )
            return {"logits": tf.identity(logits, name="logits")}

    return DinoDebug()


def main():
    args = parse_args()
    module = build_module(args.onnx)
    concrete = module.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete], module
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    model_bytes = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model_bytes)

    nhwc_input = np.load(args.input_npy).astype(np.float32)
    nchw_input = np.transpose(nhwc_input, [0, 3, 1, 2])
    onnx_session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    if onnx_session.get_inputs()[0].type == "tensor(float16)":
        nchw_input = nchw_input.astype(np.float16)
    onnx_output = onnx_session.run(None, {"pixels": nchw_input})[0].astype(
        np.float32
    )
    interpreter = tf.lite.Interpreter(model_path=str(args.output), num_threads=1)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_info["index"], nhwc_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_info["index"])
    max_abs = float(np.max(np.abs(onnx_output - tflite_output)))
    print(f"Wrote {args.output} ({len(model_bytes)} bytes)")
    print(f"max_abs_diff_onnx_tflite={max_abs:.9g}")
    print(
        f"top1_onnx={int(np.argmax(onnx_output))} "
        f"top1_tflite={int(np.argmax(tflite_output))}"
    )
    if max_abs > 2.0e-4 or np.argmax(onnx_output) != np.argmax(tflite_output):
        raise RuntimeError("TFLite export failed numerical validation")


if __name__ == "__main__":
    main()
