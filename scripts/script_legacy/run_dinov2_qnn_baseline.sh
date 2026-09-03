#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
qnn_root="${QNN_SDK_ROOT:-/home/huzq85/2-working/hexagon_npu/baselines/qairt/2.26.2.240911}"
output_dir="${OUTPUT_DIR:-/tmp/omnifetch-baselines/dinov2-debug-qnn}"
android_ndk="${ANDROID_NDK_ROOT:-/home/huzq85/softwareDev/AndroidSDK/ndk/26.3.11579264}"
py310="${QNN_PYTHON310:-${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/python3/bin/python3.10}"
qnn_py_deps="${QNN_PYTHONPATH:-/tmp/qnn26-py310-packages}"
onnx_py_deps="${ONNX_PYTHONPATH:-/tmp/onnx-only-py311}"
host_cxx_lib="${QNN_HOST_CXX_LIB:-/home/huzq85/2-working/hexagon_npu/HOST_TOOLCHAIN/lib}"
hexagon_host_lib="${HEXAGON_HOST_LIB:-/home/huzq85/2-working/hexagon_npu/HEXAGON_TOOLS/Tools/lib}"
device_dir="${QNN_DEVICE_DIR:-/data/local/tmp/omnifetch_baselines/dinov2_debug_qnn}"
project_python="${PROJECT_PYTHON:-${project_root}/../mlir-env/bin/python}"
iterations="${QNN_ITERATIONS:-20}"
perf_profile="${QNN_PERF_PROFILE:-default}"
hvx_threads="${QNN_HVX_THREADS:-1}"
prepare_only=0

usage() {
  echo "Usage: $0 [--prepare-only]"
  echo "Environment overrides: QNN_SDK_ROOT OUTPUT_DIR ANDROID_NDK_ROOT"
  echo "  QNN_PYTHON310 QNN_PYTHONPATH ONNX_PYTHONPATH QNN_DEVICE_DIR"
  echo "  QNN_ITERATIONS QNN_PERF_PROFILE QNN_HVX_THREADS"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-only)
      prepare_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${output_dir}"

for required in \
  "${py310}" \
  "${qnn_root}/bin/x86_64-linux-clang/qnn-onnx-converter" \
  "${qnn_root}/bin/x86_64-linux-clang/qnn-model-lib-generator"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required tool: ${required}" >&2
    exit 1
  fi
done

if [[ ! -d "${qnn_py_deps}/onnx" ]]; then
  echo "Installing isolated QNN Python 3.10 dependencies in ${qnn_py_deps}."
  "${project_python}" -m pip install \
    --target "${qnn_py_deps}" \
    --platform manylinux2014_x86_64 \
    --python-version 3.10 \
    --implementation cp \
    --abi cp310 \
    --only-binary=:all: \
    onnx==1.16.1 protobuf==3.20.3 numpy==1.26.4 \
    pyyaml==6.0.1 packaging==24.0 tabulate==0.9.0 scipy==1.10.1 \
    pandas==2.0.1 mako==1.1.6
fi
if [[ ! -d "${onnx_py_deps}/onnx" ]]; then
  echo "Installing isolated project-Python ONNX dependencies in ${onnx_py_deps}."
  "${project_python}" -m pip install \
    --target "${onnx_py_deps}" --no-deps onnx==1.16.1 protobuf==5.29.5
fi

PYTHONPATH="${onnx_py_deps}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${project_python}" \
  "${project_root}/benchmark_models/baselines/export_dinov2_debug_onnx.py" \
  --output-dir "${output_dir}"

sdk_python_path="${qnn_root}/lib/python"
sdk_host_lib="${qnn_root}/lib/x86_64-linux-clang"
PYTHONPATH="${qnn_py_deps}:${sdk_python_path}" \
LD_LIBRARY_PATH="${host_cxx_lib}:${hexagon_host_lib}:${sdk_host_lib}" \
  "${py310}" \
  "${qnn_root}/bin/x86_64-linux-clang/qnn-onnx-converter" \
  --input_network "${output_dir}/dinov2_debug.onnx" \
  --input_dtype pixels float16 \
  --input_layout pixels NCHW \
  --float_bitwidth 16 \
  --output_path "${output_dir}/dinov2_debug.cpp" \
  >"${output_dir}/qnn_converter.log" 2>&1
echo "QNN ONNX conversion completed."

PATH="${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}" \
  "${qnn_root}/bin/x86_64-linux-clang/qnn-model-lib-generator" \
  -c "${output_dir}/dinov2_debug.cpp" \
  -b "${output_dir}/dinov2_debug.bin" \
  -o "${output_dir}/model_libs" \
  -t aarch64-android \
  >"${output_dir}/qnn_model_lib_generator.log" 2>&1
echo "QNN Android model library completed."

if [[ "${prepare_only}" -eq 1 ]]; then
  echo "QNN model prepared at ${output_dir}; device run skipped."
  exit 0
fi

qnn_android="${qnn_root}/lib/aarch64-android"
qnn_v73="${qnn_root}/lib/hexagon-v73/unsigned"
model_so="${output_dir}/model_libs/aarch64-android/libdinov2_debug.so"
qnn_extension_so="${qnn_android}/libQnnHtpNetRunExtensions.so"

if [[ ! -f "${qnn_extension_so}" ]]; then
  echo "Missing HTP net-run extension: ${qnn_extension_so}" >&2
  exit 1
fi

printf '%s\n' \
  '{' \
  '  "graphs": [' \
  '    {' \
  '      "graph_names": ["dinov2_debug"],' \
  "      \"hvx_threads\": ${hvx_threads}" \
  '    }' \
  '  ]' \
  '}' >"${output_dir}/htp_config.json"
printf '%s\n' \
  '{' \
  '  "backend_extensions": {' \
  '    "shared_library_path": "libQnnHtpNetRunExtensions.so",' \
  '    "config_file_path": "htp_config.json"' \
  '  }' \
  '}' >"${output_dir}/backend_extensions.json"

adb shell "mkdir -p '${device_dir}'"
adb push "${qnn_root}/bin/aarch64-android/qnn-net-run" "${device_dir}/"
adb push "${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "${device_dir}/"
adb push "${qnn_android}/libQnnHtp.so" "${device_dir}/"
adb push "${qnn_android}/libQnnHtpPrepare.so" "${device_dir}/"
adb push "${qnn_android}/libQnnHtpV73Stub.so" "${device_dir}/"
adb push "${qnn_extension_so}" "${device_dir}/"
adb push "${qnn_v73}/libQnnHtpV73Skel.so" "${device_dir}/"
adb push "${model_so}" "${device_dir}/"
adb push "${output_dir}/pixels_nhwc_f32.raw" "${device_dir}/"
adb push "${output_dir}/htp_config.json" "${device_dir}/"
adb push "${output_dir}/backend_extensions.json" "${device_dir}/"

device_input_list="${device_dir}/input_list.txt"
host_input_list="${output_dir}/device_input_list.txt"
: >"${host_input_list}"
for ((iteration = 0; iteration < iterations; ++iteration)); do
  printf 'pixels:=%s/pixels_nhwc_f32.raw\n' "${device_dir}" >>"${host_input_list}"
done
adb push "${host_input_list}" "${device_input_list}"
adb shell "rm -rf '${device_dir}/output'"
adb shell "cd '${device_dir}' && \
  export LD_LIBRARY_PATH='${device_dir}:/vendor/lib64' && \
  export ADSP_LIBRARY_PATH='${device_dir};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && \
  ./qnn-net-run \
    --backend libQnnHtp.so \
    --model '$(basename "${model_so}")' \
    --input_list input_list.txt \
    --output_dir output \
    --config_file backend_extensions.json \
    --synchronous \
    --profiling_level basic \
    --perf_profile '${perf_profile}'"

adb pull "${device_dir}/output/Result_0/logits.raw" "${output_dir}/qnn_logits_f32.raw"
adb pull "${device_dir}/output/execution_metadata.yaml" "${output_dir}/qnn_execution_metadata.yaml"
adb pull "${device_dir}/output/qnn-profiling-data_0.log" "${output_dir}/qnn-profiling-data.log"
PYTHONPATH="${onnx_py_deps}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${project_python}" \
  "${project_root}/benchmark_models/baselines/export_dinov2_debug_onnx.py" \
  --output-dir "${output_dir}" \
  --compare-qnn-output "${output_dir}/qnn_logits_f32.raw"
LD_LIBRARY_PATH="${sdk_host_lib}:${host_cxx_lib}:${hexagon_host_lib}" \
  "${qnn_root}/bin/x86_64-linux-clang/qnn-profile-viewer" \
  --input_log "${output_dir}/qnn-profiling-data.log" \
  >"${output_dir}/qnn_profile.txt"
grep -A8 "Execute Stats (Average)" "${output_dir}/qnn_profile.txt" || true
echo "QNN HTP run completed; artifacts are in ${output_dir}."
