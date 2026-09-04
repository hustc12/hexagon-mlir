#!/usr/bin/env bash
# Build the exact DINOv2-small full workload once, then run QNN CPU and HTP
# strictly serially on the same Android device and input.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
qnn_root="${QNN_SDK_ROOT:-/home/huzq85/2-working/hexagon_npu/baselines/qairt/2.26.2.240911}"
output_dir="${OUTPUT_DIR:-/tmp/alps-baselines/dinov2-small-full-qnn}"
android_ndk="${ANDROID_NDK_ROOT:-/home/huzq85/softwareDev/AndroidSDK/ndk/26.3.11579264}"
py310="${QNN_PYTHON310:-${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/python3/bin/python3.10}"
qnn_py_deps="${QNN_PYTHONPATH:-/tmp/qnn26-py310-packages}"
onnx_py_deps="${ONNX_PYTHONPATH:-/tmp/onnx-only-py311}"
host_cxx_lib="${QNN_HOST_CXX_LIB:-/home/huzq85/2-working/hexagon_npu/HOST_TOOLCHAIN/lib}"
hexagon_host_lib="${HEXAGON_HOST_LIB:-/home/huzq85/2-working/hexagon_npu/HEXAGON_TOOLS/Tools/lib}"
device_dir="${QNN_DEVICE_DIR:-/data/local/tmp/alps_baselines/dinov2_small_full_qnn}"
project_python="${PROJECT_PYTHON:-${project_root}/../mlir-env/bin/python}"
iterations="${QNN_ITERATIONS:-20}"
perf_profile="${QNN_PERF_PROFILE:-default}"
hvx_threads="${QNN_HVX_THREADS:-1}"
prepare_only=0
run_only=0

usage() {
  echo "Usage: $0 [--prepare-only | --run-only]"
  echo "Environment: QNN_SDK_ROOT OUTPUT_DIR QNN_DEVICE_DIR QNN_ITERATIONS"
  echo "             QNN_PERF_PROFILE QNN_HVX_THREADS ANDROID_SERIAL"
}

while (($#)); do
  case "$1" in
    --prepare-only) prepare_only=1; shift ;;
    --run-only) run_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if ((prepare_only && run_only)); then
  echo "--prepare-only and --run-only are mutually exclusive" >&2
  exit 2
fi
[[ "${iterations}" =~ ^[1-9][0-9]*$ ]] || {
  echo "QNN_ITERATIONS must be a positive integer" >&2
  exit 2
}

mkdir -p "${output_dir}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-49d1c7b2}"

converter="${qnn_root}/bin/x86_64-linux-clang/qnn-onnx-converter"
model_generator="${qnn_root}/bin/x86_64-linux-clang/qnn-model-lib-generator"
profile_viewer="${qnn_root}/bin/x86_64-linux-clang/qnn-profile-viewer"
for required in "${py310}" "${converter}" "${model_generator}" "${profile_viewer}"; do
  [[ -e "${required}" ]] || {
    echo "Missing required QNN tool: ${required}" >&2
    exit 1
  }
done

htp_onnx="${output_dir}/dinov2_small_full.onnx"
htp_cpp="${output_dir}/dinov2_small_full.cpp"
htp_bin="${output_dir}/dinov2_small_full.bin"
htp_model_so="${output_dir}/model_libs/aarch64-android/libdinov2_small_full.so"
cpu_onnx="${output_dir}/dinov2_small_full_cpu.onnx"
cpu_cpp="${output_dir}/dinov2_small_full_cpu.cpp"
cpu_bin="${output_dir}/dinov2_small_full_cpu.bin"
cpu_model_so="${output_dir}/model_libs/aarch64-android/libdinov2_small_full_cpu.so"

if ((run_only == 0)); then
  if [[ ! -d "${qnn_py_deps}/onnx" || ! -d "${onnx_py_deps}/onnx" ]]; then
    echo "Missing the isolated ONNX dependencies used by the verified Debug flow." >&2
    echo "Run scripts/script_legacy/run_dinov2_qnn_baseline.sh --prepare-only once first." >&2
    exit 1
  fi

  PYTHONPATH="${onnx_py_deps}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${project_python}" \
    "${project_root}/benchmark_models/baselines/export_dinov2_full_onnx.py" \
    --output-dir "${output_dir}" \
    --precision fp16
  PYTHONPATH="${onnx_py_deps}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${project_python}" \
    "${project_root}/benchmark_models/baselines/export_dinov2_full_onnx.py" \
    --output-dir "${output_dir}" \
    --precision fp32

  sdk_python_path="${qnn_root}/lib/python"
  sdk_host_lib="${qnn_root}/lib/x86_64-linux-clang"
  PYTHONPATH="${qnn_py_deps}:${sdk_python_path}" \
  LD_LIBRARY_PATH="${host_cxx_lib}:${hexagon_host_lib}:${sdk_host_lib}" \
    "${py310}" "${converter}" \
    --input_network "${htp_onnx}" \
    --input_dtype pixels float16 \
    --input_layout pixels NCHW \
    --float_bitwidth 16 \
    --output_path "${htp_cpp}" \
    >"${output_dir}/qnn_htp_converter.log" 2>&1
  PYTHONPATH="${qnn_py_deps}:${sdk_python_path}" \
  LD_LIBRARY_PATH="${host_cxx_lib}:${hexagon_host_lib}:${sdk_host_lib}" \
    "${py310}" "${converter}" \
    --input_network "${cpu_onnx}" \
    --input_dtype pixels float32 \
    --input_layout pixels NCHW \
    --float_bitwidth 32 \
    --output_path "${cpu_cpp}" \
    >"${output_dir}/qnn_cpu_converter.log" 2>&1
  echo "QNN FP16 HTP and FP32 CPU ONNX conversion completed."

  PATH="${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}" \
    "${model_generator}" \
    -c "${htp_cpp}" \
    -b "${htp_bin}" \
    -o "${output_dir}/model_libs" \
    -t aarch64-android \
    >"${output_dir}/qnn_htp_model_lib_generator.log" 2>&1
  PATH="${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}" \
    "${model_generator}" \
    -c "${cpu_cpp}" \
    -b "${cpu_bin}" \
    -o "${output_dir}/model_libs" \
    -t aarch64-android \
    >"${output_dir}/qnn_cpu_model_lib_generator.log" 2>&1
  echo "QNN FP16 HTP and FP32 CPU Android model libraries completed."
fi

for required in \
  "${htp_model_so}" \
  "${cpu_model_so}" \
  "${output_dir}/pixels_nhwc_f32.raw" \
  "${output_dir}/reference_logits_fp16.raw" \
  "${output_dir}/reference_logits_fp32.raw"; do
  [[ -f "${required}" ]] || {
    echo "Missing prepared artifact: ${required}" >&2
    exit 1
  }
done
if ((prepare_only)); then
  echo "Prepared full DINOv2 QNN model in ${output_dir}; device run skipped."
  exit 0
fi

qnn_android="${qnn_root}/lib/aarch64-android"
qnn_v73="${qnn_root}/lib/hexagon-v73/unsigned"
qnn_extension_so="${qnn_android}/libQnnHtpNetRunExtensions.so"
cxx_shared="${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"

printf '%s\n' \
  '{' \
  '  "graphs": [' \
  '    {' \
  '      "graph_names": ["dinov2_small_full"],' \
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

host_input_list="${output_dir}/device_input_list.txt"
: >"${host_input_list}"
for ((iteration = 0; iteration < iterations; ++iteration)); do
  printf 'pixels:=%s/pixels_nhwc_f32.raw\n' "${device_dir}" >>"${host_input_list}"
done

adb shell "mkdir -p '${device_dir}'"
adb push "${qnn_root}/bin/aarch64-android/qnn-net-run" "${device_dir}/"
adb push "${cxx_shared}" "${device_dir}/"
adb push "${qnn_android}/libQnnCpu.so" "${device_dir}/"
adb push "${qnn_android}/libQnnHtp.so" "${device_dir}/"
adb push "${qnn_android}/libQnnHtpPrepare.so" "${device_dir}/"
adb push "${qnn_android}/libQnnHtpV73Stub.so" "${device_dir}/"
adb push "${qnn_extension_so}" "${device_dir}/"
adb push "${qnn_v73}/libQnnHtpV73Skel.so" "${device_dir}/"
adb push "${cpu_model_so}" "${device_dir}/"
adb push "${htp_model_so}" "${device_dir}/"
adb push "${output_dir}/pixels_nhwc_f32.raw" "${device_dir}/"
adb push "${host_input_list}" "${device_dir}/input_list.txt"
adb push "${output_dir}/htp_config.json" "${device_dir}/"
adb push "${output_dir}/backend_extensions.json" "${device_dir}/"

run_backend() {
  local label=$1 backend=$2 model_so=$3 precision=$4
  local config_args=()
  if [[ "${label}" == "htp" ]]; then
    config_args=(--config_file backend_extensions.json --perf_profile "${perf_profile}")
  fi

  echo "QNN_${label^^}_START iterations=${iterations}"
  adb shell "rm -rf '${device_dir}/output_${label}'"
  adb shell "cd '${device_dir}' && \
    export LD_LIBRARY_PATH='${device_dir}:/vendor/lib64' && \
    export ADSP_LIBRARY_PATH='${device_dir};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && \
    ./qnn-net-run \
      --backend '${backend}' \
      --model '$(basename "${model_so}")' \
      --input_list input_list.txt \
      --output_dir 'output_${label}' \
      --synchronous \
      --profiling_level basic \
      ${config_args[*]}"

  adb pull \
    "${device_dir}/output_${label}/Result_0/logits.raw" \
    "${output_dir}/qnn_${label}_logits_f32.raw"
  adb pull \
    "${device_dir}/output_${label}/execution_metadata.yaml" \
    "${output_dir}/qnn_${label}_execution_metadata.yaml"
  adb pull \
    "${device_dir}/output_${label}/qnn-profiling-data_0.log" \
    "${output_dir}/qnn_${label}_profiling-data.log"

  PYTHONPATH="${onnx_py_deps}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${project_python}" \
    "${project_root}/benchmark_models/baselines/export_dinov2_full_onnx.py" \
    --output-dir "${output_dir}" \
    --compare-qnn-output "${output_dir}/qnn_${label}_logits_f32.raw" \
    --precision "${precision}"

  LD_LIBRARY_PATH="${qnn_root}/lib/x86_64-linux-clang:${host_cxx_lib}:${hexagon_host_lib}" \
    "${profile_viewer}" \
    --input_log "${output_dir}/qnn_${label}_profiling-data.log" \
    >"${output_dir}/qnn_${label}_profile.txt"
  grep -A12 "Execute Stats (Average)" "${output_dir}/qnn_${label}_profile.txt" || true
  echo "QNN_${label^^}_COMPLETE"
}

# Deliberately serial. CPU uses QNN's default scheduler because qnn-net-run
# exposes no CPU thread-count control; HTP is explicitly fixed to one HVX thread.
run_backend cpu libQnnCpu.so "${cpu_model_so}" fp32
run_backend htp libQnnHtp.so "${htp_model_so}" fp16

echo "QNN_FULL_DINOV2_COMPLETE artifacts=${output_dir}"
