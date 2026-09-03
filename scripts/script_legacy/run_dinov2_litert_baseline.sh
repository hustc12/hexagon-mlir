#!/usr/bin/env bash
set -euo pipefail

# Reproduce the DINOv2 Debug LiteRT + Qualcomm HTP baseline on SM8550.
# All generated files, source worktrees, virtualenvs, and Bazel caches stay
# outside this repository.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
litert_repo="${LITERT_REPO:-/home/huzq85/2-working/hexagon_npu/baselines/LiteRT}"
litert_commit="${LITERT_COMMIT:-e879503271481c476d2be41668e75dfc9d432e90}"
litert_work="${LITERT_WORK_DIR:-/tmp/litert-qnn226-omnifetch}"
bazel_host_root="${LITERT_BAZEL_HOST_ROOT:-/tmp/litert-bazel-qnn226-omnifetch-host}"
bazel_android_root="${LITERT_BAZEL_ANDROID_ROOT:-/tmp/litert-bazel-qnn226-omnifetch-android}"
bazelisk="${BAZELISK:-/tmp/bazelisk}"
qnn_root="${QNN_SDK_ROOT:-/home/huzq85/2-working/hexagon_npu/baselines/qairt/2.26.2.240911}"
android_ndk="${ANDROID_NDK_ROOT:-/home/huzq85/softwareDev/AndroidSDK/ndk/26.3.11579264}"
android_sdk="${ANDROID_SDK_ROOT:-/home/huzq85/softwareDev/AndroidSDK}"
host_toolchain="${QNN_HOST_CXX_LIB:-/home/huzq85/2-working/hexagon_npu/HOST_TOOLCHAIN/lib}"
project_python="${PROJECT_PYTHON:-${project_root}/../mlir-env/bin/python}"
convert_venv="${LITERT_CONVERT_VENV:-/tmp/litert-dinov2-convert-venv}"
output_dir="${OUTPUT_DIR:-/tmp/omnifetch-baselines/dinov2-debug-litert}"
device_serial="${ANDROID_SERIAL:-49d1c7b2}"
device_dir="${LITERT_DEVICE_DIR:-/data/local/tmp/omnifetch_baselines/dinov2_debug_litert}"
iterations="${LITERT_ITERATIONS:-100}"
trials="${LITERT_TRIALS:-3}"
compat_patch="${project_root}/benchmark_models/baselines/litert_qnn226_compat.patch"

mkdir -p "${output_dir}"

for required in \
  "${litert_repo}/.git" \
  "${qnn_root}/include/QNN/QnnInterface.h" \
  "${android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" \
  "${bazelisk}" \
  "${compat_patch}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

if [[ ! -e "${litert_work}/WORKSPACE" ]]; then
  git -C "${litert_repo}" worktree add --detach "${litert_work}" "${litert_commit}"
fi
if git -C "${litert_work}" apply --check "${compat_patch}" 2>/dev/null; then
  git -C "${litert_work}" apply "${compat_patch}"
elif ! git -C "${litert_work}" apply --reverse --check "${compat_patch}" 2>/dev/null; then
  echo "LiteRT worktree is neither clean nor compat-patched: ${litert_work}" >&2
  exit 1
fi

qnn_prepare="${output_dir}/qnn_prepare"
if [[ ! -f "${qnn_prepare}/dinov2_debug.onnx" ]]; then
  OUTPUT_DIR="${qnn_prepare}" \
    "${project_root}/scripts/script_legacy/run_dinov2_qnn_baseline.sh" --prepare-only
fi

if [[ ! -x "${convert_venv}/bin/python" ]]; then
  "${project_python}" -m venv "${convert_venv}"
  "${convert_venv}/bin/pip" install \
    tensorflow-cpu==2.19.0 onnx==1.16.1 onnxruntime==1.20.1
fi

tflite_model="${output_dir}/dinov2_debug_exact_fp32.tflite"
"${convert_venv}/bin/python" \
  "${project_root}/benchmark_models/baselines/export_dinov2_debug_tflite.py" \
  --onnx "${qnn_prepare}/dinov2_debug.onnx" \
  --input-npy "${qnn_prepare}/pixels_nhwc_f32.npy" \
  --output "${tflite_model}" \
  2>&1 | tee "${output_dir}/tflite_export.log"

qairt_link="${output_dir}/qairt_sdk"
mkdir -p "${qairt_link}"
ln -sfn "${qnn_root}" "${qairt_link}/latest"

host_build=(
  "${bazelisk}" --batch "--output_user_root=${bazel_host_root}"
  build -c opt --cxxopt=--std=c++17 --nocheck_visibility
)
(
  cd "${litert_work}"
  LITERT_QAIRT_SDK="${qairt_link}/" "${host_build[@]}" \
    //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so \
    //litert/tools:apply_plugin_main
)

compiled_model="${output_dir}/dinov2_debug_SM8550_litert.tflite"
LD_LIBRARY_PATH="${host_toolchain}:${qnn_root}/lib/x86_64-linux-clang" \
  "${litert_work}/bazel-bin/litert/tools/apply_plugin_main" \
  --cmd=apply \
  --model="${tflite_model}" \
  --soc_manufacturer=Qualcomm \
  --soc_model=SM8550 \
  --libs="${litert_work}/bazel-bin/litert/vendors/qualcomm/compiler" \
  --o="${compiled_model}" \
  --qualcomm_log_level=off \
  2>&1 | tee "${output_dir}/litert_apply.log"

# LiteRT's configure script validates an Android SDK even though these native
# targets only need the NDK. Reuse a real platform when installed; otherwise
# create a harmless temporary SDK facade around the available build-tools.
sdk_for_config="${android_sdk}"
if [[ ! -d "${sdk_for_config}/platforms" ]]; then
  sdk_for_config="${output_dir}/android_sdk_facade"
  mkdir -p \
    "${sdk_for_config}/platforms/android-35" \
    "${sdk_for_config}/build-tools"
  ln -sfn "${android_sdk}/build-tools/37.0.0" \
    "${sdk_for_config}/build-tools/37.0.0"
  ln -sfn "${android_sdk}/build-tools/37.0.0/lib/d8.jar" \
    "${sdk_for_config}/platforms/android-35/android.jar"
fi

ln -sfn "${bazelisk}" "${output_dir}/bazel"
(
  cd "${litert_work}"
  PATH="${output_dir}:/usr/local/bin:/usr/bin:/bin" \
  PYTHON_BIN_PATH=/usr/bin/python3 \
  PYTHON_LIB_PATH=/usr/lib/python3/dist-packages \
  USE_DEFAULT_PYTHON_LIB_PATH=1 \
  TF_NEED_ROCM=0 \
  TF_NEED_CUDA=0 \
  TF_NEED_CLANG=0 \
  TF_SET_ANDROID_WORKSPACE=1 \
  ANDROID_NDK_HOME="${android_ndk}" \
  ANDROID_NDK_API_LEVEL=26 \
  ANDROID_SDK_HOME="${sdk_for_config}" \
  ANDROID_API_LEVEL=35 \
  ANDROID_BUILD_TOOLS_VERSION=37.0.0 \
  CC_OPT_FLAGS=-Wno-sign-compare \
  TF_SYSTEM_LIBS= \
    python3 configure.py

  LITERT_QAIRT_SDK="${qairt_link}/" \
  ANDROID_NDK_HOME="${android_ndk}" \
    "${bazelisk}" --batch "--output_user_root=${bazel_android_root}" \
    build -c opt --cxxopt=--std=c++17 --nocheck_visibility \
    --config=android_arm64 \
    //litert/tools:run_model \
    //litert/vendors/qualcomm/dispatch:dispatch_api_so
)

runner="${litert_work}/bazel-bin/litert/tools/run_model"
dispatch="${litert_work}/bazel-bin/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so"
runtime="${litert_work}/bazel-bin/litert/c/libLiteRtRuntimeCApi.so"
qnn_android="${qnn_root}/lib/aarch64-android"
qnn_v73="${qnn_root}/lib/hexagon-v73/unsigned"

adb -s "${device_serial}" shell mkdir -p "${device_dir}"
adb -s "${device_serial}" push \
  "${runner}" "${dispatch}" "${runtime}" \
  "${tflite_model}" "${compiled_model}" \
  "${qnn_android}/libQnnHtp.so" \
  "${qnn_android}/libQnnSystem.so" \
  "${qnn_android}/libQnnHtpV73Stub.so" \
  "${qnn_v73}/libQnnHtpV73Skel.so" \
  "${device_dir}/"

run_device_case() {
  local name="$1"
  local graph="$2"
  local accelerator="$3"
  adb -s "${device_serial}" shell \
    "cd '${device_dir}' && \
     chmod 755 run_model && \
     export LD_LIBRARY_PATH='${device_dir}' && \
     export ADSP_LIBRARY_PATH='${device_dir}' && \
     export LITERT_GRAPH='${device_dir}/${graph}' && \
     export LITERT_DISPATCH_DIR='${device_dir}' && \
     export LITERT_ACCELERATOR='${accelerator}' && \
     export LITERT_ITERATIONS='${iterations}' && \
     export LITERT_PRINT_TENSORS=0 && \
     ./run_model" \
    2>&1 | tee "${output_dir}/${name}.log"
}

# Cases and trials are intentionally serial; only one model invocation is in
# flight at any time.
for ((trial = 1; trial <= trials; ++trial)); do
  run_device_case \
    "npu_trial_${trial}" \
    "$(basename "${compiled_model}")" npu
done
for ((trial = 1; trial <= trials; ++trial)); do
  run_device_case \
    "cpu_trial_${trial}" \
    "$(basename "${tflite_model}")" cpu
done

echo "===== LiteRT DINOv2 Debug summary ====="
grep -H -E \
  "First run took|Steady-state runs excluding first|Steady-state median|Steady-state p90" \
  "${output_dir}"/npu_trial_*.log "${output_dir}"/cpu_trial_*.log
echo "Generated artifacts and logs: ${output_dir}"
