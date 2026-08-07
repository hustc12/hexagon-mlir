# ===- torch_mlir_hexagon_launcher.py ---------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os
import hashlib
import shutil
from pathlib import Path
from typing import Optional
import time
from torch import Tensor  # For type annotations
from triton.backends.qcom_hexagon_backend.compiler import (
    HexagonOptions,
)
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
from triton.backends.qcom_hexagon_backend.hexagon_launcher_base import (
    HexagonLauncherBase,
    HexagonWrapperGenerator,
    WrapperGeneratorStrings,
    create_timestamped_folder,
)
from triton.backends.qcom_hexagon_backend.utils import (
    parse_return_types,
    profile_torch_mlir_inputs,
    split_path,
    get_shape,
)
from triton._C.libtriton import qcom_hexagon_backend, ir  # type: ignore

# This file is part of a small subset of python files that uses some type-annotations
# and it passes type-verification with mypy (a type checker).
# To typecheck this set of files, do:
#   mypy compiler.py hexagon_executor.py hexagon_launcher_base.py torch_mlir_hexagon_launcher.py \
#   triton_hexagon_launcher.py --follow-untyped-imports --check-untyped-defs


class TorchMLIRWrapperGeneratorStrings(WrapperGeneratorStrings):
    def __init__(self):
        super().__init__()

        self.torch_mlir_code_body = """
int main() {{
{tensor_definition_str}
{result_struct_init}
{read_from_file_calls}
{benchmarking_and_reporting}
{update_tensor}
{write_to_file_calls}
{lwp}
return 0;
}}
"""


class TorchMlirHexagonWrapperGenerator(HexagonWrapperGenerator):
    def __init__(self, input_profs, iterations, func_name, output_profs, options: dict):
        self.options = options
        self.use_out_params = options.get("enableBufferResultsToOutParams", False)
        super().__init__(
            input_profs,
            iterations,
            func_name,
            output_profs,
            TorchMLIRWrapperGeneratorStrings(),
            options,
        )

    def _output_memrefdesc_name(self, idx):
        return f"odt{idx}"

    def generate_llvm_function_signature(self):
        """Generate lowered LLVM function definition to be called from CPP launcher"""
        if self.use_out_params:
            # buffer-results-to-out-params ABI: the function returns void and the
            # memref results are appended as trailing out-param arguments.
            function_arg_string = self.generate_llvm_function_signature_arg_string()
            for out in self.output_profs:
                if not out.rank:
                    continue
                function_arg_string += (
                    ", "
                    + self.common_strings.extern_llvm_func_with_return_args.format(
                        tensor_ctype=out.dtype, tensor_rank=out.rank
                    )[:-2]
                )
            return self.common_strings.extern_llvm_func_defn.format(
                kernel_name=self.func_name, function_arg_string=function_arg_string
            )
        function_arg_string = ""
        for out in self.output_profs:
            function_arg_string += self.common_strings.extern_llvm_func_with_return_args.format(
                tensor_ctype=out.dtype, tensor_rank=out.rank
            )
        function_arg_string += self.generate_llvm_function_signature_arg_string()
        return self.common_strings.extern_llvm_func_defn.format(
            kernel_name=self.func_name, function_arg_string=function_arg_string
        )

    def generate_llvm_function_call(self):
        """Generates actual function call to lowered LLVM function call"""
        if self.use_out_params:
            # Inputs first, then the caller-allocated output descriptors as
            # trailing out-params.
            function_call_descriptor_string = (
                self.generate_llvm_function_call_arg_string()
            )
            for idx, out in enumerate(self.output_profs):
                if not out.rank:
                    continue
                function_call_descriptor_string += (
                    ", " + self._output_memrefdesc_name(idx)
                )
            return self.common_strings.extern_llvm_func_call.format(
                func_name=self.func_name,
                descriptor_string=function_call_descriptor_string,
            )
        function_call_descriptor_string = ""
        for i in range(len(self.output_profs)):
            function_call_descriptor_string += f"&(r->r{i}), "
        function_call_descriptor_string += self.generate_llvm_function_call_arg_string()
        return self.common_strings.extern_llvm_func_call.format(
            func_name=self.func_name, descriptor_string=function_call_descriptor_string
        )

    def generate_result_struct(self):
        # In the out-param ABI the outputs are caller-allocated Tensors, so no
        # FuncResult struct is needed.
        if self.use_out_params:
            return ""
        return super().generate_result_struct()

    def generate_result_struct_init(self):
        # Allocate the output buffers on the host (caller) with the known static
        # shape and expose their memref descriptors to pass as out-params.
        if self.use_out_params:
            alloc_string = ""
            for idx, out in enumerate(self.output_profs):
                if not out.rank:
                    continue
                sizes, strides = get_shape(out.shape)
                alloc_string += (
                    f"Tensor<{out.dtype}, {out.rank}> "
                    f"{self.common_strings.output_tensor_name}{idx}"
                    f"({{{sizes}, {strides}, 128}}, MemType::HEAP);\n"
                    f"MemRefDescriptor<{out.dtype}, {out.rank}> *"
                    f"{self._output_memrefdesc_name(idx)} = "
                    f"{self.common_strings.output_tensor_name}{idx}.toMemRefDesc();\n"
                )
            return alloc_string
        return super().generate_result_struct_init()

    def generate_update_tensor_calls(self):
        # Out-param outputs (O{idx}) are already live Tensors written in place by
        # the callee, so nothing to re-wrap before dumping.
        if self.use_out_params:
            return ""
        return super().generate_update_tensor_calls()

    def generate_l2_scheduler_report(self):
        if not self.options.get("enableOmniFetchVDAE", False):
            return ""
        return """
uint64_t l2_scheduler_counts = __omni_fetch_l2_scheduler_counts();
uint64_t l2_scheduler_limits = __omni_fetch_l2_scheduler_limits();
uint64_t l2_requested_bytes = __omni_fetch_l2_requested_bytes();
uint64_t l2_issued_bytes = __omni_fetch_l2_issued_bytes();
FILE *l2_scheduler_report = fopen("perf.txt", "a");
if (l2_scheduler_report) {
  fprintf(l2_scheduler_report,
          "OmniFetchL2Scheduler: issued=%u busy_suppressed=%u "
          "page_clipped=%u unsupported=%u requested_bytes=%llu "
          "issued_bytes=%llu\\n",
          (unsigned)(l2_scheduler_counts >> 32),
          (unsigned)l2_scheduler_counts,
          (unsigned)(l2_scheduler_limits >> 32),
          (unsigned)l2_scheduler_limits,
          (unsigned long long)l2_requested_bytes,
          (unsigned long long)l2_issued_bytes);
  fclose(l2_scheduler_report);
}
"""

    def generate_benchmarking_and_reporting(self, function_call):
        if not self.options.get("enableOmniFetchPersistentWhCache", False):
            return (
                super().generate_benchmarking_and_reporting(function_call)
                + self.generate_l2_scheduler_report()
            )
        context = int.from_bytes(
            hashlib.sha256(self.func_name.encode("utf-8")).digest()[:8], "little"
        )
        generation = int(self.options.get("omniFetchWhCacheGeneration", 1))
        return f"""
__omni_fetch_wh_cache_set_context(UINT64_C({context}), {generation}u);
uint64_t cold_time_us = benchmark_time_us(1, [&]() {{
    {function_call}
}});
uint64_t cold_stats = __omni_fetch_wh_cache_stats();
uint64_t cold_w8_stats = __omni_fetch_w8_cache_stats();
std::vector<uint64_t> __warm_samples;
uint64_t warm_time_us = benchmark_samples_us({self.iterations}, __warm_samples, [&]() {{
    {function_call}
}});
uint64_t wh_cache_stats = __omni_fetch_wh_cache_stats();
uint64_t w8_cache_stats = __omni_fetch_w8_cache_stats();
__omni_fetch_wh_cache_invalidate(UINT64_C({context}), {generation}u);
uint64_t invalidated_time_us = benchmark_time_us(1, [&]() {{
    {function_call}
}});
uint64_t invalidated_stats = __omni_fetch_wh_cache_stats();
TestReport tr("{self.func_name}", warm_time_us, "us", Result::Pass);
tr.save();
FILE *wh_cache_report = fopen("perf.txt", "a");
if (wh_cache_report) {{
  fprintf(wh_cache_report,
          "OmniFetchWHCache: cold_us=%llu warm_avg_us=%llu "
          "cold_hits=%u cold_misses=%u total_hits=%u total_misses=%u "
          "invalidated_us=%llu "
          "post_invalidate_hits=%u post_invalidate_misses=%u\\n",
          (unsigned long long)cold_time_us,
          (unsigned long long)warm_time_us,
          (unsigned)(cold_stats >> 32), (unsigned)cold_stats,
          (unsigned)(wh_cache_stats >> 32), (unsigned)wh_cache_stats,
          (unsigned long long)invalidated_time_us,
          (unsigned)(invalidated_stats >> 32), (unsigned)invalidated_stats);
  fclose(wh_cache_report);
}}
FILE *w8_cache_report = fopen("perf.txt", "a");
if (w8_cache_report) {{
  fprintf(w8_cache_report,
          "OmniFetchW8Cache: cold_hits=%u cold_misses=%u "
          "total_hits=%u total_misses=%u\\n",
          (unsigned)(cold_w8_stats >> 32), (unsigned)cold_w8_stats,
          (unsigned)(w8_cache_stats >> 32), (unsigned)w8_cache_stats);
  fclose(w8_cache_report);
}}
{{
  FILE *__warm_perf_fp = fopen("perf.txt", "a");
  report_percentiles("{self.func_name}", __warm_samples, __warm_perf_fp);
  if (__warm_perf_fp) fclose(__warm_perf_fp);
}}
""" + self.generate_l2_scheduler_report()

    def generate_input_wrapper_struct_def(self):
        """Generates template for input wrapper structs"""
        return ""

    def generate_input_wrapper_structs_init(self):
        """Generates initializations for input wrapper structs"""
        return ""

    def generate_cpp_wrapper(self, file_name, exec_dir):
        """
        Generates skeleton cpp file which launches the kernel.
        """
        code_headers = self.common_strings.code_headers
        if self.options.get("enableOmniFetchPersistentWhCache", False):
            code_headers += """
// If the cost model selects no persistent site, no OmniFetch op pulls the
// device runtime bitcode into the kernel object.  Keep reporting calls
// loadable in that legitimate no-op case.  Strong runtime definitions replace
// these weak fallbacks whenever a transformed site actually exists.
extern "C" __attribute__((weak)) void
__omni_fetch_wh_cache_set_context(uint64_t, uint32_t) {}
extern "C" __attribute__((weak)) void
__omni_fetch_wh_cache_invalidate(uint64_t, uint32_t) {}
extern "C" __attribute__((weak)) uint64_t
__omni_fetch_wh_cache_stats(void) { return 0; }
extern "C" __attribute__((weak)) uint64_t
__omni_fetch_w8_cache_stats(void) { return 0; }
"""
        if self.options.get("enableOmniFetchVDAE", False):
            code_headers += """
extern "C" __attribute__((weak)) uint64_t
__omni_fetch_l2_scheduler_counts(void) { return 0; }
extern "C" __attribute__((weak)) uint64_t
__omni_fetch_l2_scheduler_limits(void) { return 0; }
extern "C" __attribute__((weak)) uint64_t
__omni_fetch_l2_requested_bytes(void) { return 0; }
extern "C" __attribute__((weak)) uint64_t
__omni_fetch_l2_issued_bytes(void) { return 0; }
"""

        code_define = self.common_strings.code_define.format(
            llvm_func_sign=self.generate_llvm_function_signature(),
            input_wrapper_struct_def="",
            result_struct_def=self.generate_result_struct(),
        )

        code_body = self.common_strings.torch_mlir_code_body.format(
            tensor_definition_str=self.generate_input_declarations(),
            input_wrapper_structs_init="",
            result_struct_init=self.generate_result_struct_init(),
            read_from_file_calls=self.generate_tensor_read_from_file_calls(
                file_name, exec_dir
            ),
            benchmarking_and_reporting=self.generate_benchmarking_and_reporting(
                self.generate_llvm_function_call()
            ),
            write_to_file_calls=self.generate_tensor_write_to_file_calls(
                file_name, exec_dir
            ),
            update_tensor=self.generate_update_tensor_calls(),
            lwp=self.generate_lwp_call(exec_dir),
        )

        return self.common_strings.code_string.format(
            code_headers=code_headers, code_define=code_define, code_body=code_body
        )


class TorchMLIRHexagonLauncher(HexagonLauncherBase):
    # Lower mlir module to (potentially several) object codes
    # which are returned in a vector, each as vector<char>
    def mlir_to_obj(self, mlir_mod: ir.module, options: dict) -> list[bytes]:
        # Must cast options to strings before passing to backend
        options = {k: str(v) for k, v in options.items()}
        modules_compiled_as_bytes = qcom_hexagon_backend.translate_linalg_to_obj(
            mlir_mod, options
        )
        return modules_compiled_as_bytes

    # Compile the mlir bytecode from file `mlir_bytecode_path` that lives in `local_dir`,
    # whose principal function to call is `func_name`, using the HexagonExecutor `hexec`.
    # Returns the TorchMlirHexagonWrapperGenerator used, and the collection of paths to
    # the shared libraries generated.
    # TODO: the HexagonExecutor `hexec` is passed because that's still the class that
    #        implements the method generate_shared_object(). Change that!
    def compile_torch_mlir(
        self,
        hexec: HexagonExecutor,
        local_dir: str,
        mlir_bytecode_path: str,
        inputs: list[Tensor],
        func_name: str,
        options: dict,
        iterations: int = 1,
    ) -> tuple[TorchMlirHexagonWrapperGenerator, list[str]]:
        # Context is being initialized through triton source code - python/src/ir.cc
        # This creates a dependency on triton code and
        # will cause issue when we separate torch-mlir workflow from triton.
        # Todo: We need to define the context in hexagon backend.
        context = ir.context()
        if options.get("lowerConstantsInSeparateSharedObjects", False):
            context.disable_multithreading()
        qcom_hexagon_backend.load_dialects(context)
        mlir_mod = qcom_hexagon_backend.parse_mlir_module_from_file(
            mlir_bytecode_path, context
        )
        # Careful, this needs to be done before calling mlir_to_obj()
        result_types = qcom_hexagon_backend.get_return_list(mlir_mod, func_name)
        result_shapes = qcom_hexagon_backend.get_return_shapes(mlir_mod, func_name)

        # Guard against lifted-constant function arguments.  torch-mlir's frozen
        # export inlines nn.Parameters and plain constant tensors, but lifts
        # register_buffer buffers (persistent or not) into leading function
        # arguments.  The host wrapper only supplies the user inputs, so any
        # extra lifted argument is passed a NULL descriptor and faults on device
        # (Bad VA 0x0).  Detect the mismatch here and fail loudly with guidance.
        arg_list = qcom_hexagon_backend.get_arg_list(mlir_mod, func_name)
        arg_shapes = qcom_hexagon_backend.get_arg_shapes(mlir_mod, func_name)
        if len(arg_list) != len(inputs):
            extra = [
                arg_shapes[i] if i < len(arg_shapes) else "?"
                for i in range(len(inputs), len(arg_list))
            ]
            raise ValueError(
                f"Entry function '{func_name}' expects {len(arg_list)} arguments "
                f"but the host provided {len(inputs)} input(s). This usually means "
                f"torch-mlir lifted register_buffer buffers into function "
                f"arguments (extra arg shapes: {extra}). Store frozen/derived "
                f"tensors as plain attributes or nn.Parameters (not buffers) so "
                f"they are inlined as constants instead of becoming arguments."
            )

        func_name_with_ciface = "_mlir_ciface_" + func_name

        # MLIR to (potentially several) object files (as strings)
        obj_modules: list[bytes] = self.mlir_to_obj(mlir_mod, options)
        # We now need to link each object code obtained (separately!), to obtain a collection of shared objects (.so)

        # We will populate this collection with the path of the generated shared libraries
        # (there will be only 1 .so if options["lowerConstantsInSeparateSharedObjects"] is False,
        # and multiple ones otherwise)
        paths_to_shared_libs_generated: list[str] = []
        # Every shared object we will build has no dependencies (hence the default to []) except for the principal module
        lib_dependencies_paths = []

        nb_modules_compiled = len(obj_modules)
        print(
            "We have",
            nb_modules_compiled,
            "object files obtained from the MLIR->obj compilation",
            (
                "that we now need to link independently into their own shared object (.so)."
                if nb_modules_compiled > 1
                else ""
            ),
        )
        # Dealing with each object code that has been generated by mlir_to_obj()
        # Note: we reverse obj_modules to start with the "constants-only" modules and to finish with the principal module
        # since the principal one will need some -l annotations
        for i, kernel_obj_as_bytes in enumerate(reversed(obj_modules)):
            # Convert the index since we've had to reverse obj_modules to finish with the principal module
            i = nb_modules_compiled - 1 - i
            print("------ Starting work on compiled module number", i + 1, "------")
            # 1 - Writting the object code for the kernel to disk
            # The principal module containg the code will keep the name of the function
            # but the modules containg only constants will be called with a suffix ([...]-consts-1.o, [...]-consts-2.o, etc)
            filename_obj = (
                func_name_with_ciface
                if (i == 0)
                else func_name_with_ciface + "-consts-" + str(i)
            )
            obj_src_path = os.path.join(local_dir, filename_obj + ".o")
            Path(obj_src_path).write_bytes(kernel_obj_as_bytes)
            print(f"==> kernel obj saved in: {obj_src_path}")

            cpp_wrapper_path = None
            # 2 - If it's the principal module only, deal with the wrapper code that we need to generate and dump
            if i == 0:
                # Creating the wrapper generator for the principal module
                return_types = parse_return_types(result_types, result_shapes)
                input_profs = profile_torch_mlir_inputs(inputs)

                # Pass options to the wrapper generator.
                # HexagonWrapperGenerator will create the call to WriteLWPOutput() if lwp is enabled.
                wrapper_generator = TorchMlirHexagonWrapperGenerator(
                    input_profs,
                    iterations,
                    func_name_with_ciface,
                    return_types,
                    options,
                )
                print("==> Wrapper generator correctly instanciated")

                exec_dir = "." if hexec.exec_mode == "device" else local_dir

                # Generating and writing the wrapper to disk
                cpp_wrapper_path = self.generate_and_dump_wrapper(
                    wrapper_generator, local_dir, func_name_with_ciface, exec_dir
                )
                print(
                    f"==> Generated a cpp wrapper to act as kernel starter: {cpp_wrapper_path}"
                )

                # The main module has for dependencies all the .so that we just built up to this point (from nb_modules_compiled-1 to 1)
                lib_dependencies_paths = paths_to_shared_libs_generated

            # 3 - Generate the corresponding shared object for the current object file
            so_path = hexec.generate_shared_object(
                cpp_wrapper_path, obj_src_path, lib_dependencies_paths
            )  # cpp_wrapper_path is None if that's not the principal module
            print("==> Shared object generated: ", so_path)
            # Add it to the collection of paths to the shared libraries that have been generated
            paths_to_shared_libs_generated.append(so_path)
        # We reverse the paths_to_shared_libs_generated to have them back in the normal order: principal first, then constants-only ones
        return (wrapper_generator, list(reversed(paths_to_shared_libs_generated)))

    # Toplevel function that compiles the mlir and execute the result.
    # Executes either on device or on simulator depending on the env var RUN_ON_SIM.
    def run_torch_mlir(
        self,
        mlir_bytecode_path: str,
        inputs: list[Tensor],
        func_name: str,
        base_dir_for_artifacts: Optional[str] = None,
        iterations: int = 1,
        options: dict = None,
        enable_etm=False,
    ) -> list[Tensor]:
        if options is None:
            options = HexagonOptions().__dict__

        # Pass lwp related info to HexagonExecutor() for creating shared obj and pulling lwp.json file if enabled.
        hexec = HexagonExecutor(options["enableLWP"], enable_etm)
        local_dir_path = create_timestamped_folder(func_name, base_dir_for_artifacts)

        filename = os.path.basename(mlir_bytecode_path)
        destination_path = os.path.join(local_dir_path, filename)
        # Copy the initial MLIR code to the timestamped folder
        shutil.copy(mlir_bytecode_path, destination_path)

        start_time = time.time()
        # Compile the mlir bytecode from file `mlir_bytecode_path`
        (wrapper_generator, paths_to_shared_libs_generated) = self.compile_torch_mlir(
            hexec,
            local_dir_path,
            mlir_bytecode_path,
            inputs,
            func_name,
            options,
            iterations,
        )
        end_time = time.time()
        print(
            f"Compilation from initial MLIR to .so took {end_time - start_time:.4f} seconds",
            flush=True,
        )

        func_name_with_ciface = "_mlir_ciface_" + func_name
        # Execute the kernel using the HexagonExecutor `hexec`
        results = self.execute_kernel(
            hexec,
            local_dir_path,
            func_name_with_ciface,
            paths_to_shared_libs_generated,
            wrapper_generator,
        )
        return results

    @staticmethod
    def _parse_perf_file(perf_path: str) -> dict:
        """Parse a pulled <lib>_perf.txt into {mean, p50, p90, p99, pmin,
        samples}. Missing fields are returned as None. Anchors match the
        additive lines emitted by report_percentiles()."""
        fields = {
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "pmin": None,
            "samples": None,
        }
        key_by_prefix = {
            "Perf:": "mean",
            "PerfP50:": "p50",
            "PerfP90:": "p90",
            "PerfP99:": "p99",
            "PerfMin:": "pmin",
            "PerfSamples:": "samples",
        }
        try:
            with open(perf_path, "r") as fp:
                for raw in fp:
                    line = raw.strip()
                    for prefix, key in key_by_prefix.items():
                        if line.startswith(prefix):
                            try:
                                fields[key] = float(line[len(prefix):].strip())
                            except ValueError:
                                pass
                            break
        except FileNotFoundError:
            pass
        return fields

    @staticmethod
    def _percentiles(values: list) -> dict:
        """Host-side p50/p90/p99/min over a list of per-round means."""
        if not values:
            return {"p50": None, "p90": None, "p99": None, "min": None, "n": 0}
        s = sorted(values)
        n = len(s)

        def pct(p):
            if n == 1:
                return s[0]
            idx = p * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return s[lo] + frac * (s[hi] - s[lo])

        return {
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "min": s[0],
            "n": n,
        }

    # Compile each profile once, then round-robin execute so thermal/DVFS drift
    # is shared across configs rather than penalizing whichever ran last.
    def run_torch_mlir_interleaved(
        self,
        configs_by_profile: dict,
        inputs: list[Tensor],
        func_name: str,
        base_dir_for_artifacts: Optional[str] = None,
        iterations: int = 1,
        rounds: int = 1,
        enable_etm=False,
    ) -> dict:
        # configs_by_profile maps a profile label to a dict with keys
        # "launch_path" (the MLIR bytecode/text to compile for that profile,
        # which may differ because HexKL profiles compile rewritten IR) and
        # "options" (the HexagonOptions dict for that profile).
        # Phase A: compile every profile once.
        compiled = {}
        for profile, cfg in configs_by_profile.items():
            options = cfg.get("options") or HexagonOptions().__dict__
            launch_path = cfg["launch_path"]
            hexec = HexagonExecutor(options["enableLWP"], enable_etm)
            local_dir_path = create_timestamped_folder(
                f"{func_name}_{profile}", base_dir_for_artifacts
            )
            filename = os.path.basename(launch_path)
            shutil.copy(launch_path, os.path.join(local_dir_path, filename))
            start_time = time.time()
            (wrapper_generator, so_paths) = self.compile_torch_mlir(
                hexec,
                local_dir_path,
                launch_path,
                inputs,
                func_name,
                options,
                iterations,
            )
            end_time = time.time()
            print(
                f"==> [interleave] Compilation from initial MLIR to .so for "
                f"profile '{profile}' took {end_time - start_time:.4f} seconds",
                flush=True,
            )
            compiled[profile] = {
                "hexec": hexec,
                "local_dir": local_dir_path,
                "so_paths": so_paths,
                "wrapper_generator": wrapper_generator,
            }

        func_name_with_ciface = "_mlir_ciface_" + func_name

        # Phase B: round-robin execute; snapshot each config's perf per round.
        per_round_means = {profile: [] for profile in compiled}
        last_results = {}
        for r in range(rounds):
            for profile, c in compiled.items():
                results = self.execute_kernel(
                    c["hexec"],
                    c["local_dir"],
                    func_name_with_ciface,
                    c["so_paths"],
                    c["wrapper_generator"],
                )
                last_results[profile] = results
                principal = c["so_paths"][0]
                ld, base, _ = split_path(principal)
                perf_path = os.path.join(ld, f"{base}_perf.txt")
                parsed = self._parse_perf_file(perf_path)
                if parsed["mean"] is not None:
                    per_round_means[profile].append(parsed["mean"])
                print(
                    f"==> [interleave] round {r + 1}/{rounds} profile "
                    f"'{profile}' mean_us={parsed['mean']}",
                    flush=True,
                )

        # Aggregate host-side percentiles over per-round means.
        print("==> [interleave] Per-profile summary (percentiles over rounds):")
        for profile, means in per_round_means.items():
            agg = self._percentiles(means)
            print(
                f"InterleaveResult profile={profile} rounds={agg['n']} "
                f"p50_us={agg['p50']} p90_us={agg['p90']} p99_us={agg['p99']} "
                f"min_us={agg['min']}",
                flush=True,
            )

        return last_results

    @staticmethod
    def get_output_tensor_path_count(wrapper_generator: HexagonWrapperGenerator):
        """Returns number of output paths by checking rank (dim)"""
        return sum([1 for out in wrapper_generator.output_profs if out.rank])
