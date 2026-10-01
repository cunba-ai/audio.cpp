@echo off
rem Build + run the fork-regression ctest suite after an upstream merge.
rem Needs a configure with ENGINE_BUILD_TESTS=ON first, e.g.:
rem   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows.ps1 ^
rem     -Preset windows-cpu-release -Target audiocpp -BuildTests ON -Jobs 2
rem This script only sets up the MSVC env (vcvars) and builds the labeled
rem test targets, then runs ctest -L fork_regression.
setlocal

rem VS Community vcvars (adjust if the edition moves); cmake/ctest from mingw64.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=D:\dev-tools\mingw64\bin;%PATH%

cd /d "%~dp0..\.."

cmake --build build/windows-cpu-release -j 2 --target ^
  asr_vad_model_path_test capi_enum_sync_test capi_shared_lib_surface_test ^
  tensor_source_memory_backed_test progress_callback_test ^
  backend_weight_store_commit_test capi_option_number_test capi_session_options_test ^
  silero_vad_loader_routing_test capi_denoise_embedded_test fork_policy_test ^
  fork_gru_scan_test fork_round_bf16_test fork_sycl_reorder_getrows_test ^
  fork_sycl_fattn_per_head_mask_test fork_sycl_concat_blocks_test ^
  fork_sycl_fattn_mkl_gate_test ^
  fork_vulkan_dispatch_clamp_test fork_backend_family_resolver_test ^
  fork_source_anchors_test fork_irodori_codec_crop_test
if errorlevel 1 exit /b 1

ctest --test-dir build/windows-cpu-release -L fork_regression --output-on-failure
