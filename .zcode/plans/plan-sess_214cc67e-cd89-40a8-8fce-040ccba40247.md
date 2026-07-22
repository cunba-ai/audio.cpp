# Rewrite `capi-build.yml` borrowing llama.cpp's reliable toolchain-install patterns

**Goal:** Keep audio.cpp's single-backend C ABI shared-library build (4 matrix entries + draft release job), but replace the fragile Windows ROCm/SYCL install steps with llama.cpp's *proven* install approach (the immediate motivation: the oneAPI pinned URL is dead → `Access Denied`).

**Decisions locked (from Q&A):**
- Borrow install patterns only — **no** Vulkan co-build, **no** `GGML_BACKEND_DL`.
- **Windows ROCm** → adopt llama.cpp's real AMD HIP SDK `.exe` install.
- **Windows SYCL** → adopt llama.cpp's offline oneAPI `.exe` + Level Zero SDK + icx/icpx.
- **Windows CUDA** → leave as-is (`Jimver/cuda-toolkit` works; no reported failure).
- **Do NOT** port the llama.cpp OSS/ossutil publish job (audio.cpp has no OSS infra/secrets; it uses a GitHub draft release).
- All SYCL targets already corrected to `INTEL` (done in prior commit).

## File: `.github/workflows/capi-build.yml` (full rewrite)

### Top-level changes
- Bump action versions to match the reference: `actions/checkout@v4 → v6`, `actions/upload-artifact@v4 → v6`, `actions/download-artifact@v4 → v7`.
- Add `permissions: { contents: read, actions: write }` (needed if we later add caches; harmless now).
- Add `concurrency: { group: capi-build-${{ github.ref }}, cancel-in-progress: true }` (already partly present; normalize).
- Add `env:` block with shared constants: `CUDA_ARCHS`, `AMDGPU_TARGETS` (gfx1100;gfx1101;gfx1102;gfx1150;gfx1151 — superset of today's single target, matching reference coverage).
- Keep the existing trigger (push/PR to main/dev/capi/** + workflow_dispatch release_tag input).

### Linux job — minimal changes (it already works in containers)
- Bump checkout/upload-artifact versions.
- Keep matrix (CUDA `nvidia/cuda:12.4.1-devel`, ROCm `rocm/dev-ubuntu-22.04:6.4-complete`, SYCL `intel/oneapi-basekit:2024.2.1`).
- Add `ccache` step (`ggml-org/ccache-action@v1.2.21`) per backend key, to match reference iteration speed. (Linux already builds fast in containers; this is a low-risk nicety.)
- Keep `-DGGML_SYCL_TARGET=INTEL` (already fixed).
- Keep the existing artifact collection (`libaudiocpp.so` + `audiocpp.h`, `strip`).

### Windows job — CUDA entry: unchanged logic, bump versions only
- Keep `Setup CUDA Toolkit` via `Jimver/cuda-toolkit@v0.2.21`, keep `Configure (CUDA)`/Build steps.

### Windows job — ROCm entry: **rewrite install** to llama.cpp's pattern
- Replace the current winget-attempt + scan + fail-fast gate with:
  - `env: HIPSDK_INSTALLER_VERSION: "26.Q1"` on the job.
  - Cache `C:\Program Files\AMD\ROCm` keyed `cache-gha-rocm-${{ env.HIPSDK_INSTALLER_VERSION }}-${{ runner.os }}`.
  - `Install ROCm`: download `https://download.amd.com/developer/eula/rocm-hub/AMD-Software-PRO-Edition-${{ env.HIPSDK_INSTALLER_VERSION }}-Win11-For-HIP.exe`, silent `-install`, `WaitForExit(600000)` with explicit timeout→kill, non-zero exit → hard error (fail-loudly preserved).
  - Keep `continue-on-error: true` (non-gating).
  - `Configure (ROCm)`: resolve `HIP_PATH` from `Resolve-Path 'C:\Program Files\AMD\ROCm\*\bin\clang.exe'`, set `CMAKE_PREFIX_PATH`, keep `-DGGML_HIP=ON -DGPU_TARGETS=...`, generator `"Unix Makefiles"` (clang), matching the reference.

### Windows job — SYCL entry: **rewrite install** to llama.cpp's pattern
- Replace the current winget `Intel.OneAPI.BaseToolkit` step (slow + untested) with the reference's proven path:
  - `env:` block with `WINDOWS_BASEKIT_URL` (the `intel-deep-learning-essentials-2025.3.3.18_offline.exe` URL the reference currently pins — **known-good**), `WINDOWS_DPCPP_MKL` component list, `LEVEL_ZERO_SDK_URL`, `ONEAPI_ROOT`.
  - `Download & Install oneAPI`: **inline** the `install-oneapi.bat` logic (audio.cpp has no such script), via a `bash` step: `curl` the `.exe`, run `Start-Process ... -ArgumentList '-s -a --silent --components <list> --eula accept'`, verify `setvars.bat` exists else fail loudly.
  - `Install Level Zero SDK`: download + `Expand-Archive` to `C:/level-zero-sdk`, set `LEVEL_ZERO_V1_SDK_PATH` (vendored ggml has `GGML_SYCL_SUPPORT_LEVEL_ZERO=ON`).
  - `Configure (SYCL)`: `call setvars.bat intel64 --force`, `-DCMAKE_CXX_COMPILER=icx` / `-DCMAKE_CXX_COMPILER=icpx` (Windows uses icx via cl frontend), `-DGGML_SYCL=ON -DGGML_SYCL_F16=OFF -DGGML_SYCL_TARGET=INTEL`.
  - Keep `continue-on-error: true`.

### Release job — unchanged logic, bump versions
- Bump download-artifact→v7, action-gh-release stays @v2. Keep draft-release flow (do NOT add OSS).

## Not carried over from the reference (and why)
- `GGML_BACKEND_DL`, `GGML_CUDA`/`GGML_HIP`/`GGML_SYCL` direct flags → audio.cpp uses `AUDIOCPP_BUILD_CAPI` + `ENGINE_ENABLE_*`; I translate, not copy.
- Vulkan SDK install + `GGML_VULKAN` everywhere → "borrow patterns only" decision.
- `pack artifacts` VERSION file + `install.bat`/`install.sh` + `7z`/`tar` archives → audio.cpp ships bare `libaudiocpp.so`/`audiocpp.dll`+`.h`, no installer.
- `publish-oss` job → no OSS infra in this repo.
- `paths:` filter on the trigger → optional; I'll add a narrow one (CMakeLists / capi/** / workflows) so unrelated commits don't rebuild all backends. *(Can drop if you prefer always-build.)*

## Verification
- YAML lint: I'll validate with `python -c "import yaml,sys; yaml.safe_load(open(...))"` after writing.
- `grep -n "INTEL_XMX"` → expect none.
- Confirm every backend's CMake line maps to an audio.cpp-supported flag (no `GGML_BACKEND_DL`).
- No commit unless you say so; you'll review the diff.

## Risk / follow-ups
- The pinned oneAPI `.exe` URL will rot again on Intel's schedule — same structural issue, but this URL is current as of llama.cpp's last green run and is the same one that works there. If it dies, swap the URL constant.
- Windows ROCm/HIP still can't *execute* (no AMD GPU on hosted runners) — only proves configure/compile. Non-gating.
- Adding `ccache` to Linux is additive; if it causes issues it's safe to remove without touching the install logic.