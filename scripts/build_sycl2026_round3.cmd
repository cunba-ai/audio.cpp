@echo off
:: Round3 clean full SYCL rebuild (vcvars + oneAPI 2026.1, -j8, serial target audiocpp)
setlocal enabledelayedexpansion

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

set ONEAPI_PREFIX=F:\sycl-oneapi-2026
set PATH=%ONEAPI_PREFIX%\compiler\2026.1\bin;%ONEAPI_PREFIX%\tbb\2023.1\bin;%ONEAPI_PREFIX%\mkl\2026.1\bin;%ONEAPI_PREFIX%\dnnl\2026.0\bin;%PATH%
set CMPLR_ROOT=%ONEAPI_PREFIX%\compiler\2026.1
set MKLROOT=%ONEAPI_PREFIX%\mkl\2026.1
set TBBROOT=%ONEAPI_PREFIX%\tbb\2023.1
set DNNL_ROOT=%ONEAPI_PREFIX%\dnnl\2026.0
set LEVEL_ZERO_V1_SDK_PATH=F:\sycl-oneapi\level-zero-sdk
set CMAKE_PREFIX_PATH=%CMPLR_ROOT%;%MKLROOT%;%TBBROOT%;%DNNL_ROOT%
set LIB=%CMPLR_ROOT%\lib;%TBBROOT%\lib;%MKLROOT%\lib;%DNNL_ROOT%\lib;%LIB%
set INCLUDE=%CMPLR_ROOT%\include;%TBBROOT%\include;%MKLROOT%\include;%DNNL_ROOT%\include;%INCLUDE%

cd /d F:\M1AO_Projects\audio.cpp

if "%1"=="configure" (
  cmake -S . -B build-sycl-2026 -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=icx ^
    -DBUILD_SHARED_LIBS=ON ^
    -DAUDIOCPP_BUILD_CAPI=ON ^
    -DAUDIOCPP_DEPLOYMENT_BUILD=ON ^
    -DENGINE_BUILD_EXAMPLES=OFF ^
    -DENGINE_ENABLE_NATIVE_CPU=OFF ^
    -DGGML_SYCL=ON ^
    -DGGML_SYCL_F16=OFF ^
    -DGGML_SYCL_TARGET=INTEL
  if errorlevel 1 exit /b 1
)

cmake --build build-sycl-2026 --parallel 8 --target audiocpp
exit /b %errorlevel%
