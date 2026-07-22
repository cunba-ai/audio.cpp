::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT
::
::  Vendored from llama.cpp (scripts/install-oneapi.bat) — Intel's official
::  silent-install helper for the oneAPI offline bootstrapper. Takes:
::    %1 = URL of the offline installer .exe
::    %2 = colon-separated component list (may be empty for full install)
::  Used by .github/workflows/capi-build.yml for the Windows SYCL job.

set URL=%1
set COMPONENTS=%2

curl.exe --output %TEMP%\webimage.exe --url %URL% --retry 5 --retry-delay 5
start /b /wait %TEMP%\webimage.exe -s -x -f webimage_extracted --log extract.log
del %TEMP%\webimage.exe
if "%COMPONENTS%"=="" (
  webimage_extracted\bootstrapper.exe -s --action install --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=0 --log-dir=.
) else (
  webimage_extracted\bootstrapper.exe -s --action install --components=%COMPONENTS% --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=0 --log-dir=.
)
set installer_exit_code=%ERRORLEVEL%
rd /s/q "webimage_extracted"
exit /b %installer_exit_code%
