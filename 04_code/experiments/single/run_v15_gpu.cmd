@echo off
rem Revision v1.5 GPU chain (family single_revision_v1_5): X6 network solutions on the GPU subset -> report.
rem Resumable (one cache file per arm in revV15\x6gpu_cache). Logs: 04_code\results\logs\single\revV15_x6gpu.log / .err
cd /d "%~dp0"
set V15_SMOKE=
set X2_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_x456.py x6gpu     >> "%L%\revV15_x6gpu.log"     2>> "%L%\revV15_x6gpu.err"     || exit /b 1
"%PY%" -u stage_x456.py report    >> "%L%\revV15_report_gpu.log" 2>> "%L%\revV15_report_gpu.err" || exit /b 2
echo V15 GPU chain finished >> "%L%\revV15_x6gpu.log"
