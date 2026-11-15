@echo off
rem Stage E2-0 gate: calibrate -> validate -> headroom (each resumable from its cache; re-run this file to resume).
rem Log: 04_code\results\logs\single\stageE2_0.log / .err ; outputs in stageE2_0\
cd /d "%~dp0"
set E2_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_e2_0.py all >> "%L%\stageE2_0.log" 2>> "%L%\stageE2_0.err" || exit /b 1
echo E2-0 chain finished >> "%L%\stageE2_0.log"
