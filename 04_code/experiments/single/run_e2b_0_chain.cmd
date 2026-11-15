@echo off
rem Stage E2b-0 gate: calibrate -> validate -> headroom -> posctrl -> gate (each step resumable from its cache).
rem Register FIRST (python stage_e2b_0.py register) and commit the yaml; then start this file. Re-run it to resume.
rem Log: 04_code\results\logs\single\stageE2b_0.log / .err ; outputs in stageE2b_0\
cd /d "%~dp0"
set E2B_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_e2b_0.py all >> "%L%\stageE2b_0.log" 2>> "%L%\stageE2b_0.err" || exit /b 1
echo E2b-0 chain finished >> "%L%\stageE2b_0.log"
