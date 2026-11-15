@echo off
rem Stage C3c chain: arms -> test -> report, each step only if the previous one succeeded.
rem Re-running this file resumes from the per-generation checkpoints and the evaluation cache.
rem Logs: 04_code\results\logs\single\stageC3c_<step>.log / .err
cd /d "%~dp0"
set C3_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_c3c.py arms   >> "%L%\stageC3c_arms.log"   2>> "%L%\stageC3c_arms.err"   || exit /b 1
"%PY%" -u stage_c3c.py test   >> "%L%\stageC3c_test.log"   2>> "%L%\stageC3c_test.err"   || exit /b 2
"%PY%" -u stage_c3c.py report >> "%L%\stageC3c_report.log" 2>> "%L%\stageC3c_report.err" || exit /b 3
echo C3c chain finished >> "%L%\stageC3c_arms.log"
