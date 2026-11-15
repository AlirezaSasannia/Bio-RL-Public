@echo off
rem Stage C3b chain: arms -> test -> report, each step only if the previous one succeeded.
rem Re-running this file resumes from the per-generation checkpoints and the evaluation cache.
rem Logs: 04_code\results\logs\single\stageC3b_<step>.log / .err
cd /d "%~dp0"
set C3_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_c3b.py arms   >> "%L%\stageC3b_arms.log"   2>> "%L%\stageC3b_arms.err"   || exit /b 1
"%PY%" -u stage_c3b.py test   >> "%L%\stageC3b_test.log"   2>> "%L%\stageC3b_test.err"   || exit /b 2
"%PY%" -u stage_c3b.py report >> "%L%\stageC3b_report.log" 2>> "%L%\stageC3b_report.err" || exit /b 3
echo C3b chain finished >> "%L%\stageC3b_arms.log"
