@echo off
rem Stage C3 chain: arms -> test -> savings -> report, each step only if the previous one succeeded.
rem Re-running this file resumes: arm runs continue from their per-generation checkpoints, evaluations from the cache.
rem Logs: 04_code\results\logs\single\stageC3_<step>.log / .err
cd /d "%~dp0"
set C3_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_c3.py arms    >> "%L%\stageC3_arms.log"    2>> "%L%\stageC3_arms.err"    || exit /b 1
"%PY%" -u stage_c3.py test    >> "%L%\stageC3_test.log"    2>> "%L%\stageC3_test.err"    || exit /b 2
"%PY%" -u stage_c3.py savings >> "%L%\stageC3_savings.log" 2>> "%L%\stageC3_savings.err" || exit /b 3
"%PY%" -u stage_c3.py report  >> "%L%\stageC3_report.log"  2>> "%L%\stageC3_report.err"  || exit /b 4
echo C3 chain finished >> "%L%\stageC3_arms.log"
