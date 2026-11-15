@echo off
rem Stage C4 chain (Windows): base -> register_arms -> arms -> test -> report, each step only if the previous succeeded.
rem Precondition: stageE2b_0\gate_summary.csv says PASS; C4-0 registered (python stage_c4.py register_base) and the
rem yaml committed. The arms spec is registered by the chain after the bases are frozen: commit that yaml afterwards.
rem Re-running this file resumes from the per-generation checkpoints and the evaluation cache.
rem Bases per GPU call: set C4_BPC before registering (default 4) and keep it for the whole stage.
cd /d "%~dp0"
set C4_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_c4.py base          >> "%L%\stageC4_base.log"   2>> "%L%\stageC4_base.err"   || exit /b 1
"%PY%" -u stage_c4.py register_arms >> "%L%\stageC4_arms.log"   2>> "%L%\stageC4_arms.err"   || exit /b 2
"%PY%" -u stage_c4.py arms          >> "%L%\stageC4_arms.log"   2>> "%L%\stageC4_arms.err"   || exit /b 3
"%PY%" -u stage_c4.py test          >> "%L%\stageC4_test.log"   2>> "%L%\stageC4_test.err"   || exit /b 4
"%PY%" -u stage_c4.py report        >> "%L%\stageC4_report.log" 2>> "%L%\stageC4_report.err" || exit /b 5
echo C4 chain finished >> "%L%\stageC4_arms.log"
