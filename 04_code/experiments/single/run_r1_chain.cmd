@echo off
rem Stage R1 (optional robustness, original world): arms -> test -> report; resumable (re-run to resume).
rem Register FIRST (python stage_r1.py register) and commit the yaml.
cd /d "%~dp0"
set R1_SMOKE=
set C3_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_r1.py arms   >> "%L%\stageR1_arms.log"   2>> "%L%\stageR1_arms.err"   || exit /b 1
"%PY%" -u stage_r1.py test   >> "%L%\stageR1_test.log"   2>> "%L%\stageR1_test.err"   || exit /b 2
"%PY%" -u stage_r1.py report >> "%L%\stageR1_report.log" 2>> "%L%\stageR1_report.err" || exit /b 3
echo R1 chain finished >> "%L%\stageR1_arms.log"
