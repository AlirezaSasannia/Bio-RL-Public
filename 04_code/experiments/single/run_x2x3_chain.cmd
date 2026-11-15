@echo off
rem Revision X2/X3 chain (family single_revision_v1_3), GPU: x3 -> arms -> x2test -> report -> family.
rem Re-running resumes from the per-generation checkpoint and the evaluation cache.
rem Logs: 04_code\results\logs\single\revX2X3_<step>.log / .err
cd /d "%~dp0"
set X2_SMOKE=
set C3_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_x2x3.py x3     >> "%L%\revX2X3_x3.log"     2>> "%L%\revX2X3_x3.err"     || exit /b 1
"%PY%" -u stage_x2x3.py arms   >> "%L%\revX2X3_arms.log"   2>> "%L%\revX2X3_arms.err"   || exit /b 2
"%PY%" -u stage_x2x3.py x2test >> "%L%\revX2X3_x2test.log" 2>> "%L%\revX2X3_x2test.err" || exit /b 3
"%PY%" -u stage_x2x3.py report >> "%L%\revX2X3_report.log" 2>> "%L%\revX2X3_report.err" || exit /b 4
"%PY%" -u stage_x2x3.py family >> "%L%\revX2X3_family.log" 2>> "%L%\revX2X3_family.err" || exit /b 5
echo X2/X3 chain finished >> "%L%\revX2X3_x3.log"
