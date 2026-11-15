@echo off
rem Revision X1 chain (family single_revision_v1_3): run -> hazard -> report, each step only if the previous succeeded.
rem CPU only (numpy). Logs: 04_code\results\logs\single\revX1_<step>.log / .err
cd /d "%~dp0"
set X1_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_x1.py run    >> "%L%\revX1_run.log"    2>> "%L%\revX1_run.err"    || exit /b 1
"%PY%" -u stage_x1.py hazard >> "%L%\revX1_hazard.log" 2>> "%L%\revX1_hazard.err" || exit /b 2
"%PY%" -u stage_x1.py report >> "%L%\revX1_report.log" 2>> "%L%\revX1_report.err" || exit /b 3
echo X1 chain finished >> "%L%\revX1_run.log"
