@echo off
rem Revision v1.5 CPU chain (family single_revision_v1_5): X4/X5 run -> re-computation (X1, B10) -> X6 tabular -> report.
rem Logs: 04_code\results\logs\single\revV15_<step>.log / .err
cd /d "%~dp0"
set V15_SMOKE=
set PY=%~dp0..\..\..\.venv\Scripts\python.exe
set L=%~dp0..\..\results\logs\single
"%PY%" -u stage_x456.py run       >> "%L%\revV15_run.log"       2>> "%L%\revV15_run.err"       || exit /b 1
"%PY%" -u stage_x456.py recompute >> "%L%\revV15_recompute.log" 2>> "%L%\revV15_recompute.err" || exit /b 2
"%PY%" -u stage_x456.py x6tab     >> "%L%\revV15_x6tab.log"     2>> "%L%\revV15_x6tab.err"     || exit /b 3
"%PY%" -u stage_x456.py report    >> "%L%\revV15_report.log"    2>> "%L%\revV15_report.err"    || exit /b 4
echo V15 CPU chain finished >> "%L%\revV15_run.log"
