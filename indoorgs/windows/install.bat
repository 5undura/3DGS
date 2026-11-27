@echo off
rem IndoorGS install (double-click).
rem Makes a private Python environment <repo>\.venv next to the indoorgs folder
rem and installs IndoorGS into it: pip install -e indoorgs[all]  (all = gui, space, fill).
rem Nothing is installed system-wide. Python itself must be installed by you.
setlocal
for %%I in ("%~dp0..") do set "PKG=%%~fI"
for %%I in ("%PKG%\..") do set "REPO=%%~fI"
set "VENV=%REPO%\.venv"
set "VPY=%VENV%\Scripts\python.exe"

if not exist "%PKG%\pyproject.toml" goto :no_pkg
if exist "%VPY%" goto :have_venv

rem --- find Python 3.10+ (py launcher first, then python on PATH) ---
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=py -3"
if defined PY goto :make_venv
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
if defined PY goto :make_venv
goto :no_python

:make_venv
echo [1/2] Creating %VENV% with: %PY%
%PY% -m venv "%VENV%"
if errorlevel 1 goto :fail
rem keep the venv out of git / GitHub Desktop
> "%VENV%\.gitignore" echo *
goto :install

:have_venv
echo [1/2] %VENV% already exists - skipped.

:install
echo [2/2] Installing IndoorGS into the venv. First time takes a few minutes...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -e "%PKG%[all]"
if errorlevel 1 goto :fail
echo.
echo Done. Next:
echo   1. new_project.bat  - make a project folder
echo   2. IndoorGS.bat     - open the screen; or drag a project folder onto it
echo Guide: indoorgs\docs\USAGE.md
pause
exit /b 0

:no_pkg
echo [!] pyproject.toml not found in %PKG%
echo     Keep this file in the indoorgs\windows folder of the repository.
pause
exit /b 1

:no_python
echo [!] Python 3.10 or newer was not found.
echo     Install it yourself from https://www.python.org/downloads/windows/
echo     and tick "Add python.exe to PATH" in the installer. Then run install.bat again.
echo     If "python" opens the Microsoft Store, turn off the App execution alias for python.
pause
exit /b 1

:fail
echo.
echo [!] Install failed. Read the messages above.
echo     To start over, delete the folder %VENV% and run install.bat again.
pause
exit /b 1
