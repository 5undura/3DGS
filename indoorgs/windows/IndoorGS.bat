@echo off
rem IndoorGS screen (double-click, or drag a project folder onto this file).
rem Runs: <repo>\.venv\Scripts\indoorgs.exe gui <project> --open
rem The screen is a local web page (127.0.0.1). Close this window to stop it.
setlocal
for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
set "IG=%REPO%\.venv\Scripts\indoorgs.exe"
if not exist "%IG%" goto :no_install

set "PROJ=%~1"
if not "%PROJ%"=="" goto :have_proj
echo Drag the project folder into this window and press Enter,
echo or type its path. The folder has project.yaml in it.
set /p "PROJ=Project folder: "
if "%PROJ%"=="" goto :no_proj
set "PROJ=%PROJ:"=%"

:have_proj
if not exist "%PROJ%\project.yaml" goto :no_yaml
"%IG%" gui "%PROJ%" --open
pause
exit /b 0

:no_install
echo [!] IndoorGS is not installed yet. Run install.bat first.
echo     Expected: %IG%
pause
exit /b 1

:no_proj
echo [!] No folder given.
pause
exit /b 1

:no_yaml
echo [!] project.yaml not found in: %PROJ%
echo     Make a project with new_project.bat first.
pause
exit /b 1
