@echo off
rem IndoorGS new project (double-click).
rem Asks for a folder name, runs "indoorgs init" which finds COLMAP / Brush / LichtFeld
rem automatically (PATH, C:\tools, and the SEARCH folders below), then opens project.yaml.
setlocal

rem ===== edit these two lines for your PC =====
rem SEARCH: folders to look for tools in (depth 4). Several: "F:\Sample\tools" "D:\tools"
set SEARCH="F:\Sample\tools"
rem PROJECTS: where a plain folder name goes. English path is safer than Korean.
set "PROJECTS=%USERPROFILE%\IndoorGS_projects"
rem ===========================================

for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
set "IG=%REPO%\.venv\Scripts\indoorgs.exe"
if not exist "%IG%" goto :no_install

echo Project name, for example student_hall (goes under %PROJECTS%),
echo or a full path, for example D:\scan\student_hall
set "NAME="
set /p "NAME=Name or path: "
if "%NAME%"=="" goto :no_name
set "NAME=%NAME:"=%"
set "TARGET=%PROJECTS%\%NAME%"
echo(%NAME%| findstr /c:":" >nul && set "TARGET=%NAME%"
echo(%NAME%| findstr /b /c:"\\" >nul && set "TARGET=%NAME%"

if exist "%TARGET%\project.yaml" goto :exists
"%IG%" init "%TARGET%" --search %SEARCH%
if errorlevel 1 goto :fail
:exists

echo.
echo Next: write the videos under "sections:" in project.yaml, save, and close Notepad.
start "" /wait notepad "%TARGET%\project.yaml"
"%IG%" check "%TARGET%" --search %SEARCH%
echo.
choice /m "Open the IndoorGS screen now"
if errorlevel 2 goto :end
call "%~dp0IndoorGS.bat" "%TARGET%"
:end
exit /b 0

:no_install
echo [!] IndoorGS is not installed yet. Run install.bat first.
pause
exit /b 1

:no_name
echo [!] No name given.
pause
exit /b 1

:fail
echo [!] indoorgs init failed. Read the message above.
pause
exit /b 1
