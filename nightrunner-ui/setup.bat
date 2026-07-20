@echo off
REM Double-click this file to set up and launch night-runner-ui.
REM
REM FIRST TIME ONLY: Windows may show a blue "Windows protected your PC"
REM screen (SmartScreen). If that happens, click "More info", then click
REM "Run anyway". This only happens because the file isn't digitally
REM signed by a registered publisher - it's expected for a small project
REM like this, not a sign anything's wrong.

cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
    python setup.py
    goto :end
)

where py >nul 2>nul
if %errorlevel%==0 (
    py setup.py
    goto :end
)

echo Python doesn't seem to be installed on this PC.
echo.
echo Easiest fix: use NightRunnerSetup.exe instead of this file - it doesn't
echo need Python installed at all. You can find it on the GitHub Releases
echo page for this project.
echo.
echo Or, install Python from https://www.python.org/downloads/ ^(make sure
echo to tick "Add Python to PATH" during install^) and re-run this file.
echo.
pause

:end
