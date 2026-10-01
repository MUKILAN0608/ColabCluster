@echo off
setlocal
pushd "%~dp0colabcluster"
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Create colabcluster\.venv and install dependencies using colabcluster\README.md first.
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn controller.main:app --reload --host 127.0.0.1 %*
set "controllerExitCode=%errorlevel%"
popd
exit /b %controllerExitCode%
