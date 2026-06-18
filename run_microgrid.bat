@echo off
setlocal

set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
set VENV_PYTHON=%~dp0.venv\Scripts\python.exe

if exist "%VENV_PYTHON%" (
  set PYTHON_EXE=%VENV_PYTHON%
)

cd /d "%~dp0src"
for %%S in (1 2 3) do (
  echo.
  echo ========================================
  echo Running seed %%S
  echo ========================================
  "%PYTHON_EXE%" main.py --config=facmac_smac --env-config=microgrid with seed=%%S batch_size_run=1 use_cuda=False use_tensorboard=False t_max=5000
  if errorlevel 1 (
    echo.
    echo Seed %%S failed. Exiting.
    exit /b 1
  )
)
echo.
echo All seeds finished.
pause
