@echo off
REM ============================================================
REM  setup_env.bat
REM  Crea el entorno virtual en la raiz del proyecto,
REM  instala las librerias y registra el kernel de Jupyter
REM ============================================================

REM Ir a la carpeta raiz de MiniProyecto2 (subir un nivel desde Setup)
cd /d "%~dp0\.."

SET ENV_NAME=env_miniproyecto2
SET KERNEL_NAME=Python (MiniProyecto 2 - RNN)
SET REQ_FILE=requirements.txt

IF NOT EXIST "%REQ_FILE%" (
    IF EXIST "Setup\%REQ_FILE%" (
        SET REQ_FILE=Setup\%REQ_FILE%
    )
)

echo.
echo =====================================================
echo  [1/4] Creando entorno virtual en raiz: %ENV_NAME%
echo =====================================================
python -m venv %ENV_NAME%
IF ERRORLEVEL 1 (
    echo ERROR: No se pudo crear el entorno virtual.
    echo Asegurate de tener Python instalado.
    pause & exit /b 1
)

echo.
echo =====================================================
echo  [2/4] Activando el entorno
echo =====================================================
call %ENV_NAME%\Scripts\activate.bat

echo.
echo =====================================================
echo  [3/4] Instalando dependencias desde %REQ_FILE%
echo =====================================================
python -m pip install --upgrade pip
pip install -r %REQ_FILE%
IF ERRORLEVEL 1 (
    echo ERROR: Fallo la instalacion de dependencias.
    pause & exit /b 1
)

echo.
echo =====================================================
echo  [4/4] Registrando el kernel de Jupyter
echo =====================================================
python -m ipykernel install --user --name %ENV_NAME% --display-name "%KERNEL_NAME%"
IF ERRORLEVEL 1 (
    echo ERROR: No se pudo registrar el kernel.
    pause & exit /b 1
)

echo.
echo =====================================================
echo  LISTO! Entorno configurado correctamente.
echo  Kernel registrado como: "%KERNEL_NAME%"
echo  Ubicacion: %CD%\%ENV_NAME%
echo.
echo  Para iniciar Jupyter:
echo    call %ENV_NAME%\Scripts\activate
echo    jupyter notebook
echo =====================================================
pause
