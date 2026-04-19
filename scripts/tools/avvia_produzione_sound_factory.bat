@echo off
TITLE ARIA Sound Factory - Production
cd /d "%~dp0"

:: Configurazione Percorsi
set PYTHON_EXE=C:\Users\Roberto\aria\envs\audiocraft-env\python.exe
set SCRIPT=scripts\production\process_shopping_list.py
set JSON_LIST=data\production\inbound\master_shopping_list_micro.json

echo =======================================================
echo     ARIA SOUND FACTORY - BATCH PRODUCTION (v1.1)
echo =======================================================
echo.
echo Ambiente: %PYTHON_EXE%
echo Shopping List: %JSON_LIST%
echo.
echo Avvio in corso...
echo.

if not exist "%PYTHON_EXE%" (
    echo [ERRORE] Ambiente Python non trovato in %PYTHON_EXE%
    pause
    exit /b
)

"%PYTHON_EXE%" "%SCRIPT%" "%JSON_LIST%"

echo.
echo =======================================================
echo     Produzione terminata o interrotta.
echo =======================================================
pause
