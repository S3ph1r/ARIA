@echo off
title Avvio ARIA — "La Stampante di Inferenza"
color 0b

set ARIA_ROOT=C:\Users\Roberto\aria
set MINICONDA_ROOT=C:\Users\Roberto\miniconda3
set PATH=C:\Users\Roberto\aria\envs\sox\sox-14.4.2;%PATH%

echo =======================================================
echo =         ARIA — "La Stampante di Inferenza"          =
echo =  Solo Orchestrator — backend TTS avviati on-demand  =
echo =======================================================
echo.
echo  I backend TTS si avviano AUTOMATICAMENTE quando ARIA
echo  riceve task e si spengono dopo 45 min di inattivita':
echo.
echo   [Fish task]  -> Voice Cloning :8081 + Fish TTS :8080
echo   [Qwen3 task] -> Qwen3 TTS :8083
echo   [ACE-Step]   -> ACE-Step XL :8084
echo.

echo [1/1] Pulizia processi Python precedenti (zombie prevention)...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM python3.exe >nul 2>&1
timeout /t 2 >nul

echo [2/3] Sincronizzazione Warehouse (Allineamento Modelli)...
powershell -ExecutionPolicy Bypass -File "%ARIA_ROOT%\scripts\sync_junctions.ps1"

echo [3/3] Avvio NODE ORCHESTRATOR (Tray Icon + Process Manager)...
start "ARIA ORCHESTRATOR" cmd /k "cd /d %ARIA_ROOT% & echo ===== ARIA NODE ORCHESTRATOR ===== & %MINICONDA_ROOT%\python.exe aria_node_controller\main_tray.py --no-backends"

echo.
echo =======================================================
echo  ARIA attiva e in ascolto su Redis.
echo  Semaforo e parametri disponibili via Tray Icon.
echo =======================================================
timeout /t 5 >nul
