@echo off
TITLE ARIA SOUND FACTORY - HYPERION TEST
cd /d "C:\Users\Roberto\aria\ACE-Step-1.5"

:: Uso diretto del python dell'ambiente per evitare problemi con PATH di conda
set "PYTHON_EXE=C:\Users\Roberto\aria\envs\dias-sound-engine\python.exe"

echo.
echo ============================================================
echo   ARIA SOUND FACTORY - TEST GENERAZIONE LUNGA (HYPERION)
echo ============================================================
echo.
echo Prompt: "A dense, cinematic soundscape for a far-future space opera..."
echo Durata: 180 secondi
echo.

:: Avvio generazione
:: Il comando (echo y & echo.) serve per rispondere automaticamente ai prompt del CLI
(echo y & echo.) | "%PYTHON_EXE%" cli.py --config hyperion_test.toml

echo.
echo ------------------------------------------------------------
echo Generazione terminata.
echo Controlla: C:\Users\Roberto\aria\data\outputs\hyperion_test
echo ------------------------------------------------------------
pause
