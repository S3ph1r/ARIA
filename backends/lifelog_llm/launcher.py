"""
ARIA — Lifelog LLM Launcher

Thin Python wrapper that execs llama-server.exe (prebuilt CUDA binary).
Called by the orchestrator via _build_cmd; re-invokes the binary with the
same argv so the orchestrator's process tracking works normally.

Must be run from the lifelog-llm conda env (python.exe in envs/lifelog-llm).
The binary is expected at: aria_root/tools/llama/llama-server.exe

2026-09-05: self-reporting del PID reale su file (stesso schema già in
produzione su backends/flux_imagegen/server.py dal 2026-08-15, vedi quel
file per la storia completa dell'indagine originale — qui solo il riassunto
applicato a QUESTO launcher).

PRIMA: nessun self-reporting per qwen3-14b-q4km. _kill_proc() in
aria_node_controller/core/orchestrator.py ricadeva quindi sempre sul
fallback (proc.poll() sul Popen del lanciatore 'start') — su Windows quel
lanciatore termina da solo pochi secondi dopo aver aperto la finestra
reale, quindi il controllo era quasi sempre falso: il comando di kill non
partiva MAI. Confermato dal vivo il 2026-09-05 durante il reprocess storico
di Lifelog2 (Stage D su qwen3-14b-q4km alternato a Stage G/covers su Flux2
sulla stessa GPU): un tentativo di swap "GPU Exclusivity: Termino
qwen3-14b-q4km per far posto a flux2-klein-4b" non è mai stato seguito da
nessuna riga di conferma di terminazione — a differenza della direzione
opposta (Flux2, che il self-reporting ce l'ha), che termina pulita col
proprio PID. Risultato: il processo llama-server.exe (e la finestra) restano
vivi e residenti in VRAM mentre il backend successivo tenta di caricarsi
sulla stessa GPU — esattamente il rischio "due modelli in VRAM insieme"
segnalato da Roberto.

DOPO: questo launcher scrive PID (il proprio, os.getpid() — è lui il
genitore diretto di llama-server.exe via subprocess.run, quindi un
taskkill /T sul suo PID termina anche il figlio) + create_time (protezione
contro il riuso del PID) su file, stessa convenzione già letta da
_read_pid_file() in orchestrator.py: aria_root/logs/pids/{model_id}.pid.
Rimosso nel finally, dopo che llama-server.exe è già terminato per
qualunque motivo (kill esterno, crash, chiusura normale).

Dipendenza psutil SOLO per create_time() — avvolta in un try/except
esplicito: se psutil non è installato in questo env (lifelog-llm è un env
minimale, mai verificato prima d'ora se lo include), il self-reporting
viene saltato con un log chiaro invece di far fallire l'avvio di
llama-server.exe — un self-reporting mancante riporta al comportamento
precedente (bug noto, non un peggioramento), un avvio fallito sarebbe
invece un regresso reale. Se vedi il warning sotto nei log, installalo con
`pip install psutil` nell'env lifelog-llm per attivare il fix.
"""

import json
import logging
import subprocess
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[launcher] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "qwen3-14b-q4km"
PID_DIR = Path(r"C:\Users\Roberto\aria\logs\pids")
PID_FILE = PID_DIR / f"{MODEL_ID}.pid"


def _write_pid_file() -> None:
    try:
        import psutil
    except ImportError:
        logger.warning(
            "psutil non disponibile in questo env — self-reporting PID saltato "
            "(comportamento precedente: kill non affidabile allo swap). "
            "Installa con: pip install psutil"
        )
        return
    try:
        PID_DIR.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        create_time = psutil.Process(pid).create_time()
        PID_FILE.write_text(json.dumps({"pid": pid, "create_time": create_time}))
        logger.info(f"PID file scritto: {PID_FILE} (pid={pid})")
    except Exception:
        logger.exception(f"Errore scrivendo il PID file {PID_FILE}")


def _remove_pid_file() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
        logger.info(f"PID file rimosso: {PID_FILE}")
    except Exception:
        logger.exception(f"Errore rimuovendo il PID file {PID_FILE}")


if __name__ == "__main__":
    aria_root = Path(__file__).parent.parent.parent

    llama_exe = aria_root / "tools" / "llama" / "llama-server.exe"
    if not llama_exe.exists():
        print(f"[launcher] ERROR: llama-server.exe not found at {llama_exe}", flush=True)
        print(f"[launcher] Run scripts/install_lifelog_llm.ps1 to download it.", flush=True)
        sys.exit(1)

    cmd = [str(llama_exe)] + sys.argv[1:]
    print(f"[launcher] Starting: {' '.join(cmd[:4])} ...", flush=True)

    # Scritto PRIMA di avviare llama-server.exe: così l'orchestratore vede
    # questo launcher (e quindi può ucciderlo) anche se l'avvio del binario
    # si blocca o è lento — stesso principio già usato in flux2/server.py.
    _write_pid_file()
    try:
        # cwd = binary directory so Windows finds ggml-cuda.dll and other DLLs
        proc = subprocess.run(cmd, cwd=str(llama_exe.parent))
        sys.exit(proc.returncode)
    finally:
        _remove_pid_file()
