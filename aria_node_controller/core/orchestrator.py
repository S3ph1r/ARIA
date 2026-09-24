import os
import sys
import threading
import time
import requests
import json
import base64
import socket
import http.server
import socketserver
import subprocess
import psutil


from pathlib import Path
from .queue_manager import AriaQueueManager
from .cloud_manager import CloudManager
from .rate_limiter import GeminiRateLimiter
from .batch_optimizer import BatchOptimizer
from .logger import get_logger
import re
from .models import AriaTaskResult
from .registry_manager import AriaRegistryManager
from .telemetry import TelemetryDB
from . import config_manager

class AriaAssetHandler(http.server.SimpleHTTPRequestHandler):
    """
    Handler personalizzato per ARIA.
    Gestisce il routing logico dei file statici su porta 8082.
    """
    def translate_path(self, path):
        # Rimuove il primo '/'
        clean_path = path[1:]

        # 1. Routing verso gli Assets (Anteprime Vocali)
        if clean_path.startswith("assets/"):
            return str(ARIA_ROOT / "data" / clean_path)

        # 2. Routing verso gli Outputs (File generati)
        if clean_path.startswith("outputs/"):
            return str(ARIA_ROOT / "data" / clean_path)

        # 3. COMPATIBILITÀ LEGACY: Se non c'è prefisso, assumiamo sia un output
        # Questo serve per DIAS Stage D (versioni precedenti a v6.5)
        return str(ARIA_ROOT / "data" / "outputs" / clean_path)

    def do_DELETE(self):
        """Gestisce DELETE /filename — elimina un output WAV dopo download confermato da Stage D."""
        import logging
        _logger = logging.getLogger("aria.asset_server")
        file_path = self.translate_path(self.path)
        try:
            os.remove(file_path)
            _logger.info(f"Asset eliminato su richiesta Stage D: {file_path}")
            self.send_response(200)
            self.end_headers()
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
        except Exception as e:
            _logger.warning(f"DELETE fallito per {file_path}: {e}")
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Sopprime log HTTP per ogni richiesta (troppo verboso)

# Backends
try:
    from backends.qwen3_tts import Qwen3TTSBackend
    from backends.qwen35_llm import Qwen35LLMBackend
    from backends.lifelog_llm import LifelogLLMBackend
    from backends.acestep import ACEStepBackend
    from backends.audiocraft import AudiocraftBackend
    from backends.lifelog_asr import LifelogASRBackend
    from backends.lifelog_whisperx import LifelogWhisperXBackend
    from backends.flux_imagegen import FluxImageGenBackend
    _BACKENDS_AVAILABLE = True
    HAS_ACESTEP = True
    HAS_AUDIOCRAFT = True
except ImportError:
    try:
        from ..backends.qwen3_tts import Qwen3TTSBackend
        from ..backends.qwen35_llm import Qwen35LLMBackend
        from ..backends.lifelog_llm import LifelogLLMBackend
        from ..backends.acestep import ACEStepBackend
        from ..backends.audiocraft import AudiocraftBackend
        from ..backends.lifelog_asr import LifelogASRBackend
        from ..backends.lifelog_whisperx import LifelogWhisperXBackend
        from ..backends.flux_imagegen import FluxImageGenBackend
        _BACKENDS_AVAILABLE = True
        HAS_ACESTEP = True
        HAS_AUDIOCRAFT = True
    except ImportError:
        _BACKENDS_AVAILABLE = False
        HAS_ACESTEP = False
        HAS_AUDIOCRAFT = False
        LifelogLLMBackend = None
        FluxImageGenBackend = None


logger = get_logger("node.orchestrator")

FISH_TTS_HOST    = "http://localhost:8080"
FISH_ENCODE_HOST = "http://localhost:8081"
QWEN3_TTS_HOST   = "http://localhost:8083"

# Directory e Path — auto-detect basato sul sistema operativo
if os.name == "nt":
    ARIA_ROOT      = Path(os.path.expanduser("~")) / "aria"
    MINICONDA_ROOT = Path(os.path.expanduser("~")) / "miniconda3"
else:
    # Linux LXC (Based on project structure)
    ARIA_ROOT      = Path("/home/Projects/NH-Mini/sviluppi/ARIA")
    MINICONDA_ROOT = Path("/home/roberto/miniconda3") # Standard paths

ARIA_OUTPUT_DIR = ARIA_ROOT / "data" / "outputs"
HTTP_PORT       = 8082

# Secondi di coda vuota prima di terminare un backend
IDLE_TIMEOUT_S = 2700



class ModelProcessManager:
    """
    Gestisce il ciclo di vita dei processi backend TTS (Fish, Qwen3).

    - Avvia il server on-demand quando arrivano task in coda.
    - Lo termina dopo IDLE_TIMEOUT_S secondi di coda vuota.
    - Non usa 'conda activate': punta direttamente al Python dell'env conda.

    Configurazione per modello:
      start_cmd   : lista di token per subprocess.Popen
      health_url  : URL health check
      startup_wait: secondi max per il primo health check (JIT 1° avvio)
    """

    MODEL_CONFIGS = {
        "fish-s1-mini": {
            "port":         8080,
            "health_url":   "http://localhost:8080/v1/health",
            "startup_wait": 90,
            "companion":    "voice-cloning",
        },
        "voice-cloning": {
            "port":         8081,
            "health_url":   "http://localhost:8081/health",
            "startup_wait": 60,
        },
        "qwen3-tts-1.7b": {
            "port":         8083,
            "health_url":   "http://localhost:8083/health",
            "startup_wait": 240,
        },
        "qwen3-tts-custom": {
            "port":         8083,  # Stessa porta di Qwen3 Base (Mutuamente esclusivi)
            "health_url":   "http://localhost:8083/health",
            "startup_wait": 240,
        },
        "qwen3.5-35b-moe-q3ks": {
            "port":         8085,
            "health_url":   "http://localhost:8085/v1/health", # Fallback check
            "startup_wait": 300,
        },
    }


    def __init__(self, aria_root: Path, miniconda_root: Path):
        self.aria_root      = aria_root
        self.miniconda_root = miniconda_root
        self._procs: dict[str, subprocess.Popen] = {}
        self._idle_since: dict[str, float]        = {}
        self._starting: set[str]                  = set()
        self._lock = threading.RLock()  # RLock: _ensure_single holds it while calling _kill_proc

        # Caricamento Manifest dei Backend
        self.MODEL_CONFIGS = self._load_manifest()
        self.local_ip = get_node_ip()

    def _load_manifest(self) -> dict:
        manifest_path = self.aria_root / "aria_node_controller" / "config" / "backends_manifest.json"
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
                logger.info(f"Manifest backend caricato correttamente da {manifest_path}")
                return data.get("backends", {})
        except Exception as e:
            logger.error(f"Errore caricamento manifest {manifest_path}: {e}")
            # Fallback a un dizionario vuoto per evitare crash
            return {}

    def _build_cmd(self, model_id: str) -> list:
        """Costruisce il comando di avvio per il modello dato basandosi sul Manifest.

        Ricarica il manifest da disco ad ogni chiamata: i backend si avviano di rado
        (cold start ~ogni 45min al massimo), e così modificare `server_args`/`args`/
        `llm_contract` NON richiede il riavvio dell'orchestratore — basta il deploy del file.
        """
        self.MODEL_CONFIGS = self._load_manifest()
        cfg = self.MODEL_CONFIGS.get(model_id)
        if not cfg:
            raise ValueError(f"Nessuna configurazione trovata nel manifest per model_id='{model_id}'")

        # 1. Risoluzione Python Executable
        env_prefix = cfg.get("env_prefix")
        if not env_prefix:
            # Se non c'è env_prefix, usa il python della base (poco probabile per AI backends)
            python = "python"
        else:
            python_path = self.aria_root / env_prefix / "python.exe"
            if not python_path.exists() and os.name != 'nt':
                 # Su Linux il binario è 'python' o 'python3' dentro bin/
                 python_path = self.aria_root / env_prefix / "bin" / "python"
            python = str(python_path)

        # 2. Risoluzione Script principale
        script_rel = cfg.get("script")
        if not script_rel:
            raise ValueError(f"Script non definito per backend '{model_id}' nel manifest.")
        script_abs = str(self.aria_root / script_rel)

        # 3. Costruzione comando base
        cmd = [python, script_abs]

        def _resolve(val: str) -> str:
            # Se il valore sembra un path relativo ad ARIA, lo risolviamo su aria_root
            if isinstance(val, str) and val.startswith(("data/", "aria_node_controller/", "envs/")):
                return str(self.aria_root / val)
            return str(val)

        # 4a. Argomenti — lista piatta (retrocompat: fish/acestep/audiocraft/...)
        for arg in cfg.get("args", []):
            cmd.append(_resolve(arg))

        # 4b. server_args — dict {flag: valore} (nuovo, usato da qwen3-14b-q4km).
        #     bool True  -> flag nudo (--jinja)
        #     bool False / None -> omesso
        #     altro      -> --flag valore  (path risolti su aria_root)
        for key, val in cfg.get("server_args", {}).items():
            if val is False or val is None:
                continue
            cmd.append(f"--{key}")
            if val is True:
                continue
            cmd.append(_resolve(val))

        return cmd

    def _health_check(self, model_id: str) -> bool:
        url = self.MODEL_CONFIGS[model_id]["health_url"].replace("localhost", self.local_ip)
        try:
            r = requests.get(url, timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def ensure_running(self, model_id: str) -> bool:
        """
        Garantisce che il processo del modello sia attivo.
        Se il modello ha un 'companion', lo avvia prima.
        Ritorna True se pronto, False se fuori timeout.
        """
        # Avvia il companion prima (es. voice-cloning per fish-s1-mini)
        companion = self.MODEL_CONFIGS.get(model_id, {}).get("companion")
        if companion:
            if not self._ensure_single(companion):
                logger.error(f"Companion '{companion}' non pronto per {model_id}")
                return False
            time.sleep(2)  # piccolo delay race condition companion → principale

        return self._ensure_single(model_id)

    def _ensure_single(self, model_id: str) -> bool:
        """Avvia un singolo processo e attende il health check."""

        with self._lock:
            # 0. Gestione Esclusività GPU (Solo un backend locale alla volta)
            # Killiamo proattivamente ogni altro backend tracciato
            for other_id in list(self._procs.keys()):
                if other_id != model_id:
                    logger.info(f"GPU Exclusivity: Termino {other_id} per far posto a {model_id}...")
                    self._kill_proc(other_id)

            # Nuclear Option (Windows): Kill per titolo per ogni altro modello nel manifest (zombie detection)
            if os.name == 'nt':
                for mid in self.MODEL_CONFIGS.keys():
                    if mid != model_id:
                        title = f"ARIA Backend: {mid}"
                        try:
                            subprocess.run(f'taskkill /F /FI "WINDOWTITLE eq {title}*"', shell=True, capture_output=True, timeout=5)
                        except subprocess.TimeoutExpired:
                            pass

            # 1. Controllo proattivo...
            if self._health_check(model_id):
                logger.info(f"{model_id}: backend già attivo e responsivo (rilevato esternamente).")
                self._idle_since.pop(model_id, None)
                # 2026-09-05: stesso motivo del blocco gemello dopo l'health
                # check dell'avvio fresco sotto — questo è il percorso PIÙ
                # comune in steady-state (il backend risulta già attivo alla
                # quasi totalità dei controlli), quindi il posto più
                # frequente in cui scoprire un PID mai catturato finora (es.
                # un backend partito prima che questo fix esistesse, o senza
                # self-reporting).
                # 2026-09-24: passa da _discover_pid_by_window_title() diretto
                # a _get_tracked_pid(), che prova PRIMA self._procs[model_id]
                # (il Popen che questo stesso orchestratore ha spawnato — ora
                # sempre corretto, vedi _ensure_single sotto) — la scoperta
                # via titolo resta solo per un backend rimasto vivo da un
                # riavvio precedente dell'orchestratore, mai tracciato in
                # questa sessione.
                if os.name == 'nt' and self._read_pid_file(model_id) is None:
                    tracked_pid = self._get_tracked_pid(model_id)
                    if tracked_pid is not None:
                        self._write_pid_file_for(model_id, tracked_pid)
                return True

            proc = self._procs.get(model_id)

            # 2. Controllo Warm-up (Memory di ARIA)
            if model_id in self._starting:
                logger.info(f"{model_id}: backend già in fase di warm-up, attendo il pronto...")
                return True # L'attesa reale avverrà nel loop esterno di ensure_running

            # 3. Gestione processo esistente ma non responsivo
            # 2026-08-15: controlla PRIMA il PID reale auto-riportato — senza
            # questo, un backend che si auto-riporta (es. flux2-klein-4b) ma
            # è semplicemente lento a rispondere finiva SEMPRE nel ramo
            # "terminato inaspettatamente" (perché il Popen del lanciatore
            # 'start' è morto da tempo, vedi _kill_proc), che rilanciava un
            # processo NUOVO sopra quello vecchio ancora vivo invece di
            # ucciderlo prima — causa della "pila" di finestre/processi
            # zombie diagnosticata oggi.
            real_pid = self._read_pid_file(model_id)
            if real_pid is not None:
                logger.warning(f"{model_id}: processo reale (pid={real_pid}) attivo ma non risponde. Lo termino per riavvio pulito...")
                self._kill_proc(model_id)
            elif proc and proc.poll() is not None:
                logger.warning(f"{model_id}: processo terminato inaspettatamente, riavvio...")
            elif proc and proc.poll() is None:
                # Se arriviamo qui, il processo esiste ma NON è responsive.
                # Killiamo solo se NON è in fase di avvio (già gestito sopra)
                logger.warning(f"{model_id}: processo attivo ma non risponde. Lo termino per riavvio pulito...")
                self._kill_proc(model_id)
            
            # 4. Registrazione avvio
            self._starting.add(model_id)

            # Avvio
            try:
                cmd = self._build_cmd(model_id)
                logger.info(f"Avvio backend {model_id} in finestra Console dedicata...")

                # Risoluzione working_dir dal manifest (es. 'backends/acestep')
                # Se presente, viene usato come CWD del processo e aggiunto a PYTHONPATH
                # così il package locale (es. 'acestep') è importabile senza installazione.
                cfg = self.MODEL_CONFIGS.get(model_id, {})
                working_dir_rel = cfg.get("working_dir")
                if working_dir_rel:
                    process_cwd = str(self.aria_root / working_dir_rel)
                else:
                    process_cwd = str(self.aria_root)

                if os.name == 'nt':
                    # Magic CMD escape bug: Se usiamo 'start' con 'cmd /k', dobbiamo stare attenti a come passiamo la stringa.
                    # Il modo più sicuro è NON quotare l'intera cmd_str se i singoli pezzi sono già quotati,
                    # oppure usare un trucco specifico di CMD se ci sono spazi.
                    cmd_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
                    title = f"ARIA Backend: {model_id}"

                    # Iniezione PATH per SoX e altre dipendenze core
                    env = os.environ.copy()
                    sox_path = str(self.aria_root / "envs" / "sox" / "Library" / "bin")
                    if os.path.exists(sox_path):
                        env["PATH"] = sox_path + os.pathsep + env.get("PATH", "")
                        logger.info(f"Injected SoX path: {sox_path}")

                    # Iniezione PYTHONPATH con working_dir per package locali
                    if working_dir_rel:
                        env["PYTHONPATH"] = process_cwd + os.pathsep + env.get("PYTHONPATH", "")
                        logger.info(f"Working dir → {process_cwd} (aggiunto a PYTHONPATH)")

                    # Iniezione variabili d'ambiente custom dal manifest
                    custom_env = cfg.get("env", {})
                    if custom_env:
                        for k, v in custom_env.items():
                            env[k] = str(v)
                            logger.info(f"Custom Env: {k}={v}")

                    # Blocca download HuggingFace: i modelli sono già disponibili via junction NTFS
                    env["HF_HUB_OFFLINE"] = "1"
                    env["TRANSFORMERS_OFFLINE"] = "1"

                    # Iniezione Credenziali MinIO da Config Manager
                    env["ARIA_MINIO_ENDPOINT"] = config_manager.MINIO_ENDPOINT
                    env["ARIA_MINIO_ACCESS_KEY"] = config_manager.MINIO_ACCESS_KEY
                    env["ARIA_MINIO_SECRET_KEY"] = config_manager.MINIO_SECRET_KEY
                    logger.info(f"Injected MinIO credentials for {model_id} (Endpoint: {config_manager.MINIO_ENDPOINT})")

                    # 2026-09-24 (Gap A1-5, vedi docs/aria-state-of-gaps.md):
                    # rimosso 'start "title" cmd.exe /k ...' con shell=True.
                    # Con quel wrapper, new_proc.pid era il processo 'start'
                    # (muore da solo in pochi secondi) — mai il backend vero.
                    # Per questo servivano self-reporting via pid-file e
                    # scoperta via titolo finestra dopo il fatto. Ma Windows
                    # 11 delega l'allocazione di OGNI nuova console a Windows
                    # Terminal (WindowsTerminal.exe + OpenConsole.exe, un
                    # processo separato che possiede la finestra visibile) —
                    # e un match per titolo può quindi prendere il PID del
                    # CONTENITORE invece del backend. Confermato dal vivo:
                    # PID salvato per qwen3-14b-q4km era WindowsTerminal.exe,
                    # completamente slegato dal vero albero
                    # cmd.exe→python.exe→llama-server.exe. Un taskkill /T su
                    # quel PID rischia di abbattere l'intera app terminale
                    # con qualunque altra finestra/scheda al suo interno.
                    #
                    # Fix: spawniamo 'cmd.exe /c' DIRETTAMENTE (niente
                    # 'start', niente shell=True) con CREATE_NEW_CONSOLE.
                    # new_proc.pid è così SEMPRE il PID reale di questo
                    # cmd.exe, restituito da Windows al momento stesso dello
                    # spawn — nessuna scoperta, nessuna ambiguità, immune a
                    # quale terminale Windows scelga per renderizzare la
                    # finestra (self._procs[model_id] lo traccia già sotto).
                    # '/c' invece di '/k': la finestra si chiude da sola a
                    # fine processo — niente prompt vuoto da richiudere
                    # separatamente dopo un kill. Il comando 'title' imposta
                    # comunque il titolo visibile, invariato per chi legge
                    # il desktop o per _position_window()/EnumWindows sotto.
                    #
                    # 2026-09-24, aggiornamento stesso giorno (Roberto — bug
                    # riprodotto dal vivo subito dopo il fix sopra): il PID
                    # tracciato era corretto (verificato: `self._procs[model_id]
                    # .pid` = il vero cmd.exe), il filtro anti-contenitore in
                    # _discover_pid_by_window_title ha correttamente scartato
                    # WindowsTerminal.exe, e NESSUN taskkill di ARIA ha preso
                    # di mira quel PID in tutta la sessione osservata — eppure
                    # una finestra PowerShell indipendente (Sniper) si è
                    # comunque chiusa nello stesso istante dello spawn.
                    # CREATE_NEW_CONSOLE non basta a evitare la delega: Windows
                    # 11 la applica comunque a QUALUNQUE nuova console allocata
                    # dal sistema, indipendentemente da come la si richiede —
                    # e un WindowsTerminal.exe che consolida/rimpiazza finestre
                    # proprie in modi non prevedibili dall'esterno resta un
                    # rischio strutturale finché la delega è in gioco, anche
                    # con un tracking del PID ormai corretto.
                    # Fix definitivo: invocare esplicitamente 'conhost.exe'
                    # come eseguibile — richiesta esplicita del console host
                    # classico, che Windows onora senza ridelegare a Windows
                    # Terminal (verificato dal vivo su PC 139: spawn di prova
                    # via conhost.exe → zero nuove istanze WindowsTerminal.exe/
                    # OpenConsole.exe comparse, albero pulito
                    # conhost.exe→cmd.exe). La finestra resta visibile con i
                    # log in tempo reale (richiesto esplicitamente da
                    # Roberto) — cambia solo l'aspetto (console classica
                    # invece di una scheda Windows Terminal), non la funzione.
                    new_proc = subprocess.Popen(
                        ["conhost.exe", "cmd.exe", "/c", f'title {title} && {cmd_str}'],
                        cwd=process_cwd,
                        env=env,
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                    )
                else:
                    # Fallback standard per Linux/Mac (Mantiene i log su file per non sporcare stdout)
                    log_dir = self.aria_root / "logs"
                    log_dir.mkdir(exist_ok=True)
                    log_out = open(log_dir / f"startup_{model_id.replace('-','_')}.log", "a")
                    log_err = open(log_dir / f"startup_{model_id.replace('-','_')}_err.log", "a")

                    env = os.environ.copy()
                    if working_dir_rel:
                        env["PYTHONPATH"] = process_cwd + os.pathsep + env.get("PYTHONPATH", "")

                    new_proc = subprocess.Popen(
                        cmd,
                        stdout=log_out,
                        stderr=log_err,
                        cwd=process_cwd,
                        env=env,
                    )

                self._procs[model_id] = new_proc
                self._idle_since.pop(model_id, None)
            except Exception as e:
                logger.error(f"Impossibile avviare {model_id}: {e}")
                return False

        # Wait for health check (fuori dal lock per non bloccare altri modelli o la dashboard)
        max_wait = self.MODEL_CONFIGS[model_id]["startup_wait"]
        logger.info(f"Attesa health check {model_id} (max {max_wait}s)...")
        
        success = False
        try:
            for _ in range(max_wait):
                if self._health_check(model_id):
                    logger.info(f"{model_id}: health check OK")
                    success = True
                    break
                time.sleep(1)
        finally:
            with self._lock:
                self._starting.discard(model_id)

        if not success:
            logger.error(f"{model_id}: timeout health check ({max_wait}s)")
            return False

        # 2026-09-05: scoperta universale del PID (vedi _get_tracked_pid) —
        # copre ogni backend, non solo quelli con self-reporting esplicito.
        # A questo punto il backend ha già passato l'health check quindi la
        # finestra esiste di sicuro — nessun retry necessario.
        # 2026-09-24 (Gap A1-5): passato da _discover_pid_by_window_title()
        # diretto a _get_tracked_pid() — qui `self._procs[model_id]` è
        # appena stato popolato dallo spawn fresco poche righe sopra, quindi
        # questo è ORA il percorso più comune per ottenere il PID (secondo
        # livello dell'helper), non più la scoperta via titolo (terzo
        # livello, usata solo se anche self._procs mancasse per qualche
        # motivo).
        if os.name == 'nt' and self._read_pid_file(model_id) is None:
            tracked_pid = self._get_tracked_pid(model_id)
            if tracked_pid is not None:
                self._write_pid_file_for(model_id, tracked_pid)
            else:
                logger.warning(
                    f"{model_id}: nessun PID tracciabile dopo l'health "
                    f"check — _kill_proc ricadrà sul fallback (proc.poll())."
                )
        # Posizione fissa (2026-09-05, Roberto): solo qui, al primo avvio
        # fresco — mai sul percorso "backend già attivo" sopra, che gira ad
        # ogni health check e sposterebbe la finestra in continuazione.
        self._position_window(model_id)
        return success

    def mark_idle(self, model_id: str):
        """Segnala che la coda di questo modello era vuota in questo ciclo."""
        if model_id not in self._idle_since:
            self._idle_since[model_id] = time.time()

    def shutdown_idle_backends(self):
        """Termina i backend (e i loro companion) che sono idle da > IDLE_TIMEOUT_S."""
        now = time.time()
        for model_id, idle_since in list(self._idle_since.items()):
            if now - idle_since >= IDLE_TIMEOUT_S:
                # Termina prima il principale, poi il companion (es. fish → voice-cloning)
                self._kill_proc(model_id)
                companion = self.MODEL_CONFIGS.get(model_id, {}).get("companion")
                if companion:
                    self._kill_proc(companion)
                self._idle_since.pop(model_id, None)

    def _read_pid_file(self, model_id: str) -> "int | None":
        """Legge il PID reale auto-riportato dal backend, se implementa il
        self-reporting (2026-08-15, oggi solo flux2-klein-4b — vedi
        _write_pid_file in backends/flux_imagegen/server.py).

        Verifica create_time contro il processo vivo per proteggersi da un
        PID riusato da un processo diverso nel frattempo (il file può
        restare stale dopo un kill secco che salta la pulizia Python).
        Ritorna None se il file manca, è corrotto, o il PID non corrisponde
        più a un processo vivo con lo stesso create_time registrato — in
        quel caso il chiamante deve ricadere sul comportamento precedente
        (Popen/titolo finestra).
        """
        pid_path = self.aria_root / "logs" / "pids" / f"{model_id}.pid"
        try:
            data = json.loads(pid_path.read_text())
            pid, create_time = int(data["pid"]), float(data["create_time"])
            proc = psutil.Process(pid)
            if abs(proc.create_time() - create_time) < 1.0:
                return pid
        except Exception:
            pass
        return None

    def _forget_pid_file(self, model_id: str) -> None:
        pid_path = self.aria_root / "logs" / "pids" / f"{model_id}.pid"
        try:
            pid_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _discover_pid_by_window_title(self, model_id: str) -> "int | None":
        """Scoperta universale del PID via titolo finestra (2026-09-05,
        Roberto: "aria non può salvare il titolo finestra e il PID in un
        file quando starta i backend?") — alternativa al self-reporting
        che serve SOLO per i backend toccati a mano uno per uno
        (flux2-klein-4b, qwen3-14b-q4km oggi). Questa funzione copre
        QUALUNQUE backend, incluso whisperx/fish-speech/qwen3-tts/acestep/
        qwen3-asr/qwen3.5-moe, senza bisogno di modificarne il codice: gira
        lato orchestratore, interroga tasklist per lo stesso titolo già
        usato ovunque nel file ("ARIA Backend: {model_id}").

        Nota architetturale importante: il PID trovato così è quello di
        cmd.exe (il processo che POSSIEDE la finestra — è lui ad averla
        aperta con 'start "titolo" cmd.exe /k ...'), non quello del
        backend vero (che gira come SUO figlio). Questo è un vantaggio, non
        un limite: un taskkill /T su questo PID termina insieme sia il
        backend sia la finestra che lo contiene, in un colpo solo — niente
        bisogno del secondo taskkill per titolo che _kill_proc fa oggi
        subito dopo. Il self-reporting resta preferito quando esiste
        (_read_pid_file guarda prima quello): cattura il processo del
        backend stesso un istante dopo la sua creazione, questa invece
        dipende da un tasklist riuscito e da un titolo finestra univoco.

        Ritorna None su qualunque errore o se nessuna finestra corrisponde
        (mai un'eccezione al chiamante — chi chiama deve poter continuare
        comunque col comportamento attuale se la scoperta fallisce)."""
        if os.name != 'nt':
            return None
        title = f"ARIA Backend: {model_id}"
        try:
            r = subprocess.run(
                f'tasklist /FI "WINDOWTITLE eq {title}*" /FO CSV /NH',
                shell=True, capture_output=True, timeout=5, text=True,
            )
            out = (r.stdout or "").strip()
            if r.returncode != 0 or not out or out.upper().startswith("INFO:"):
                # "INFO: No tasks..." è l'output normale di tasklist quando
                # nessuna finestra corrisponde — non un errore da loggare.
                return None
            # Formato CSV: "Image Name","PID","Session Name","Session#","Mem Usage"
            first_row = out.splitlines()[0]
            fields = [f.strip('"') for f in first_row.split('","')]
            image_name, pid = fields[0], int(fields[1])

            # 2026-09-24 (Gap A1-5): Windows 11 delega l'allocazione di ogni
            # nuova console a Windows Terminal — un match per titolo può
            # restituire il PID del CONTENITORE (WindowsTerminal.exe /
            # OpenConsole.exe) invece del processo applicativo, se la
            # finestra è ospitata lì. Un taskkill /T su quel PID abbatte
            # l'intera app terminale con qualunque altra finestra/scheda al
            # suo interno. Confermato dal vivo: PID scoperto per
            # qwen3-14b-q4km era WindowsTerminal.exe, non il backend —
            # scoperto perché ha chiuso una finestra PowerShell indipendente
            # (Sniper watchdog) al primo swap successivo. Rifiutiamo
            # esplicitamente questi image name generici prima di fidarci del
            # PID: meglio nessun PID (si ricade sul fallback superiore) che
            # un PID troppo largo.
            #
            # 2026-09-24, aggiornamento: 'conhost.exe' rimosso da questo
            # elenco. Da quando lo spawn invoca esplicitamente conhost.exe
            # (vedi _ensure_single) invece di lasciare che Windows deleghi a
            # Windows Terminal, un conhost.exe scoperto qui È il processo
            # legittimo che abbiamo spawnato noi — un conhost.exe è sempre
            # dedicato a UNA console, mai condiviso tra finestre/app diverse
            # come può esserlo WindowsTerminal.exe. Solo quest'ultimo (e il
            # suo OpenConsole.exe) restano nella lista, per il caso residuo
            # di un backend avviato prima di questo fix o con la vecchia
            # delega ancora attiva.
            if image_name.lower() in {"windowsterminal.exe", "openconsole.exe"}:
                logger.warning(
                    f"{model_id}: titolo finestra trovato ma il processo è "
                    f"{image_name} (contenitore terminale, non il backend) — "
                    f"scarto, nessun PID salvato."
                )
                return None
            return pid
        except Exception:
            logger.exception(f"{model_id}: errore scoprendo il PID via titolo finestra")
            return None

    def _get_tracked_pid(self, model_id: str) -> "int | None":
        """PID reale più affidabile noto per questo modello, in ordine di
        fiducia (2026-09-24, Gap A1-5):

        1. Self-reporting su file (`_read_pid_file`), quando il backend lo
           implementa — cattura il PID un istante dopo la sua creazione.
        2. Il Popen tracciato da QUESTO orchestratore (`self._procs`) — dal
           fix del 2026-09-24 (spawn diretto, niente più wrapper 'start')
           `proc.pid` è sempre il PID reale del backend, non più quello di
           uno shell lanciatore morto da tempo. Nessuna chiamata esterna
           necessaria, nessuna ambiguità col terminale usato per la finestra.
        3. Scoperta via titolo finestra — SOLO se nessuno dei due precedenti
           è disponibile: caso residuo di un backend rimasto vivo da un
           riavvio precedente dell'orchestratore (mai spawnato in questa
           sessione, quindi assente da `self._procs`). Filtrata contro i
           processi-contenitore in `_discover_pid_by_window_title`.
        """
        pid = self._read_pid_file(model_id)
        if pid is not None:
            return pid
        proc = self._procs.get(model_id)
        if proc is not None and proc.poll() is None:
            return proc.pid
        if os.name == 'nt':
            return self._discover_pid_by_window_title(model_id)
        return None

    def _write_pid_file_for(self, model_id: str, pid: int) -> None:
        """Come write_pid_file() nei singoli backend (vedi flux_imagegen/
        server.py, lifelog_llm/launcher.py), ma chiamata lato orchestratore
        per un PID scoperto via _discover_pid_by_window_title invece che
        auto-riportato dal backend stesso. Usa lo stesso psutil già
        importato qui (mai il problema "psutil disponibile nell'env del
        backend?" che i singoli launcher devono gestire con un try/except —
        l'orchestratore ha sempre psutil, è già una sua dipendenza)."""
        pid_path = self.aria_root / "logs" / "pids" / f"{model_id}.pid"
        try:
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            create_time = psutil.Process(pid).create_time()
            pid_path.write_text(json.dumps({"pid": pid, "create_time": create_time}))
            logger.info(f"{model_id}: PID scoperto via titolo finestra e salvato ({pid}).")
        except Exception:
            logger.exception(f"{model_id}: errore salvando il PID scoperto {pid}")

    def _position_window(self, model_id: str, x: int = 0, y: int = 0) -> None:
        """Sposta la finestra Console del backend a una posizione fissa sul
        desktop (2026-09-05, Roberto: le finestre si aprivano sparse per lo
        schermo — Windows/il console host non hanno un modo nativo di
        fissare la posizione di avvio via riga di comando, l'unico modo
        affidabile è muoverla via API Win32 SUBITO dopo che esiste).

        Chiamata SOLO al primo avvio fresco (mai sul percorso "backend già
        attivo", che gira praticamente ad ogni health check — riposizionare
        lì sposterebbe la finestra in continuazione anche se l'utente
        l'avesse spostata a mano nel frattempo, un comportamento fastidioso
        e non richiesto).

        EnumWindows/GetWindowText sono limitati alla STESSA sessione
        Windows del chiamante — funziona perché questo codice gira
        nell'orchestratore, che è nella stessa sessione desktop in cui le
        finestre vengono aperte (stesso motivo per cui taskkill per titolo
        funziona già altrove in questo file). Verificato che EnumWindows da
        una sessione DIVERSA (es. una sessione SSH separata) non vede
        queste finestre — se mai questa funzione venisse richiamata da un
        contesto fuori sessione, fallirebbe silenziosamente allo stesso modo
        di _discover_pid_by_window_title quando non trova nulla.

        Nessuna nuova dipendenza: solo ctypes (stdlib), già disponibile
        ovunque gira Python su Windows."""
        if os.name != 'nt':
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            title_prefix = f"ARIA Backend: {model_id}"
            found: list[int] = []

            @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            def _enum_proc(hwnd, _lparam):
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if buf.value.startswith(title_prefix):
                        found.append(hwnd)
                return True

            user32.EnumWindows(_enum_proc, 0)
            if not found:
                logger.debug(f"{model_id}: nessuna finestra trovata da riposizionare (titolo {title_prefix!r}).")
                return
            SWP_NOSIZE, SWP_NOZORDER = 0x0001, 0x0004
            ok = user32.SetWindowPos(found[0], 0, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER)
            if ok:
                logger.info(f"{model_id}: finestra riposizionata a ({x}, {y}).")
            else:
                logger.warning(f"{model_id}: SetWindowPos fallito per la finestra trovata.")
        except Exception:
            logger.exception(f"{model_id}: errore riposizionando la finestra")

    def _kill_proc(self, model_id: str):
        """Termina il processo di un singolo modello se attivo.

        2026-08-15: preferisce il PID reale auto-riportato (via pid-file,
        vedi _read_pid_file) quando disponibile — chirurgico, uccide
        esattamente quel processo per PID esatto invece di affidarsi al
        match per titolo finestra con wildcard '*' (rischio di colpire
        finestre con titolo simile) o al Popen del lanciatore 'start', che
        su Windows termina da solo entro pochi secondi dall'aver aperto la
        finestra reale — rendendo proc.poll() inaffidabile per sapere se
        QUESTO backend è ancora vivo. Bug diagnosticato quel giorno: la
        guardia 'proc and proc.poll() is None' sotto era quasi sempre falsa
        per un backend idle da tempo (il lanciatore era già morto da minuti),
        quindi il blocco di kill non partiva MAI — ARIA si "dimenticava" del
        backend senza mai ucciderlo, VRAM/RAM restavano occupate a tempo
        indeterminato. Fallback al comportamento precedente per i backend
        che non implementano ancora il self-reporting del PID.

        2026-09-05 (Roberto, due bug trovati insieme durante il reprocess
        storico di Lifelog2 — qwen3-14b-q4km/Flux2 in alternanza sulla stessa
        GPU):

        (a) Il self-reporting del PID esisteva SOLO per flux2-klein-4b da
        agosto — mai esteso a qwen3-14b-q4km. Confermato dal vivo sui log:
        uno swap "Termino qwen3-14b-q4km per far posto a flux2-klein-4b" non
        era MAI seguito da nessuna riga di conferma (né PID né fallback
        titolo) — il ramo PID falliva silenziosamente (nessun pid-file da
        leggere) e il ramo fallback non partiva mai per lo stesso motivo di
        agosto (proc.poll() falso). Risultato: llama-server.exe restava vivo
        in VRAM mentre Flux2 tentava di caricarsi sulla stessa GPU. Esteso il
        self-reporting anche al launcher di qwen3-14b-q4km (vedi
        backends/lifelog_llm/launcher.py) — stessa causa, stesso fix già
        provato su Flux2 da un mese.

        (b) Anche quando il kill per PID riesce, uccide solo l'ALBERO
        radicato in quel PID — mai il cmd.exe ANTENATO che ha aperto la
        finestra reale (il PID auto-riportato è il processo del backend, non
        il cmd.exe che l'ha lanciato con 'start ... cmd.exe /k ...'). Prima
        di oggi la finestra restava quindi sempre aperta con un prompt vuoto
        dopo un kill per PID, anche quando il kill funzionava (visibile su
        Flux2 pure). Ora, dopo il kill per PID, si tenta SEMPRE anche la
        chiusura per titolo finestra (stessa chiamata già usata nel fallback
        sotto) — le due cose sono complementari, non alternative: PID per
        essere sicuri di uccidere il processo giusto, titolo per chiudere
        anche il guscio cmd.exe che lo conteneva. Aggiunto anche un controllo
        esplicito del codice di uscita di ogni taskkill (prima il codice
        dichiarava "terminato" incondizionatamente, senza controllare se il
        comando fosse davvero riuscito).

        2026-09-24 (Gap A1-5): sostituito il solo `_read_pid_file` con
        `_get_tracked_pid`, che prova anche `self._procs[model_id].pid`
        prima della scoperta via titolo finestra. Prima di oggi, se un
        backend non aveva pid-file (self-reporting mancante, es.
        qwen3-14b-q4km senza psutil nel suo env), questo ramo non faceva
        NULLA su Windows — nessun kill per PID, si contava solo sulla
        chiusura per titolo finestra sotto, che dipende a sua volta da un
        pid-file scritto altrove da _discover_pid_by_window_title() — PID
        che si è dimostrato dal vivo poter essere quello sbagliato
        (WindowsTerminal.exe invece del backend, vedi quella funzione).
        Ora `self._procs[model_id].pid` è sempre valido (spawn diretto,
        niente più wrapper 'start') e copre questo buco senza dipendere da
        nessun self-reporting né da nessuna scoperta per titolo."""
        with self._lock:
            real_pid = self._get_tracked_pid(model_id)
            if real_pid is not None:
                logger.info(f"{model_id}: terminazione via PID reale {real_pid}.")
                try:
                    if os.name == 'nt':
                        r = subprocess.run(f'taskkill /PID {real_pid} /T /F', shell=True, capture_output=True, timeout=5)
                        if r.returncode == 0:
                            logger.info(f"{model_id}: processo terminato (pid={real_pid}).")
                        else:
                            logger.warning(
                                f"{model_id}: taskkill /PID {real_pid} rc={r.returncode} — "
                                f"{r.stderr.decode(errors='replace').strip()!r}"
                            )
                    else:
                        psutil.Process(real_pid).terminate()
                        logger.info(f"{model_id}: processo terminato (pid={real_pid}).")
                except Exception:
                    logger.exception(f"{model_id}: errore terminando PID {real_pid}")
                self._forget_pid_file(model_id)
            else:
                proc = self._procs.get(model_id)
                if proc and proc.poll() is None:
                    logger.info(f"{model_id}: terminazione processo (idle timeout / shutdown, fallback titolo).")
                    if os.name != 'nt':
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        logger.info(f"{model_id}: processo terminato (fallback titolo).")

            # Chiusura della finestra Console (sempre tentata su Windows, non
            # solo nel ramo senza pid-file — vedi punto (b) sopra: un kill per
            # PID non tocca mai il cmd.exe antenato che ha aperto la finestra).
            if os.name == 'nt':
                title = f"ARIA Backend: {model_id}"
                r = subprocess.run(
                    f'taskkill /FI "WINDOWTITLE eq {title}*" /T /F', shell=True, capture_output=True, timeout=5
                )
                if r.returncode == 0:
                    logger.info(f"{model_id}: finestra Console chiusa (titolo {title!r}).")
                else:
                    # rc != 0 qui è spesso solo "nessuna finestra con quel titolo
                    # trovata" (già chiusa, o mai stata aperta con 'start') — non
                    # un errore da segnalare ad ogni giro, solo un debug.
                    logger.debug(
                        f"{model_id}: nessuna finestra Console da chiudere (titolo {title!r}, rc={r.returncode})."
                    )
            self._procs.pop(model_id, None)

    def _is_proc_active(self, model_id: str) -> bool:
        """Ritorna True se il processo è registrato e ancora attivo.

        2026-08-15: controlla prima il PID reale auto-riportato (affidabile),
        poi ricade sul Popen tracciato (inaffidabile su Windows per i
        backend lanciati via 'start cmd', vedi _kill_proc)."""
        if self._read_pid_file(model_id) is not None:
            return True
        proc = self._procs.get(model_id)
        return proc is not None and proc.poll() is None

    def shutdown_all(self):
        """Termina tutti i backend (e companion) all'arresto dell'orchestratore."""
        # 1. Kill dei processi tracciati via Popen
        for model_id in list(self._procs.keys()):
            self._kill_proc(model_id)
        
        # 2. Nuclear Option (Windows): Kill forzato per TITOLO per tutti i modelli nel manifest
        # Questo cattura processi "zombie" o avviati prima dell'orchestratore attuale.
        if os.name == 'nt':
            for model_id in self.MODEL_CONFIGS.keys():
                title = f"ARIA Backend: {model_id}"
                try:
                    subprocess.run(f'taskkill /F /FI "WINDOWTITLE eq {title}*"', shell=True, capture_output=True, timeout=5)
                except subprocess.TimeoutExpired:
                    pass

        self._procs.clear()
        self._idle_since.clear()



def _detect_local_ip():
    """Auto-detect dell'IP locale visibile sulla LAN."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_node_ip():
    """Legge l'IP del nodo dalle impostazioni, con fallback ad auto-detect."""
    try:
        from settings_gui import load_settings
        settings = load_settings()
        ip = settings.get("node_ip", "").strip()
        if ip:
            return ip
    except Exception:
        pass
    return _detect_local_ip()

class NodeOrchestrator:
    def __init__(self, redis_client):
        self.local_ip = get_node_ip()
        self.aria_root = ARIA_ROOT
        logger.info(f"Node IP resolved to: {self.local_ip}")
        self.qm = AriaQueueManager(redis_client)
        self.telemetry = TelemetryDB(ARIA_ROOT / "logs" / "aria-telemetry.db")
        self.qm.telemetry = self.telemetry
        self.optimizer = BatchOptimizer(redis_client)
        self.running = False
        self.thread = None
        self.http_thread = None

        # Semaforo locale copiato dalla Tray Icon
        self.semaphore_green = True

        # Cache RAM per token cloni
        self.token_cache = {}
        self.current_tasks = {} # model_id -> job_id

        # Backend lazy instances
        self._qwen3_backend = Qwen3TTSBackend() if _BACKENDS_AVAILABLE else None
        self._qwen35_backend = Qwen35LLMBackend() if _BACKENDS_AVAILABLE else None
        self._lifelog_llm_backend = LifelogLLMBackend() if _BACKENDS_AVAILABLE else None
        self._acestep_backend    = ACEStepBackend() if HAS_ACESTEP else None
        self._audiocraft_backend = AudiocraftBackend() if HAS_AUDIOCRAFT else None
        self._asr_backend        = LifelogASRBackend() if _BACKENDS_AVAILABLE else None
        self._whisperx_backend   = LifelogWhisperXBackend() if _BACKENDS_AVAILABLE else None
        self._flux_backend       = FluxImageGenBackend() if _BACKENDS_AVAILABLE and FluxImageGenBackend else None

        self.process_manager = ModelProcessManager(
            aria_root=ARIA_ROOT,
            miniconda_root=MINICONDA_ROOT,
        )

        # Global Rate Limiter for Cloud Tasks
        self.rate_limiter = GeminiRateLimiter(redis_client=redis_client)

        # Cloud Manager — handles sequential API tasks (Gemini, etc.)
        self.cloud_manager = CloudManager(
            queue_manager=self.qm,
            aria_root=ARIA_ROOT,
            rate_limiter=self.rate_limiter,
            current_tasks_ref=self.current_tasks
        )

        self.registry = AriaRegistryManager(
            aria_root=ARIA_ROOT,
            redis_client=redis_client,
            local_ip=self.local_ip
        )

    def _start_http_server(self):
        """Avvia l'Asset Server HTTP nativo per file statici su C:/Users/Roberto/aria/data"""
        os.makedirs(ARIA_OUTPUT_DIR, exist_ok=True)

        Handler = AriaAssetHandler
        # Per permettere restart puliti anche in caso di crash
        socketserver.TCPServer.allow_reuse_address = True

        with socketserver.TCPServer(("0.0.0.0", HTTP_PORT), Handler) as httpd:
            logger.info(f"Asset Server HTTP avviato su {self.local_ip}:{HTTP_PORT} (Serving ARIA_ROOT/data)")
            while self.running:
                httpd.handle_request()

    def start(self):
        if self.running: return
        self.running = True

        # Avvia Backend Orchestrator
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        # Start Cloud Manager
        self.cloud_manager.start()

        # Publish Master Registry (Discovery)
        self.registry.publish()

        logger.info("Orchestrator task loop and CloudManager started. Master Registry published.")

        # Avvia HTTP Asset Server Parallelo
        self.http_thread = threading.Thread(target=self._start_http_server, daemon=True)
        self.http_thread.start()

    def stop(self):
        self.running = False

        logger.info("--- SHUTDOWN SEQUENCE STARTED ---")

        # Give _run_loop time to exit its current iteration before we touch shared state
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=4)

        # 1. Stop dei manager logici
        logger.info("[1/5] Stopping CloudManager...")
        self.cloud_manager.stop()            # ferma il loop cloud

        logger.info("[2/5] Shutting down AI Backends...")
        self.process_manager.shutdown_all()   # termina tutti i backend locali
        
        # 2. Kill della Dashboard (se attiva)
        if os.name == 'nt':
            logger.info("[3/5] Terminating Dashboard server...")
            cmd_kill_db = 'powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like \'*dashboard/server.py*\' -or $_.CommandLine -like \'*dashboard\\\\server.py*\' } | Stop-Process -Force"'
            try:
                subprocess.run(cmd_kill_db, shell=True, capture_output=True, timeout=5)
            except Exception as e:
                logger.warning(f"Dashboard kill failed or timed out: {e}")
        
        # 3. Sblocco e stop dell'Asset Server HTTP
        logger.info("[4/5] Stopping Asset Server...")
        try:
             requests.get(f"http://127.0.0.1:{HTTP_PORT}/", timeout=0.5)
        except:
             pass
             
        if self.thread:
            self.thread.join(timeout=2)
        if self.http_thread:
            self.http_thread.join(timeout=2)

        # 4. Cleanup Redis
        logger.info("[5/5] Closing Redis connection...")
        try:
            self.qm.redis.close()
        except:
            pass
            
        logger.info("--- SHUTDOWN COMPLETE ---")


    def set_semaphore(self, state: bool):
        self.semaphore_green = state
        logger.info(f"Orchestrator semaphore set to {'GREEN' if state else 'RED'}")

    def _discover_voices(self) -> list:
        """Scansiona sia la nuova cartella data/assets/voices che la legacy data/voices."""
        voices = set()

        # 1. Nuova gerarchia (Standard)
        asset_voices_dir = self.aria_root / "data" / "assets" / "voices"
        if asset_voices_dir.exists():
            for d in asset_voices_dir.iterdir():
                if d.is_dir(): voices.add(d.name)

        # 2. Vecchia gerarchia (Legacy)
        legacy_voices_dir = self.aria_root / "data" / "voices"
        if legacy_voices_dir.exists():
            for d in legacy_voices_dir.iterdir():
                if d.is_dir(): voices.add(d.name)

        return list(voices)

        # Ogni sottocartella in data/voices/ è una voce
        return [d.name for d in voices_dir.iterdir() if d.is_dir()]

    def _send_heartbeat(self):
        """Invia lo stato del nodo a Redis per monitoraggio globale."""
        try:
            from datetime import datetime, timezone
            status = {
                "node_ip": self.local_ip,
                "status": "online",  # The node/gateway itself is online
                "gpu_status": "online" if self.semaphore_green else "paused",
                "cloud_status": "online",  # Cloud tasks are decoupled from GPU semaphore
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "active_backends": list(self.process_manager._procs.keys()),
                "current_tasks": self.current_tasks,
                "available_voices": self._discover_voices(),
            }
            key = f"aria:global:node:{self.local_ip}:status"
            self.qm.redis.set(key, json.dumps(status), ex=60)
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")

    def _run_loop(self):
        # Base models known by the node
        model_logic_ids = ["fish-s1-mini", "qwen3-tts-1.7b", "qwen3-tts-custom", "qwen3.5-35b-moe-q3ks", "qwen3-14b-q4km", "acestep-1.5-xl-sft", "qwen3-asr-1.7b", "whisperx-large-v3", "flux2-klein-4b"]
        current_model = None

        last_heartbeat = 0
        while self.running:
            # 1. Discover all active LOCAL queues for these models
            # New Pattern: aria:q:*:local:{model_id}:*
            known_models = {}
            for model_id in model_logic_ids:
                pattern = self.optimizer.build_queue_key("*", model_id, "local", "*")
                for q_key in self.qm.redis.scan_iter(match=pattern):
                    # Map the specific client queue to the model logic ID for the optimizer
                    known_models[f"{model_id}:{q_key}"] = q_key
            
            if known_models:
                logger.info(f"Discovered queues: {list(known_models.values())}")
            # Heartbeat ogni 5 secondi per dashboard reattiva
            if time.time() - last_heartbeat > 5:
                self._send_heartbeat()
                last_heartbeat = time.time()

            if not self.semaphore_green:
                time.sleep(2)
                continue

            # Spegni backend idle
            self.process_manager.shutdown_idle_backends()

            try:
                decision = self.optimizer.decide_next_queue(known_models, current_model)
                if not decision:
                    # Nessuna coda attiva — marca tutti i backend caricati come idle
                    # Usa _procs.keys() invece di known_models: quando le code sono vuote
                    # known_models è vuoto e mark_idle() non verrebbe mai chiamato,
                    # impedendo a shutdown_idle_backends() di spegnere i backend inattivi.
                    for mid in list(self.process_manager._procs.keys()):
                        self.process_manager.mark_idle(mid)
                    time.sleep(1)
                    continue

                next_model_id, queue_key = decision
                if current_model != next_model_id:
                    logger.info(f"Switching batch focus to model: {next_model_id} (queue: {queue_key})")
                    current_model = next_model_id

                raw_json, payload = self.qm.fetch_task(queue_key, timeout=2)
                if not payload:
                    # Coda vuota per questo modello
                    self.process_manager.mark_idle(next_model_id)
                    continue

                # Coda ha un task — assicurati che il backend sia attivo
                # Split model_id if it's a composite logic ID (model:queue)
                base_model_id = next_model_id.split(':')[0] if ':' in next_model_id else next_model_id

                if not self.process_manager.ensure_running(base_model_id):
                    logger.error(f"Backend {base_model_id} (from {next_model_id}) non disponibile, task riaccodato.")
                    # Re-inserisce il task in testa alla coda
                    import redis as _redis_mod
                    # Il task era già prelevato, lo re-incodiamo
                    self.qm.redis.lpush(queue_key, raw_json)
                    time.sleep(10)
                    continue

                logger.info(f"Processing task {payload.job_id} for {payload.model_id}")
                self.current_tasks[base_model_id] = payload.job_id
                try:
                    self._process_task(payload)
                finally:
                    self.current_tasks.pop(base_model_id, None)

            except Exception as e:
                logger.error("Error in orchestrator loop", exc_info=True)
                time.sleep(5)

    def _encode_audio_to_tokens(self, audio_path: str) -> tuple:
        """Returns (tokens_bytes, wav_bytes) for voice cloning request."""
        logger.info(f"Encoding reference audio from {audio_path}...")
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            resp = requests.post(f"{FISH_ENCODE_HOST}/encode", files={"file": ("ref.wav", audio_bytes)}, timeout=120)
            resp.raise_for_status()
            logger.info("Tokens encoded successfully.")
            data = resp.json()
            tokens_b64 = data.get("npy_base64", "")
            # Return both the decoded NPY token bytes AND the original WAV bytes.
            # Fish/v1/tts uses `audio` (wav b64) for voice cloning identity,
            # and `tokens` (npy b64) as a server-side cache key / shortcut.
            return base64.b64decode(tokens_b64), audio_bytes
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def _get_wav_duration(self, file_path: Path) -> float:
        import wave
        try:
            with wave.open(str(file_path), 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                return frames / float(rate)
        except Exception as e:
            logger.error(f"Errore calcolo durata WAV {file_path}: {e}")
            return 0.0

    def _process_task(self, task):
        start_t = time.time()

        # --- Idempotency Check (SOA v2.1) ---
        if task.model_type == "imagegen" or task.model_id == "flux2-klein-4b":
            filename = f"{task.job_id}.jpeg"
        elif task.model_id == "fish-s1-mini":
            filename = f"{task.job_id}_scene-001.wav"
        else:
            filename = f"{task.job_id}.wav"

        local_out_path = ARIA_OUTPUT_DIR / filename

        if local_out_path.exists():
            logger.info(f"Idempotenza ARIA: File {filename} già presente. Salto inferenza.")
            public_url = f"http://{self.local_ip}:{HTTP_PORT}/{filename}"

            if task.model_type == "imagegen" or task.model_id == "flux2-klein-4b":
                output_payload = {
                    "image_url":  public_url,
                    "local_path": str(local_out_path),
                    "cached":     True,
                }
            else:
                duration_s = self._get_wav_duration(local_out_path)
                output_payload = {
                    "audio_url":        public_url,
                    "duration_seconds": duration_s,
                    "cached":           True,
                }

            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="done",
                processing_time_seconds=time.time() - start_t,
                output=output_payload,
            )
            self.qm.post_result(task, result)
            return

        if task.model_id.startswith("qwen3-tts"):
            self._process_qwen3_task(task, start_t)
        elif task.model_id == "audiocraft-medium":
            self._process_audiocraft_task(task, start_t)
        elif task.model_id == "acestep-1.5-xl-sft" or task.model_type == "mus":
            self._process_acestep_task(task, start_t)
        elif task.model_id == "qwen3.5-35b-moe-q3ks":
            self._process_llm_task(task, start_t)
        elif task.model_id == "qwen3-14b-q4km":
            self._process_lifelog_llm_task(task, start_t)
        elif task.model_id == "flux2-klein-4b" or task.model_type == "imagegen":
            self._process_flux_task(task, start_t)
        elif task.model_id in ("qwen3-asr-1.7b", "whisperx-large-v3") or task.model_type == "stt":
            self._process_asr_task(task, start_t)
        elif task.model_id == "fish-s1-mini":
            try:
                # --- Intent-based Resolution (SOA v2.0) ---
                voice_id = task.payload.get("voice_id")
                voice_local_path = None
                prompt_text = task.payload.get("prompt_text") # Optional override

                if voice_id:
                    # Resolve from internal Voice Library
                    voice_library_dir = ARIA_ROOT / "data" / "voices"
                    voice_dir = voice_library_dir / voice_id
                    ref_wav = voice_dir / "ref.wav"
                    ref_txt_file = voice_dir / "ref.txt"

                    if ref_wav.exists():
                        voice_local_path = str(ref_wav)
                        logger.info(f"Resolved intent '{voice_id}' to {voice_local_path}")

                        # Resolve prompt text if not provided
                        if not prompt_text and ref_txt_file.exists():
                            try:
                                # Try UTF-8 first
                                try:
                                    with open(ref_txt_file, "r", encoding="utf-8") as f:
                                        prompt_text = f.read().strip()
                                except UnicodeDecodeError:
                                    # Fallback to latin-1/iso-8859-1 for Windows-style files
                                    with open(ref_txt_file, "r", encoding="latin-1") as f:
                                        prompt_text = f.read().strip()
                                    logger.info(f"Read ref.txt using latin-1 fallback for '{voice_id}'")

                                logger.info(f"Resolved reference text for '{voice_id}' from ref.txt")
                            except Exception as e:
                                logger.error(f"Failed to read ref.txt for {voice_id}: {e}")
                    else:
                        logger.warning(f"Voice ID '{voice_id}' requested but path {ref_wav} not found.")

                # Legacy fallback if no voice_id or lookup failed
                if not voice_local_path:
                    if getattr(task, "file_refs", None) and getattr(task.file_refs, "input", None):
                       for ref in task.file_refs.input:
                           if ref.ref_id == "voice_reference" and ref.local_path:
                               voice_local_path = ref.local_path

                # Gestione Reference Token (Da path locale risolto o inviato)
                tokens = None
                tokens_bytes = None
                wav_bytes_ref = None
                if voice_local_path:
                    if voice_local_path in self.token_cache:
                        tokens_bytes, wav_bytes_ref = self.token_cache[voice_local_path]
                    else:
                        tokens_bytes, wav_bytes_ref = self._encode_audio_to_tokens(voice_local_path)
                        self.token_cache[voice_local_path] = (tokens_bytes, wav_bytes_ref)

                # Synthesize TTS with Chunking
                base_data = {
                    "format": task.payload.get("output_format", "wav"),
                    "streaming": False,
                    "normalize": False,  # CRITICAL: Keep false to preserve emotion tags and breaks
                    "temperature": task.payload.get("temperature", 0.7),
                    "top_p": task.payload.get("top_p", 0.8),  # Increased top_p for more prosody variation
                    "repetition_penalty": task.payload.get("repetition_penalty", 1.1)
                }

                if tokens_bytes and wav_bytes_ref:
                    tokens_b64 = base64.b64encode(tokens_bytes).decode("utf-8")
                    # IMPORTANT: `audio` must be the original WAV bytes (not NPY tokens).
                    # Fish uses `audio` for voice identity (accent/timbre cloning).
                    # `tokens` is only used as a server-side optimisation/cache key.
                    wav_b64 = base64.b64encode(wav_bytes_ref).decode("utf-8")
                    if not prompt_text:
                        prompt_text = "Il cammino dell'uomo timorato è minacciato da ogni parte dalle iniquità degli esseri egoisti e dalla tirannia degli uomini malvagi."
                        logger.warning("No prompt_text found, using default for accent stability.")

                    base_data["references"] = [{
                        "tokens": tokens_b64,
                        "audio": wav_b64,  # Raw WAV bytes for voice cloning identity
                        "text": prompt_text
                    }]

                full_text = task.payload.get("text", "")

                # --- Advanced Chunking & Silence Injection ---
                # Strategy: split by (break), (long-break), and \n\n.
                # Generate real WAV silence for pauses instead of relying on Fish TTL.
                import re
                raw_segments = re.split(r'(\(long-break\)|\(break\)|\n\n)', full_text)

                actions = []
                for seg in raw_segments:
                    if seg == "(long-break)":
                        actions.append({"type": "silence", "duration": 1.5})
                    elif seg == "(break)":
                        actions.append({"type": "silence", "duration": 0.5})
                    elif seg == "\n\n":
                        actions.append({"type": "silence", "duration": 0.5})
                    else:
                        text_seg = seg.strip()
                        if text_seg:
                            # Split into max 120 words chunks if still too long
                            words = text_seg.split()
                            for w_i in range(0, len(words), 120):
                                chunk_text = " ".join(words[w_i:w_i+120])
                                actions.append({"type": "text", "content": chunk_text})

                logger.info(f"Text parsed into {len(actions)} sequential actions (speech + silence).")

                audio_parts = []
                for i, action in enumerate(actions):
                    if action["type"] == "silence":
                        logger.info(f"Action {i+1}/{len(actions)}: Injecting silence {action['duration']}s")
                        audio_parts.append(float(action["duration"]))

                    elif action["type"] == "text":
                        chunk = action["content"]
                        logger.info(f"Action {i+1}/{len(actions)}: TTS chunk ({len(chunk.split())} words)")
                        chunk_data = base_data.copy()

                        # Prepend a sacrificial break to absorb the S1-mini first-word cutoff bug
                        safe_chunk = f"(break) {chunk}"
                        chunk_data["text"] = safe_chunk

                        # Diagnostic log for only the first text chunk
                        if not any(isinstance(p, bytes) for p in audio_parts):
                            log_data = chunk_data.copy()
                            if "references" in log_data:
                                log_data["references"] = [{
                                    "text": r["text"],
                                    "audio": f"<b64_string_len_{len(r['audio'])}>"
                                } for r in log_data["references"]]
                            logger.info(f"First Chunk TTS Payload Diagnostic: {json.dumps(log_data, indent=2)}")

                        logger.info(f"Requesting TTS Synthesis to Fish Server at {FISH_TTS_HOST}/v1/tts")
                        resp = requests.post(f"{FISH_TTS_HOST}/v1/tts", json=chunk_data, timeout=900)
                        resp.raise_for_status()
                        audio_parts.append(resp.content)

                # Merge chunks and silence
                audio_bytes = self._merge_wavs(audio_parts)

                duration_s = time.time() - start_t

                # Salvataggio Asset Piatto nella HTTP Directory Locale
                filename = f"{task.job_id}_scene-001.wav"
                local_out_path = ARIA_OUTPUT_DIR / filename

                with open(local_out_path, "wb") as f:
                     f.write(audio_bytes)
                logger.info(f"Wrote generated WAV to {local_out_path}")

                # Ritorna l'URL HTTP Pubblico al Container LXC / Client
                public_url = f"http://{self.local_ip}:{HTTP_PORT}/outputs/{filename}"

                result = AriaTaskResult(
                    job_id=task.job_id,
                    client_id=task.client_id,
                    model_type=task.model_type,
                    model_id=task.model_id,
                    status="done",
                    processing_time_seconds=duration_s,
                    output={"audio_url": public_url, "duration_seconds": duration_s} # Niente più "output_path" grezzo Unix
                )
                self.qm.post_result(task, result)

            except Exception as e:
                logger.error(f"Task Failed: {e}", exc_info=True)
                result = AriaTaskResult(
                    job_id=task.job_id,
                    client_id=task.client_id,
                    model_type=task.model_type,
                    model_id=task.model_id,
                    status="error",
                    processing_time_seconds=time.time() - start_t,
                    error=str(e)
                )
                self.qm.post_result(task, result)
        else:
            logger.warning(f"Unsupported model_id: {task.model_id}")
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="error",
                processing_time_seconds=time.time() - start_t,
                error=f"model_id non supportato: {task.model_id}"
            )
            self.qm.post_result(task, result)

    def _process_qwen3_task(self, task, start_t: float):
        """Dispatch di un task TTS verso Qwen3TTSBackend."""
        if not self._qwen3_backend:
            raise RuntimeError("Qwen3TTSBackend non disponibile (import fallito).")

        # Garantisce che il modello corretto sia in esecuzione (Gestione Swap JIT)
        if not self.process_manager.ensure_running(task.model_id):
            raise RuntimeError(f"Impossibile avviare il backend Qwen3 per {task.model_id}")

        try:
            # Assicura che il job_id sia presente nel payload per il salvataggio file
            if "job_id" not in task.payload:
                task.payload["job_id"] = task.job_id

            result_data = self._qwen3_backend.run(
                payload=task.payload,
                aria_root=ARIA_ROOT,
                local_ip=self.local_ip
            )
            duration_s = time.time() - start_t
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="done",
                processing_time_seconds=duration_s,
                output={
                    "audio_url":        result_data["audio_url"],
                    "duration_seconds": result_data.get("duration_seconds"),
                    "chunks_count":     result_data.get("chunks_count"),
                    "metrics":          result_data.get("metrics", {}),
                }
            )
            self.qm.post_result(task, result)
        except Exception as e:
            logger.error(f"Qwen3 task failed: {e}", exc_info=True)
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="error",
                processing_time_seconds=time.time() - start_t,
                error=str(e)
            )
            self.qm.post_result(task, result)
    def _process_llm_task(self, task, start_t: float):
        """Dispatch di un task LLM verso Qwen35LLMBackend."""
        if not self._qwen35_backend:
            raise RuntimeError("Qwen35LLMBackend non disponibile.")

        if not self.process_manager.ensure_running(task.model_id):
            raise RuntimeError(f"Impossibile avviare il backend LLM per {task.model_id}")

        try:
            result_data = self._qwen35_backend.run(
                payload=task.payload,
                aria_root=ARIA_ROOT,
                local_ip=self.local_ip
            )
            duration_s = time.time() - start_t

            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="done",
                processing_time_seconds=duration_s,
                output={
                    "text":     result_data["text"],
                    "thinking": result_data.get("thinking"),
                    "usage":    result_data.get("usage")
                }
            )
            self.qm.post_result(task, result)
        except Exception as e:
            logger.error(f"LLM task failed: {e}", exc_info=True)
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="error",
                processing_time_seconds=time.time() - start_t,
                error=str(e)
            )
            self.qm.post_result(task, result)


    def _process_lifelog_llm_task(self, task, start_t: float):
        """Dispatch di un task LLM enrichment verso LifelogLLMBackend (Qwen3-14B Q4_K_M, porta 8090).

        Ricarica il modulo del backend ad ogni task: è stateless e i task LLM sono radi, così
        un deploy del wrapper (git pull) NON richiede il riavvio dell'orchestratore. Se il reload
        fallisce si ricade sull'istanza creata all'avvio (comportamento precedente).
        """
        backend = self._lifelog_llm_backend
        try:
            import importlib
            _mod = (sys.modules.get("backends.lifelog_llm")
                    or sys.modules.get("aria_node_controller.backends.lifelog_llm"))
            if _mod is not None:
                importlib.reload(_mod)
                backend = _mod.LifelogLLMBackend()
        except Exception as e:
            logger.warning("LifelogLLM: reload modulo fallito, uso l'istanza di avvio: %s", e)
            backend = self._lifelog_llm_backend

        if not backend:
            raise RuntimeError("LifelogLLMBackend non disponibile.")

        if not self.process_manager.ensure_running(task.model_id):
            raise RuntimeError(f"Impossibile avviare il backend Lifelog LLM per {task.model_id}")

        try:
            result_data = backend.run(
                payload=task.payload,
                aria_root=ARIA_ROOT,
                local_ip=self.local_ip,
            )
            duration_s = time.time() - start_t

            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="done",
                processing_time_seconds=duration_s,
                output={
                    "text":          result_data["text"],
                    "thinking":      result_data.get("thinking"),
                    "usage":         result_data.get("usage"),
                    "finish_reason": result_data.get("finish_reason"),
                },
            )
            self.qm.post_result(task, result)
        except Exception as e:
            logger.error("Lifelog LLM task failed: %s", e, exc_info=True)
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="error",
                processing_time_seconds=time.time() - start_t,
                error=str(e),
            )
            self.qm.post_result(task, result)

    def _process_flux_task(self, task, start_t: float):
        """Dispatch di un task image generation verso FluxImageGenBackend (FLUX.2-klein-4B, porta 8092)."""
        if not self._flux_backend:
            raise RuntimeError("FluxImageGenBackend non disponibile.")

        if not self.process_manager.ensure_running(task.model_id):
            raise RuntimeError(f"Impossibile avviare il backend FLUX per {task.model_id}")

        try:
            result_data = self._flux_backend.run(
                payload=task.payload,
                aria_root=ARIA_ROOT,
                local_ip=self.local_ip,
            )
            duration_s = time.time() - start_t

            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="done",
                processing_time_seconds=duration_s,
                output=result_data,
            )
            self.qm.post_result(task, result)
        except Exception as e:
            logger.error("FLUX task failed: %s", e, exc_info=True)
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="error",
                processing_time_seconds=time.time() - start_t,
                error=str(e),
            )
            self.qm.post_result(task, result)

    def _merge_wavs(self, audio_parts: list) -> bytes:
        """Merge wav chunks and inject silences. Assumes same format/samplerate from the first WAV part."""
        import wave
        import io

        if not audio_parts:
            return b""

        # Find the first real WAV to get params
        params = None
        for part in audio_parts:
            if isinstance(part, bytes):
                with wave.open(io.BytesIO(part), 'rb') as w:
                    params = w.getparams()
                break

        if not params:
            # Only silences?
            return b""

        out_buf = io.BytesIO()
        with wave.open(out_buf, 'wb') as w_out:
            w_out.setparams(params)

            for part in audio_parts:
                if isinstance(part, bytes):
                    with wave.open(io.BytesIO(part), 'rb') as w_in:
                        frames = w_in.readframes(w_in.getnframes())
                        w_out.writeframes(frames)
                elif isinstance(part, float):
                    # Generate silence based on params
                    num_frames = int(params.framerate * part)
                    bytes_per_frame = params.nchannels * params.sampwidth
                    w_out.writeframes(b'\x00' * (num_frames * bytes_per_frame))

        return out_buf.getvalue()
    def _process_acestep_task(self, task, start_t: float):
        """Dispatch di un task MUS verso ACEStepBackend (Dias Sound Engine)."""
        if not self._acestep_backend:
            raise RuntimeError("ACEStepBackend non disponibile (import fallito).")

        # Gestione JIT: assicura che il server musicale sia attivo
        if not self.process_manager.ensure_running(task.model_id):
            raise RuntimeError(f"Impossibile avviare il backend musicale per {task.model_id}")

        try:
            result_data = self._acestep_backend.run(
                payload=task.payload,
                aria_root=ARIA_ROOT,
                local_ip=self.local_ip
            )
            duration_s = time.time() - start_t

            output: dict = {
                "audio_url":        result_data.get("audio_url"),
                "duration_seconds": result_data.get("duration_seconds"),
            }
            # Stem URLs presenti solo se run_demucs=True e HTDemucs ha avuto successo
            if result_data.get("stems"):
                output["stems"] = result_data["stems"]

            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="done",
                processing_time_seconds=duration_s,
                output=output,
            )
            self.qm.post_result(task, result)
            logger.info(
                f"Music Task Completed: {task.job_id}"
                + (f" | stems: {list(result_data['stems'].keys())}" if result_data.get("stems") else "")
            )

        except Exception as e:
            logger.error(f"Music Task Failed: {e}", exc_info=True)
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="error",
                processing_time_seconds=time.time() - start_t,
                error=str(e)
            )
            self.qm.post_result(task, result)

    def _process_audiocraft_task(self, task, start_t: float):
        """Dispatch di un task AMB/SFX/STING verso AudiocraftBackend."""
        if not self._audiocraft_backend:
            raise RuntimeError("AudiocraftBackend non disponibile.")

        if not self.process_manager.ensure_running(task.model_id):
            raise RuntimeError(f"Impossibile avviare il backend Audiocraft ({task.model_id})")

        try:
            result_data = self._audiocraft_backend.run(
                payload=task.payload,
                aria_root=ARIA_ROOT,
                local_ip=self.local_ip,
            )
            duration_s = time.time() - start_t

            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="done",
                processing_time_seconds=duration_s,
                output={
                    "audio_url":        result_data.get("audio_url"),
                    "duration_seconds": result_data.get("duration_seconds"),
                },
            )
            self.qm.post_result(task, result)
            logger.info(f"Audiocraft Task Completed: {task.job_id}")

        except Exception as e:
            logger.error(f"Audiocraft Task Failed: {e}", exc_info=True)
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="error",
                processing_time_seconds=time.time() - start_t,
                error=str(e),
            )
            self.qm.post_result(task, result)

    def _process_asr_task(self, task, start_t: float):
        """Dispatch di un task STT verso il backend appropriato (qwen3-asr o whisperx)."""
        if task.model_id == "whisperx-large-v3":
            backend = self._whisperx_backend
        else:
            backend = self._asr_backend

        if not backend:
            raise RuntimeError(f"Backend STT non disponibile per {task.model_id}.")

        if not self.process_manager.ensure_running(task.model_id):
            raise RuntimeError(f"Impossibile avviare il backend ASR per {task.model_id}")

        try:
            result_data = backend.run(
                payload=task.payload,
                aria_root=ARIA_ROOT,
                local_ip=self.local_ip,
            )
            duration_s = time.time() - start_t

            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="done",
                processing_time_seconds=duration_s,
                output=result_data.get("output", {})
            )
            self.qm.post_result(task, result)
            logger.info(f"ASR Task Completed: {task.job_id}")

        except Exception as e:
            logger.error(f"ASR Task Failed: {e}", exc_info=True)
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="error",
                processing_time_seconds=time.time() - start_t,
                error=str(e),
            )
            self.qm.post_result(task, result)
