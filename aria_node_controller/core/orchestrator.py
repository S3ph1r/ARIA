import os
import threading
import time
import requests
import json
import base64
import socket
import http.server
import socketserver
import subprocess


from pathlib import Path
from .queue_manager import AriaQueueManager
from .cloud_manager import CloudManager
from .rate_limiter import GeminiRateLimiter
from .batch_optimizer import BatchOptimizer
from .logger import get_logger
import re
from .models import AriaTaskResult

# Backends
try:
    from backends.qwen3_tts import Qwen3TTSBackend
    from backends.qwen35_llm import Qwen35LLMBackend
    _BACKENDS_AVAILABLE = True
except ImportError:
    try:
        from ..backends.qwen3_tts import Qwen3TTSBackend
        from ..backends.qwen35_llm import Qwen35LLMBackend
        _BACKENDS_AVAILABLE = True
    except ImportError:
        _BACKENDS_AVAILABLE = False


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
        self._lock = threading.Lock()

    def _build_cmd(self, model_id: str) -> list:
        """Costruisce il comando di avvio per il modello dato."""
        if model_id == "voice-cloning":
            python   = str(self.aria_root / "envs" / "fish-speech-env" / "python.exe")
            fish_dir = self.aria_root / "envs" / "fish-speech"
            return [python, str(fish_dir / "voice_cloning_server.py")]
        elif model_id == "fish-s1-mini":

            python = str(self.aria_root / "envs" / "fish-speech-env" / "python.exe")
            fish_dir  = self.aria_root / "envs" / "fish-speech"
            model_dir = self.aria_root / "data" / "models" / "fish-s1-mini"
            return [
                python,
                str(fish_dir / "tools" / "api_server.py"),
                "--listen", "0.0.0.0:8080",
                "--llama-checkpoint-path",   str(model_dir),
                "--decoder-checkpoint-path", str(model_dir / "codec.pth"),
                "--decoder-config-name",     "modded_dac_vq",
            ]
        elif model_id in ["qwen3-tts-1.7b", "qwen3-tts-custom"]:
            python = str(self.aria_root / "envs" / "qwen3tts" / "python.exe")
            server = self.aria_root / "aria_node_controller" / "qwen3_server.py"
            
            # Determina il path del modello
            model_sub = "qwen3-tts-1.7b" if model_id == "qwen3-tts-1.7b" else "qwen3-tts-1.7b-customvoice"
            model_path = str(self.aria_root / "data" / "models" / model_sub)
            
            port = self.MODEL_CONFIGS[model_id]["port"]
            
            return [
                python, 
                str(server),
                "--model-path", model_path,
                "--port", str(port)
            ]
        elif model_id == "qwen3.5-35b-moe-q3ks":
            python = str(self.aria_root / "envs" / "nh-qwen35-llm" / "python.exe")
            server = self.aria_root / "aria_node_controller" / "llm_server.py"
            return [python, str(server)]
        else:
            raise ValueError(f"Nessuna configurazione per model_id='{model_id}'")

    def _health_check(self, model_id: str) -> bool:
        url = self.MODEL_CONFIGS[model_id]["health_url"]
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
            # 0. Gestione Conflitti Porta (SOA v2.1)
            # Se un ALTRO modello sta usando la stessa porta, dobbiamo killarlo 
            # per liberare la GPU e il socket.
            target_port = self.MODEL_CONFIGS[model_id].get("port")
            if target_port:
                for other_id, other_config in self.MODEL_CONFIGS.items():
                    if other_id != model_id and other_config.get("port") == target_port:
                        if self._is_proc_active(other_id):
                            logger.info(f"Port Conflict: {other_id} occupa porta {target_port}. Termino per far posto a {model_id}...")
                            self._kill_proc(other_id)

            # 1. Controllo proattivo...
            if self._health_check(model_id):
                logger.info(f"{model_id}: backend già attivo e responsivo (rilevato esternamente).")
                self._idle_since.pop(model_id, None)
                return True

            proc = self._procs.get(model_id)

            # Il processo era morto o non esisteva
            if proc and proc.poll() is not None:
                logger.warning(f"{model_id}: processo terminato inaspettatamente, riavvio...")
            elif proc and proc.poll() is None:
                # Se arriviamo qui, il processo esiste ma NON è responsive (altrimenti saremmo usciti sopra)
                logger.warning(f"{model_id}: processo attivo ma non risponde. Lo termino per riavvio pulito...")
                self._kill_proc(model_id)

            # Avvio
            try:
                cmd = self._build_cmd(model_id)
                logger.info(f"Avvio backend {model_id} in finestra Console dedicata...")
                
                if os.name == 'nt':
                    # Su Windows avvia una Command Window separata e la tiene aperta (cmd /k)
                    # Convertiamo cmd in una stringa stando attenti agli spazi.
                    # Il "Magic CMD escape bug" su Windows richiede che l'intera stringa successiva a /k
                    # venga wrappata in quote esterne se i parametri interni hanno stringhe.
                    cmd_str = " ".join(f'"{c}"' for c in cmd)
                    title = f"ARIA Backend: {model_id}"
                    
                    # Iniezione PATH per SoX e altre dipendenze core
                    env = os.environ.copy()
                    sox_path = str(self.aria_root / "envs" / "sox" / "Library" / "bin")
                    if os.path.exists(sox_path):
                        env["PATH"] = sox_path + os.pathsep + env.get("PATH", "")
                        logger.info(f"Injected SoX path: {sox_path}")

                    new_proc = subprocess.Popen(
                        f'start "{title}" cmd.exe /k "{cmd_str}"',
                        shell=True,
                        cwd=str(self.aria_root),
                        env=env
                    )
                else:
                    # Fallback standard per Linux/Mac (Mantiene i log su file per non sporcare stdout)
                    log_dir = self.aria_root / "logs"
                    log_dir.mkdir(exist_ok=True)
                    log_out = open(log_dir / f"startup_{model_id.replace('-','_')}.log", "a")
                    log_err = open(log_dir / f"startup_{model_id.replace('-','_')}_err.log", "a")

                    new_proc = subprocess.Popen(
                        cmd,
                        stdout=log_out,
                        stderr=log_err,
                        cwd=str(self.aria_root),
                    )
                    
                self._procs[model_id] = new_proc
                self._idle_since.pop(model_id, None)
            except Exception as e:
                logger.error(f"Impossibile avviare {model_id}: {e}")
                return False

        # Wait for health check (fuori dal lock per non bloccare)
        max_wait = self.MODEL_CONFIGS[model_id]["startup_wait"]
        logger.info(f"Attesa health check {model_id} (max {max_wait}s)...")
        for _ in range(max_wait):
            if self._health_check(model_id):
                logger.info(f"{model_id}: health check OK")
                return True
            time.sleep(1)

        logger.error(f"{model_id}: timeout health check ({max_wait}s)")
        return False

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

    def _kill_proc(self, model_id: str):
        """Termina il processo di un singolo modello se attivo."""
        with self._lock:
            proc = self._procs.get(model_id)
            if proc and proc.poll() is None:
                logger.info(f"{model_id}: terminazione processo (idle timeout / shutdown).")
                
                if os.name == 'nt':
                    # Siccome abbiamo lanciato con 'start cmd', il Popen originale è solo
                    # l'esecutore 'start'. Dobbiamo killare l'albero processi reale dal titolo.
                    title = f"ARIA Backend: {model_id}"
                    subprocess.run(f'taskkill /FI "WINDOWTITLE eq {title}*" /T /F', shell=True, capture_output=True)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        
                logger.info(f"{model_id}: processo terminato.")
            self._procs.pop(model_id, None)

    def _is_proc_active(self, model_id: str) -> bool:
        """Ritorna True se il processo è registrato e ancora attivo."""
        proc = self._procs.get(model_id)
        return proc is not None and proc.poll() is None

    def shutdown_all(self):
        """Termina tutti i backend (e companion) all'arresto dell'orchestratore."""
        for model_id in list(self._procs.keys()):
            self._kill_proc(model_id)
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
        self.optimizer = BatchOptimizer(redis_client)
        self.running = False
        self.thread = None
        self.http_thread = None
        
        # Semaforo locale copiato dalla Tray Icon
        self.semaphore_green = True
        
        # Cache RAM per token cloni
        self.token_cache = {}

        # Backend lazy instances
        self._qwen3_backend = Qwen3TTSBackend() if _BACKENDS_AVAILABLE else None
        self._qwen35_backend = Qwen35LLMBackend() if _BACKENDS_AVAILABLE else None

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
            rate_limiter=self.rate_limiter
        )

        
    def _start_http_server(self):
        """Avvia l'Asset Server HTTP nativo per file statici su C:/Users/Roberto/aria/data/outputs"""
        os.makedirs(ARIA_OUTPUT_DIR, exist_ok=True)
        # Cambia working directory per il SimpleHTTPRequestHandler
        os.chdir(ARIA_OUTPUT_DIR)
        
        Handler = http.server.SimpleHTTPRequestHandler
        # Per permettere restart puliti anche in caso di crash
        socketserver.TCPServer.allow_reuse_address = True
        
        with socketserver.TCPServer(("0.0.0.0", HTTP_PORT), Handler) as httpd:
            logger.info(f"Asset Server HTTP avviato su {self.local_ip}:{HTTP_PORT} (Serving {ARIA_OUTPUT_DIR})")
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
        
        logger.info("Orchestrator task loop and CloudManager started")
        
        # Avvia HTTP Asset Server Parallelo
        self.http_thread = threading.Thread(target=self._start_http_server, daemon=True)
        self.http_thread.start()

    def stop(self):
        self.running = False
        self.process_manager.shutdown_all()   # termina Fish e Qwen3 se attivi
        self.cloud_manager.stop()            # ferma il loop cloud
        # Server HTTP gestirà al massimo una richiesta fake per sbloccarsi
        try:
             requests.get(f"http://127.0.0.1:{HTTP_PORT}/", timeout=1)
        except:
             pass
        if self.thread:
            self.thread.join(timeout=3)
        if self.http_thread:
            self.http_thread.join(timeout=3)
        logger.info("Orchestrator thread stopped")


    def set_semaphore(self, state: bool):
        self.semaphore_green = state
        logger.info(f"Orchestrator semaphore set to {'GREEN' if state else 'RED'}")

    def _discover_voices(self) -> list:
        """Scansiona la cartella data/voices per trovare le voci disponibili."""
        voices_dir = self.aria_root / "data" / "voices"
        if not voices_dir.exists():
            return []
        
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
                "available_voices": self._discover_voices(),
            }
            key = f"aria:global:node:{self.local_ip}:status"
            self.qm.redis.set(key, json.dumps(status), ex=60)
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")

    def _run_loop(self):
        # Base models known by the node
        model_logic_ids = ["fish-s1-mini", "qwen3-tts-1.7b", "qwen3-tts-custom", "qwen3.5-35b-moe-q3ks"]
        current_model = None
        
        last_heartbeat = 0
        while self.running:
            # 1. Discover all active LOCAL queues for these models
            # Pattern: global:queue:*:local:{model_id}:*
            known_models = {}
            for model_id in model_logic_ids:
                pattern = self.optimizer.build_queue_key("*", model_id, "local", "*")
                for q_key in self.qm.redis.scan_iter(match=pattern):
                    # Map the specific client queue to the model logic ID for the optimizer
                    known_models[f"{model_id}:{q_key}"] = q_key
            # Heartbeat ogni 20 secondi
            if time.time() - last_heartbeat > 20:
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
                    # Nessuna coda attiva — marca tutti come idle
                    for mid in known_models:
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
                self._process_task(payload)

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
        # Determiniamo il nome file atteso per questo task
        if task.model_id == "fish-s1-mini":
            filename = f"{task.job_id}_scene-001.wav"
        else:
            filename = f"{task.job_id}.wav"
            
        local_out_path = ARIA_OUTPUT_DIR / filename
        
        if local_out_path.exists():
            logger.info(f"Idempotenza ARIA: File {filename} già presente. Salto inferenza.")
            duration_s = self._get_wav_duration(local_out_path)
            # Nota: Uniformiamo il porto del server asset (8082) per ogni tipo di output
            public_url = f"http://{self.local_ip}:{HTTP_PORT}/{filename}"
            
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                model_id=task.model_id,
                status="done",
                processing_time_seconds=time.time() - start_t,
                output={
                    "audio_url": public_url, 
                    "duration_seconds": duration_s,
                    "cached": True
                }
            )
            self.qm.post_result(task, result)
            return

        if task.model_id.startswith("qwen3-tts"):
            self._process_qwen3_task(task, start_t)
        elif task.model_id == "qwen3.5-35b-moe-q3ks":
            self._process_llm_task(task, start_t)
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
                public_url = f"http://{self.local_ip}:{HTTP_PORT}/{filename}"

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
                    "rtf":              result_data.get("metrics", {}).get("rtf"),
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
