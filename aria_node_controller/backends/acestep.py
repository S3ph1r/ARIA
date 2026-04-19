"""
ARIA — ACE-Step 1.5 XL Connector (Livello 0)
==============================================
Connettore HTTP tra l'Orchestratore ARIA e il wrapper CLI persistente
(backends/acestep/aria_wrapper_server.py, porta 8084).

Pattern identico a Qwen3TTSBackend:
  - POST /generate (JSON payload) → risposta sincrona con path output
  - Il server wrapper gestisce TOML + subprocess cli.py internamente
"""

import uuid
import shutil
import logging
import subprocess
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger("node.backend.acestep")

ACESTEP_WRAPPER_URL = "http://127.0.0.1:8084"


class ACEStepBackend:
    """
    Backend musicale per ACE-Step 1.5 XL (SFT).
    Implementa l'interfaccia usata da NodeOrchestrator.
    """

    model_id   = "acestep-1.5-xl-sft"
    model_type = "mus"

    def load(self, model_path: str, config: dict) -> None:
        """Verifica che il wrapper server sia raggiungibile."""
        try:
            r = requests.get(f"{ACESTEP_WRAPPER_URL}/health", timeout=5)
            r.raise_for_status()
            info = r.json()
            logger.info(
                f"ACE-Step Wrapper OK | cli.py={info.get('cli_py')} | "
                f"ready={info.get('ready')}"
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"ACE-Step wrapper non raggiungibile su {ACESTEP_WRAPPER_URL}. "
                "Il processo deve essere avviato dall'Orchestratore JIT."
            )
        except Exception as e:
            raise RuntimeError(f"Health check ACE-Step fallito: {e}")

    def unload(self) -> None:
        """No-op — il wrapper è un processo esterno gestito dall'Orchestratore."""
        logger.info("ACE-Step backend: unload (no-op, processo esterno)")

    def is_loaded(self) -> bool:
        """Controlla se il wrapper risponde al health check."""
        try:
            r = requests.get(f"{ACESTEP_WRAPPER_URL}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # HTDemucs — stem separation (GPU, eseguito dopo la generazione del master)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_htdemucs(
        self,
        master_path: Path,
        stems_dir: Path,
        aria_root: Path,
    ) -> Optional[dict]:
        """
        Separa il master WAV in stem via HTDemucs (modello htdemucs_6s).
        Ritorna un dict {stem_name: Path} con i file prodotti, o None su errore.

        Stem prodotti: bass, drums, other  (vocals omesso — PAD è strumentale).
        HTDemucs viene eseguito con il Python dell'env dias-sound-engine che
        include torch + demucs. Output in stems_dir/{stem_name}.wav.
        """
        python_exe = aria_root / "envs" / "dias-sound-engine" / "python.exe"
        if not python_exe.exists():
            import sys
            python_exe = Path(sys.executable)

        stems_dir.mkdir(parents=True, exist_ok=True)

        # HTDemucs scrive in: {stems_dir}/htdemucs_6s/{master_name}/{stem}.wav
        # Non usiamo --filename: non disponibile in tutte le versioni di demucs.
        # I file vengono trovati e spostati in posizione flat dal codice sotto.
        cmd = [
            str(python_exe), "-m", "demucs",
            "-n", "htdemucs_6s",
            "-o", str(stems_dir),
            str(master_path),
        ]

        logger.info(f"[ACE-Step] HTDemucs: separazione stem per {master_path.name}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,   # 30 min max per tracce molto lunghe
            )
        except subprocess.TimeoutExpired:
            logger.error("[ACE-Step] HTDemucs: timeout (>30min)")
            return None

        if result.returncode != 0:
            logger.error(f"[ACE-Step] HTDemucs fallito (rc={result.returncode}): {result.stderr[-500:]}")
            return None

        # Demucs scrive in: {stems_dir}/htdemucs_6s/{master_stem_name}/{stem}.wav
        # Spostiamo i file in posizione flat: {stems_dir}/{stem}.wav
        wanted = ["bass", "drums", "other"]
        stem_paths: dict = {}
        nested_base = stems_dir / "htdemucs_6s" / master_path.stem

        for stem_name in wanted:
            flat = stems_dir / f"{stem_name}.wav"
            if flat.exists():
                stem_paths[stem_name] = flat
                continue
            nested = nested_base / f"{stem_name}.wav"
            if nested.exists():
                shutil.move(str(nested), str(flat))
                stem_paths[stem_name] = flat
            else:
                logger.warning(f"[ACE-Step] HTDemucs: stem '{stem_name}' non trovato in {nested_base}")

        logger.info(f"[ACE-Step] HTDemucs OK: {list(stem_paths.keys())}")
        return stem_paths if stem_paths else None

    def run(self, payload: dict, aria_root: Path, local_ip: str) -> dict:
        """
        Invia un task musicale al wrapper CLI e aspetta il risultato.

        Parametri attesi in payload (da Redis/DIAS):
          - prompt         : descrizione semantica del suono (obbligatorio)
          - duration       : durata in secondi (default 30.0)
          - seed           : seed per riproducibilità (default 42, -1 = casuale)
          - guidance_scale : CFG scale del DiT (default 7.0)
          - output_style   : 'pad', 'amb', 'sfx', 'sting' (default 'pad')
          - lyrics         : testo vocale (opzionale, default = strumentale)
        """
        prompt = payload.get("prompt") or payload.get("text", "")
        if not prompt:
            raise ValueError("Campo 'prompt' obbligatorio per ACE-Step")

        job_id      = payload.get("job_id") or str(uuid.uuid4())
        style       = payload.get("output_style", "pad")
        seed        = payload.get("seed", 42)
        run_demucs  = bool(payload.get("run_demucs", False))

        request_body = {
            "job_id":          job_id,
            "prompt":          prompt,
            "lyrics":          payload.get("lyrics", ""),
            "duration":        float(payload.get("duration", 30.0)),
            "seed":            int(seed),
            "guidance_scale":  float(payload.get("guidance_scale", 7.0)),
            "inference_steps": int(payload.get("inference_steps", 60)),
            "output_style":    style,
            "audio_format":    "wav",
            "thinking":               bool(payload.get("thinking", True)),
            "use_constrained_decoding": True,
            "offload_to_cpu":         True,
            "offload_dit_to_cpu":     True,
            "backend":                "pt",
        }

        timeout = payload.get("timeout_seconds", 7200)

        logger.info(
            f"[ACE-Step] Submit | job={job_id} | style={style} | "
            f"duration={request_body['duration']}s | seed={seed} | demucs={run_demucs}"
        )

        try:
            response = requests.post(
                f"{ACESTEP_WRAPPER_URL}/generate",
                json=request_body,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout ({timeout}s) durante generazione ACE-Step — job={job_id}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Errore chiamata ACE-Step wrapper: {e}")

        if result.get("status") != "completed":
            err = result.get("error", "Errore sconosciuto")
            raise RuntimeError(f"ACE-Step generation failed: {err}")

        # ── Costruisce URL pubblica via Asset Server (porta 8082) ─────────────
        audio_path = result.get("audio_path", "")

        def _to_url(abs_path: str) -> str:
            try:
                rel = Path(abs_path).relative_to(aria_root / "data")
                return f"http://{local_ip}:8082/{rel.as_posix()}"
            except ValueError:
                return f"http://{local_ip}:8082/assets/sound_library/{style}/{job_id}/{Path(abs_path).name}"

        audio_url  = _to_url(audio_path)
        score_path = result.get("score_path", "")
        score_url  = _to_url(score_path) if score_path else ""

        logger.info(
            f"[ACE-Step] Generazione OK | job={job_id} | "
            f"audio={Path(audio_path).name} | score={'sì' if score_path else 'no'}"
        )

        out = {
            "audio_url":        audio_url,
            "score_url":        score_url,
            "local_path":       audio_path,
            "score_path":       score_path,
            "output_style":     style,
            "duration_seconds": result.get("duration_seconds", request_body["duration"]),
            "status":           "success",
        }

        # ── HTDemucs stem separation (solo PAD, GPU lato ARIA) ───────────────
        if run_demucs and audio_path:
            master   = Path(audio_path)
            stems_dir = master.parent / "stems"
            stem_paths = self._run_htdemucs(master, stems_dir, aria_root)

            if stem_paths:
                out["stems"] = {
                    name: _to_url(str(path))
                    for name, path in stem_paths.items()
                }
                logger.info(f"[ACE-Step] Stem pronti: {list(out['stems'].keys())}")
            else:
                logger.warning(f"[ACE-Step] HTDemucs fallito per {job_id} — stem non disponibili")

        return out
