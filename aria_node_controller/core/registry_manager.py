import os
import json
from pathlib import Path
import redis
from .logger import get_logger

logger = get_logger("node.registry")

class AriaRegistryManager:
    """
    Manages the 'Master Registry' of ARIA.
    Scans backends_manifest.json and data/assets/ to build a unified catalog.
    Publishes the result to Redis for client discovery.
    """
    REDIS_KEY_MASTER = "aria:registry:master"

    def __init__(self, aria_root: Path, redis_client: redis.Redis):
        self.aria_root = aria_root
        self.redis = redis_client
        self.manifest_path = aria_root / "aria_node_controller" / "config" / "backends_manifest.json"
        # Standard: data/assets/ (nuova gerarchia)
        self.assets_dir = aria_root / "data" / "assets"
        # Supporto legacy: data/voices/ e data/models/ (per compatibilità durante la transizione)
        self.legacy_voices_dir = aria_root / "data" / "voices"
        self.legacy_models_dir = aria_root / "data" / "models"

    def build_registry(self) -> dict:
        """Scans filesystem and manifest to build the full JSON."""
        registry = {
            "node_id": os.getenv("COMPUTERNAME", os.getenv("HOSTNAME", "aria-node-unknown")),
            "status": "online",
            "backends": {},
            "assets": {
                "models": {},
                "voices": {},
                "loras": {},
                "personas": {}
            }
        }

        # 1. Load Backends from Manifest
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    registry["backends"] = manifest.get("backends", {})
            except Exception as e:
                logger.error(f"Failed to read backends_manifest: {e}")

        # 2. Scan Standard Assets (data/assets/{type}/{id})
        if self.assets_dir.exists():
            for asset_type_dir in self.assets_dir.iterdir():
                if not asset_type_dir.is_dir():
                    continue
                
                type_name = asset_type_dir.name
                if type_name not in registry["assets"]:
                    registry["assets"][type_name] = {}

                for asset_id_dir in asset_type_dir.iterdir():
                    if not asset_id_dir.is_dir():
                        continue
                    
                    profile_path = asset_id_dir / "profile.json"
                    if profile_path.exists():
                        try:
                            with open(profile_path, "r", encoding="utf-8") as f:
                                profile = json.load(f)
                                # Aggiunta automatica del path del campione se esiste ref.wav
                                ref_path = asset_id_dir / "ref.wav"
                                if ref_path.exists():
                                    profile["sample_path"] = f"{type_name}/{asset_id_dir.name}/ref.wav"
                                
                                registry["assets"][type_name][asset_id_dir.name] = profile
                        except Exception as e:
                            logger.error(f"Failed to read profile.json in {asset_id_dir.name}: {e}")
                    else:
                        voice_data = {
                            "id": asset_id_dir.name, 
                            "status": "legacy",
                            "note": "Profilo manuale mancante"
                        }
                        if (asset_id_dir / "ref.wav").exists():
                            voice_data["sample_path"] = f"{type_name}/{asset_id_dir.name}/ref.wav"
                        registry["assets"][type_name][asset_id_dir.name] = voice_data

        # 3. Scan Legacy Voices (se non già presenti negli assets standard)
        if self.legacy_voices_dir.exists():
            for voice_dir in self.legacy_voices_dir.iterdir():
                if not voice_dir.is_dir() or voice_dir.name in registry["assets"]["voices"]:
                    continue
                
                # Cerca profile.json anche qui (Pragmatic approach)
                profile_path = voice_dir / "profile.json"
                if profile_path.exists():
                    try:
                        with open(profile_path, "r", encoding="utf-8") as f:
                            profile = json.load(f)
                            profile["status"] = profile.get("status", "legacy")
                            if (voice_dir / "ref.wav").exists():
                                profile["sample_path"] = f"legacy_voices/{voice_dir.name}/ref.wav"
                            registry["assets"]["voices"][voice_dir.name] = profile
                    except Exception as e:
                        logger.error(f"Failed to read profile.json in legacy voice {voice_dir.name}: {e}")
                else:
                    voice_data = {
                        "id": voice_dir.name,
                        "status": "legacy",
                        "note": "Trovata in data/voices/ (senza profilo)"
                    }
                    if (voice_dir / "ref.wav").exists():
                        voice_data["sample_path"] = f"legacy_voices/{voice_dir.name}/ref.wav"
                    registry["assets"]["voices"][voice_dir.name] = voice_data

        # 4. Scan Legacy Models (se non già presenti negli assets standard)
        if self.legacy_models_dir.exists():
            for model_dir in self.legacy_models_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name in registry["assets"]["models"]:
                    continue
                
                profile_path = model_dir / "profile.json"
                if profile_path.exists():
                    try:
                        with open(profile_path, "r", encoding="utf-8") as f:
                            profile = json.load(f)
                            profile["status"] = profile.get("status", "legacy")
                            registry["assets"]["models"][model_dir.name] = profile
                    except Exception as e:
                        logger.error(f"Failed to read profile.json in legacy model {model_dir.name}: {e}")
                else:
                    registry["assets"]["models"][model_dir.name] = {
                        "id": model_dir.name,
                        "status": "legacy",
                        "note": "Trovata in data/models/ (senza profilo)"
                    }

        return registry

    def publish(self):
        """Builds and pushes the registry to Redis."""
        try:
            registry = self.build_registry()
            self.redis.set(self.REDIS_KEY_MASTER, json.dumps(registry))
            # TTL di 7 giorni per sicurezza (ma viene aggiornato ad ogni avvio)
            self.redis.expire(self.REDIS_KEY_MASTER, 7 * 86400)
            logger.info(f"Master Registry pubblicato su Redis: {self.REDIS_KEY_MASTER}")
        except Exception as e:
            logger.error(f"Errore critico durante la pubblicazione del Master Registry: {e}")

if __name__ == "__main__":
    # Test veloce se eseguito direttamente (richiede Redis attivo)
    # python -m aria_node_controller.core.registry_manager
    pass
