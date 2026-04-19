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

    def __init__(self, aria_root: Path, redis_client: redis.Redis, local_ip: str = "127.0.0.1"):
        self.aria_root = aria_root
        self.redis = redis_client
        self.local_ip = local_ip
        self.manifest_path = aria_root / "aria_node_controller" / "config" / "backends_manifest.json"
        # Root degli assets: data/assets/
        self.assets_base_dir = aria_root / "data" / "assets"

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
                "personas": {},
                "pad": {},
                "amb": {},
                "sfx": {},
                "sting": {}
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

        # 2. Scan Assets (data/assets/)
        if self.assets_base_dir.exists():
            # Scansioniamo tutte le top-level directory in data/assets
            # ESCLUDIAMO 'models' perché contiene i pesi e non asset consumabili
            EXCLUDED_DIRS = ["models"]
            
            for top_dir in self.assets_base_dir.iterdir():
                if not top_dir.is_dir() or top_dir.name in EXCLUDED_DIRS:
                    continue
                
                # Se è 'sound_library', scendiamo di un livello per le categorie (pad, sfx, etc.)
                dirs_to_scan = []
                if top_dir.name == "sound_library":
                    dirs_to_scan = [d for d in top_dir.iterdir() if d.is_dir()]
                else:
                    dirs_to_scan = [top_dir]

                for category_dir in dirs_to_scan:
                    type_name = category_dir.name
                    if type_name not in registry["assets"]:
                        registry["assets"][type_name] = {}

                    for asset_id_dir in category_dir.iterdir():
                        if not asset_id_dir.is_dir(): continue
                        
                        profile_path = asset_id_dir / "profile.json"
                        # Calcolo URL relativo per la porta 8082
                        # Se è dentro sound_library, il path HTTP include 'sound_library'
                        is_sl = "sound_library" in str(category_dir.relative_to(self.assets_base_dir))
                        http_subpath = f"sound_library/{type_name}" if is_sl else type_name

                        if profile_path.exists():
                            try:
                                with open(profile_path, "r", encoding="utf-8") as f:
                                    profile = json.load(f)
                                    if "description" not in profile and "prompt" in profile:
                                        profile["description"] = profile["prompt"]

                                    asset_wav = asset_id_dir / f"{asset_id_dir.name}.wav"
                                    if not asset_wav.exists():
                                        wavs = list(asset_id_dir.glob("*.wav"))
                                        if wavs: asset_wav = wavs[0]
                                        else: asset_wav = None

                                    if asset_wav and asset_wav.exists():
                                        profile["sample_url"] = f"http://{self.local_ip}:8082/assets/{http_subpath}/{asset_id_dir.name}/{asset_wav.name}"
                                    
                                    registry["assets"][type_name][asset_id_dir.name] = profile
                            except Exception as e:
                                logger.error(f"Failed to read profile.json in {asset_id_dir.name}: {e}")
                        else:
                            # Legacy fallback (voci e vecchi asset senza profile.json)
                            voice_data = {"id": asset_id_dir.name, "status": "legacy"}
                            wavs = list(asset_id_dir.glob("*.wav"))
                            if wavs:
                                voice_data["sample_url"] = f"http://{self.local_ip}:8082/assets/{http_subpath}/{asset_id_dir.name}/{wavs[0].name}"
                            registry["assets"][type_name][asset_id_dir.name] = voice_data

        return registry

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
    # Invocazione manuale del registro: python -m aria_node_controller.core.registry_manager
    import sys
    from pathlib import Path
    
    # Risaliamo alla root del progetto
    ARIA_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ARIA_ROOT))
    
    # Caricamento configurazioni ufficiali
    # Nota: Usiamo try/except per fallback manuale se la GUI/settings non sono disponibili
    try:
        from aria_node_controller.settings_gui import load_settings
        settings = load_settings()
        r_host = settings.get("redis_host", "127.0.0.1")
        r_port = settings.get("redis_port", 6379)
        r_pass = settings.get("redis_password", None)
        node_ip = settings.get("local_ip", "127.0.0.1")
    except Exception:
        # Fallback a valori di default se i settings falliscono
        r_host = "127.0.0.1"
        r_port = 6379
        r_pass = None
        node_ip = "127.0.0.1"

    print(f"[*] Inizializzazione Registro Master ARIA...")
    print(f"[*] Connessione a Redis: {r_host}:{r_port}")
    
    try:
        r_client = redis.Redis(host=r_host, port=r_port, password=r_pass, decode_responses=True)
        r_client.ping()
        
        manager = AriaRegistryManager(ARIA_ROOT, r_client, local_ip=node_ip)
        print(f"[*] Scansione Warehouse in corso: {manager.assets_base_dir}...")
        manager.publish()
        print(f"[✅] Sincronizzazione completata con successo.")
        
    except Exception as e:
        print(f"[❌] Errore durante la sincronizzazione: {e}")
