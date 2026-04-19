import redis
import json
from pathlib import Path
from aria_node_controller.core.registry_manager import AriaRegistryManager
from aria_node_controller.config import get_config

def final_certification():
    print("🎯 Certificazione Finale Registro Sound Factory...")
    ARIA_ROOT = Path("c:/Users/Roberto/aria")
    config = get_config()
    
    r_client = redis.Redis(
        host=config.redis.host, 
        port=config.redis.port, 
        db=config.redis.db, 
        decode_responses=True
    )
    
    manager = AriaRegistryManager(ARIA_ROOT, r_client, local_ip=config.node.ip)
    manager.publish()
    
    # Verifica immediata
    data = r_client.get("aria:registry:master")
    if data:
        registry = json.loads(data)
        missing = []
        # Elenco degli ID attesi per Uomini in Rosso
        expected = [
            "mus_retro_futuristic_dread", "amb_open_alien", "amb_enclosed_cave",
            "sfx_impact_metal_heavy", "sfx_bio_creature_large", "sfx_bio_creature_swarm",
            "test_earthquake_02", "sting_tension_high", "sting_reveal_shock", "sting_horror_shriek"
        ]
        
        found_count = 0
        all_assets = {}
        for cat in registry["assets"]:
            all_assets.update(registry["assets"][cat])
            
        for eid in expected:
            if eid in all_assets:
                found_count += 1
            else:
                missing.append(eid)
                
        print(f"✅ Registro aggiornato. Asset trovati: {found_count}/10")
        if missing:
            print(f"⚠️ Mancanti nel registro: {missing}")
    else:
        print("❌ Impossibile leggere il registro Redis.")

if __name__ == "__main__":
    final_certification()
