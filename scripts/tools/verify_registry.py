import redis
import json

# Puntiamo all'IP corretto del server Redis (LXC 120)
REDIS_HOST = '192.168.1.120'

try:
    r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    data = r.get('aria:registry:master')

    if not data:
        print(f"ERRORE: Registro 'aria:registry:master' non trovato su {REDIS_HOST}")
    else:
        registry = json.loads(data)
        assets = registry.get("assets", {})
        
        print(f"--- VERIFICA REGISTRO ARIA (Host: {REDIS_HOST}) ---")
        print(f"Node ID: {registry.get('node_id')}")
        
        print("\nCategorie Asset (Sound Library):")
        for cat in ['pad', 'amb', 'sfx', 'sting']:
            inventory = assets.get(cat, {})
            print(f" - {cat}: {len(inventory)} asset")
        
        print("\nCategorie Speciali:")
        for cat in ['voices', 'personas', 'models']:
            items = list(assets.get(cat, {}).keys())
            print(f" - {cat}: {len(items)} trovati")
            if items:
                print(f"   Esempi: {items[:5]}")

        if 'voices' in assets and len(assets['voices']) > 0:
            print("\n[OK] Voci rilevate e pronte per Stage D.")
        else:
            print("\n[WARNING] Nessuna voce trovata nel registro!")
            
except Exception as e:
    print(f"ERRORE di connessione a {REDIS_HOST}: {e}")
