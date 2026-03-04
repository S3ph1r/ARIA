import sys
import os
import threading
import time
import redis
from aria_node_controller.core.orchestrator import NodeOrchestrator
from aria_node_controller.settings_gui import load_settings

def init_redis():
    settings = load_settings()
    try:
        r = redis.Redis(
            host=settings["redis_host"],
            port=settings["redis_port"],
            password=settings["redis_password"] or None,
            decode_responses=True
        )
        r.ping()
        print(f"[*] Connesso a Redis su {settings['redis_host']}:{settings['redis_port']}")
        return r
    except Exception as e:
        print(f"[!] Errore di connessione a Redis: {e}")
        return None

def main():
    print("=== ARIA Node Controller (CLI) ===")
    redis_client = init_redis()
    if not redis_client:
        sys.exit(1)
        
    orchestrator = NodeOrchestrator(redis_client)
    
    # Check if semaphore is already set to red, otherwise default to green
    current_sem = redis_client.get("aria:gpu:semaphore")
    if current_sem == "red":
        print("[*] Semaforo inizialmente ROSSO")
        orchestrator.set_semaphore(False)
    else:
        print("[*] Semaforo inizialmente VERDE")
        orchestrator.set_semaphore(True)
        
    print("[*] Avvio Orchestratore...")
    orchestrator.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[*] Arresto in corso...")
        orchestrator.stop()

if __name__ == "__main__":
    main()
