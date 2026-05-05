import sys
import os
import threading
import time
import subprocess
import webbrowser
import pystray
from PIL import Image, ImageDraw
from pystray import MenuItem as item
import redis

from settings_gui import load_settings, open_settings_window
from core.orchestrator import NodeOrchestrator
from core.config_manager import refresh_config
from core.logger import setup_logging

SEMAPHORE_GREEN = True
redis_client = None
orchestrator = None
fish_tts_proc = None
fish_vqgan_proc = None

def init_redis():
    global redis_client
    settings = load_settings()
    try:
        redis_client = redis.Redis(
            host=settings["redis_host"],
            port=settings["redis_port"],
            password=settings["redis_password"] or None,
            decode_responses=True
        )
        redis_client.ping()
        print(f"✅ Connesso a Redis su {settings['redis_host']}:{settings['redis_port']}")
        
        # Recupera lo stato attuale dal semaforo
        current = redis_client.get("aria:gpu:semaphore")
        global SEMAPHORE_GREEN
        if current == "green":
            SEMAPHORE_GREEN = True
        elif current == "red":
            SEMAPHORE_GREEN = False
        else:
            # Default
            SEMAPHORE_GREEN = True
            redis_client.set("aria:gpu:semaphore", "green")
            
    except Exception as e:
        print(f"❌ Errore di connessione a Redis: {e}")
        redis_client = None

def update_redis_semaphore(state: bool):
    global redis_client, orchestrator
    if orchestrator:
        orchestrator.set_semaphore(state)
        
    if redis_client:
        try:
            redis_client.set("aria:gpu:semaphore", "green" if state else "red")
            print(f"Semaforo GPU impostato su {'GREEN' if state else 'RED'}")
        except Exception as e:
            print(f"Errore scrittura semaforo su Redis: {e}")

def generate_icon_image(color="green"):
    """
    Genera un'icona base con un cerchio colorato.
    color: 'green', 'red', 'yellow', 'gray'
    """
    width = 64
    height = 64
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    dc = ImageDraw.Draw(image)
    
    colors = {
        "green": (0, 255, 0, 255),
        "red": (255, 0, 0, 255),
        "yellow": (255, 255, 0, 255),
        "gray": (128, 128, 128, 255)
    }
    
    fill_color = colors.get(color, colors["gray"])
    dc.ellipse((8, 8, 56, 56), fill=fill_color)
    return image

def set_semaphore(icon, state):
    global SEMAPHORE_GREEN
    SEMAPHORE_GREEN = state
    # Aggiorna icona e tooltip
    color = "green" if state else "red"
    gpu_desc = "GPU Disponibile" if state else "GPU Occupata (Gaming)"
    icon.icon = generate_icon_image(color)
    icon.title = f"ARIA Gateway - {gpu_desc} | Cloud Gateway: ATTIVO"
    
    # Aggiorna Redis
    threading.Thread(target=update_redis_semaphore, args=(state,), daemon=True).start()

def menu_action_green(icon, item):
    set_semaphore(icon, True)

def menu_action_red(icon, item):
    set_semaphore(icon, False)

def _open_settings_thread(icon):
    # La GUI customtkinter deve girare nel thread principale se possibile, ma pystray blocca il thread.
    # Faremo girare la GUI in un thread separato o subprocess. Poiché pystray e tkinter nello stesso processo 
    # possono dare problemi su MacOS (ma su Windows di solito no), proviamo in un thread.
    print("Apertura impostazioni...")
    def on_save(s):
        refresh_config()
        init_redis()
    open_settings_window(on_save_callback=on_save)

def menu_action_dashboard(icon, item):
    webbrowser.open("http://localhost:8089")

def menu_action_settings(icon, item):
    threading.Thread(target=_open_settings_thread, args=(icon,), daemon=True).start()

def menu_action_exit(icon, item):
    global orchestrator, fish_tts_proc, fish_vqgan_proc
    if orchestrator:
        orchestrator.stop()
    
    print("Arresto dei server in background...")
    if fish_tts_proc:
        try:
            fish_tts_proc.terminate()
        except Exception:
            pass
    if fish_vqgan_proc:
        try:
            fish_vqgan_proc.terminate()
        except Exception:
            pass
            
    icon.stop()

def setup(icon):
    global orchestrator, fish_tts_proc, fish_vqgan_proc
    icon.visible = True
    # Init redis at startup
    init_redis()
    
    # 1. Init Orchestrator
    if redis_client:
        orchestrator = NodeOrchestrator(redis_client)
        orchestrator.set_semaphore(SEMAPHORE_GREEN)
        orchestrator.start()
    else:
        print("Redis non disponibile all'avvio: Orchestrator non avviato.")
        return

    # 2. Avvio dei Backend Locali (Fish Speech) via Orchestratore
    if "--no-backends" not in sys.argv:
        print("Avvio dei server AI Backend via Orchestratore...")
        
        # Fish TTS (8080)
        if orchestrator.ensure_running("fish-s1-mini"):
            print(" -> Fish TTS Server pronto (Porta 8080)")
        else:
            print(" -> [ERRORE] Impossibile avviare Fish TTS")

        # Fish Voice Cloning VQGAN (8081) - Già gestito come companion o avviabile esplicitamente
        if orchestrator.ensure_running("voice-cloning"):
            print(" -> Fish Voice Cloning Server pronto (Porta 8081)")
        else:
            print(" -> [ERRORE] Impossibile avviare Voice Cloning")
    else:
        print("Flag --no-backends rilevato. I server AI devono essere avviati esternamente o JIT.")
    
    # Correct the initial icon based on redis state
    color = "green" if SEMAPHORE_GREEN else "red"
    icon.icon = generate_icon_image(color)

def create_menu():
    return pystray.Menu(
        item('🟢 GPU Libera (Workflow AI Completo)', menu_action_green, checked=lambda item: SEMAPHORE_GREEN, radio=True),
        item('🔴 GPU Occupata (Solo Cloud Gateway)', menu_action_red, checked=lambda item: not SEMAPHORE_GREEN, radio=True),
        pystray.Menu.SEPARATOR,
        item('🖥️ Apri Dashboard (8089)', menu_action_dashboard),
        pystray.Menu.SEPARATOR,
        item('⚙️ Impostazioni...', menu_action_settings),
        pystray.Menu.SEPARATOR,
        item('❌ Esci', menu_action_exit)
    )

def main():
    setup_logging()
    initial_icon = generate_icon_image("gray")
    
    icon = pystray.Icon(
        "ARIA_Node",
        initial_icon,
        "ARIA Node - Inizializzazione...",
        menu=create_menu()
    )
    
    print("Avvio della Tray Icon del Node Controller di ARIA...")
    icon.run(setup)
    print("Uscita completata.")

if __name__ == '__main__':
    main()
