import customtkinter as ctk
import json
import os
from pathlib import Path

SETTINGS_FILE = Path("node_settings.json")

DEFAULT_SETTINGS = {
    "redis_host": "192.168.1.120",
    "redis_port": 6379,
    "redis_password": "",
    "samba_path": "Z:\\",
    "autostart": False
}

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

class SettingsWindow(ctk.CTk):
    def __init__(self, on_save_callback=None):
        super().__init__()
        self.title("ARIA Node Controller - Settings")
        self.geometry("400x450")
        self.resizable(False, False)
        
        self.on_save_callback = on_save_callback
        self.settings = load_settings()
        
        # Title
        self.lbl_title = ctk.CTkLabel(self, text="Configurazione ARIA Node", font=("Arial", 20, "bold"))
        self.lbl_title.pack(pady=20)
        
        # Redis Host
        self.frame_redis = ctk.CTkFrame(self)
        self.frame_redis.pack(pady=10, padx=20, fill="x")
        
        self.lbl_redis_host = ctk.CTkLabel(self.frame_redis, text="Redis Host / IP:")
        self.lbl_redis_host.pack(anchor="w", padx=10, pady=(10, 0))
        self.entry_redis_host = ctk.CTkEntry(self.frame_redis, width=300)
        self.entry_redis_host.insert(0, self.settings["redis_host"])
        self.entry_redis_host.pack(padx=10, pady=5)
        
        # Redis Port
        self.lbl_redis_port = ctk.CTkLabel(self.frame_redis, text="Redis Porta:")
        self.lbl_redis_port.pack(anchor="w", padx=10)
        self.entry_redis_port = ctk.CTkEntry(self.frame_redis, width=100)
        self.entry_redis_port.insert(0, str(self.settings["redis_port"]))
        self.entry_redis_port.pack(anchor="w", padx=10, pady=5)
        
        # Samba Path
        self.frame_samba = ctk.CTkFrame(self)
        self.frame_samba.pack(pady=10, padx=20, fill="x")
        
        self.lbl_samba = ctk.CTkLabel(self.frame_samba, text="Percorso Cartella Condivisa (Samba):")
        self.lbl_samba.pack(anchor="w", padx=10, pady=(10, 0))
        self.entry_samba = ctk.CTkEntry(self.frame_samba, width=300)
        self.entry_samba.insert(0, self.settings["samba_path"])
        self.entry_samba.pack(padx=10, pady=5)
        
        # Autostart
        self.frame_options = ctk.CTkFrame(self)
        self.frame_options.pack(pady=10, padx=20, fill="x")
        
        self.var_autostart = ctk.StringVar(value="on" if self.settings["autostart"] else "off")
        self.chk_autostart = ctk.CTkCheckBox(self.frame_options, text="Avvia automaticamente all'avvio di Windows", 
                                             variable=self.var_autostart, onvalue="on", offvalue="off")
        self.chk_autostart.pack(padx=10, pady=10, anchor="w")
        
        # Save Button
        self.btn_save = ctk.CTkButton(self, text="Salva Impostazioni", command=self.save_and_close)
        self.btn_save.pack(pady=20)
        
    def save_and_close(self):
        self.settings["redis_host"] = self.entry_redis_host.get()
        try:
            self.settings["redis_port"] = int(self.entry_redis_port.get())
        except ValueError:
            pass # ignore invalid port for now
            
        self.settings["samba_path"] = self.entry_samba.get()
        self.settings["autostart"] = self.var_autostart.get() == "on"
        
        save_settings(self.settings)
        if self.on_save_callback:
            self.on_save_callback(self.settings)
        self.destroy()

def open_settings_window(on_save_callback=None):
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = SettingsWindow(on_save_callback=on_save_callback)
    app.mainloop()

if __name__ == "__main__":
    open_settings_window()
