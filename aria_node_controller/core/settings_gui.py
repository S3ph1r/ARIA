import customtkinter as ctk
import json
import os
from pathlib import Path

SETTINGS_FILE = Path("node_settings.json")

DEFAULT_SETTINGS = {
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_password": "",
    "node_ip": "",  # Vuoto = auto-detect IP dalla scheda di rete
    "autostart": False,
    "minio_endpoint": "localhost:9000",
    "minio_access_key": "minioadmin",
    "minio_secret_key": "minioadmin"
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
        self.geometry("450x650")
        self.resizable(False, False)
        
        self.on_save_callback = on_save_callback
        self.settings = load_settings()
        
        # Scrollable Frame per contenere tutto senza affollare
        self.scroll_frame = ctk.CTkScrollableFrame(self, width=420, height=600)
        self.scroll_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Title
        self.lbl_title = ctk.CTkLabel(self.scroll_frame, text="Configurazione ARIA Node", font=("Arial", 20, "bold"))
        self.lbl_title.pack(pady=15)
        
        # --- REDIS SECTION ---
        self.frame_redis = ctk.CTkFrame(self.scroll_frame)
        self.frame_redis.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(self.frame_redis, text="REDIS (Orchestrazione)", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.lbl_redis_host = ctk.CTkLabel(self.frame_redis, text="Host / IP:")
        self.lbl_redis_host.pack(anchor="w", padx=10)
        self.entry_redis_host = ctk.CTkEntry(self.frame_redis, width=350)
        self.entry_redis_host.insert(0, self.settings["redis_host"])
        self.entry_redis_host.pack(padx=10, pady=5)
        
        self.lbl_redis_port = ctk.CTkLabel(self.frame_redis, text="Porta:")
        self.lbl_redis_port.pack(anchor="w", padx=10)
        self.entry_redis_port = ctk.CTkEntry(self.frame_redis, width=100)
        self.entry_redis_port.insert(0, str(self.settings["redis_port"]))
        self.entry_redis_port.pack(anchor="w", padx=10, pady=5)
        
        # --- MINIO SECTION ---
        self.frame_minio = ctk.CTkFrame(self.scroll_frame)
        self.frame_minio.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(self.frame_minio, text="MINIO (Storage)", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.lbl_minio_end = ctk.CTkLabel(self.frame_minio, text="Endpoint (IP:Porta):")
        self.lbl_minio_end.pack(anchor="w", padx=10)
        self.entry_minio_end = ctk.CTkEntry(self.frame_minio, width=350)
        self.entry_minio_end.insert(0, self.settings.get("minio_endpoint", "localhost:9000"))
        self.entry_minio_end.pack(padx=10, pady=5)
        
        self.lbl_minio_acc = ctk.CTkLabel(self.frame_minio, text="Access Key:")
        self.lbl_minio_acc.pack(anchor="w", padx=10)
        self.entry_minio_acc = ctk.CTkEntry(self.frame_minio, width=350)
        self.entry_minio_acc.insert(0, self.settings.get("minio_access_key", "minioadmin"))
        self.entry_minio_acc.pack(padx=10, pady=5)
        
        self.lbl_minio_sec = ctk.CTkLabel(self.frame_minio, text="Secret Key:")
        self.lbl_minio_sec.pack(anchor="w", padx=10)
        self.entry_minio_sec = ctk.CTkEntry(self.frame_minio, width=350, show="*")
        self.entry_minio_sec.insert(0, self.settings.get("minio_secret_key", "minioadmin"))
        self.entry_minio_sec.pack(padx=10, pady=5)
        
        # --- NODE SECTION ---
        self.frame_node = ctk.CTkFrame(self.scroll_frame)
        self.frame_node.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(self.frame_node, text="NODO ARIA", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.lbl_node_ip = ctk.CTkLabel(self.frame_node, text="IP Locale (vuoto = auto):")
        self.lbl_node_ip.pack(anchor="w", padx=10)
        self.entry_node_ip = ctk.CTkEntry(self.frame_node, width=350, placeholder_text="Auto-detect")
        self.entry_node_ip.insert(0, self.settings.get("node_ip", ""))
        self.entry_node_ip.pack(padx=10, pady=5)
        
        # Autostart
        self.var_autostart = ctk.StringVar(value="on" if self.settings["autostart"] else "off")
        self.chk_autostart = ctk.CTkCheckBox(self.scroll_frame, text="Avvia automaticamente con Windows", 
                                             variable=self.var_autostart, onvalue="on", offvalue="off")
        self.chk_autostart.pack(padx=30, pady=10, anchor="w")
        
        # Save Button
        self.btn_save = ctk.CTkButton(self.scroll_frame, text="Salva Configurazione", command=self.save_and_close, height=40, font=("Arial", 14, "bold"))
        self.btn_save.pack(pady=20)
        
    def save_and_close(self):
        self.settings["redis_host"] = self.entry_redis_host.get()
        try:
            self.settings["redis_port"] = int(self.entry_redis_port.get())
        except ValueError: pass
            
        self.settings["minio_endpoint"] = self.entry_minio_end.get().strip()
        self.settings["minio_access_key"] = self.entry_minio_acc.get().strip()
        self.settings["minio_secret_key"] = self.entry_minio_sec.get().strip()
        
        self.settings["node_ip"] = self.entry_node_ip.get().strip()
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
