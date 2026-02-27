from settings_gui import load_settings

# We dynamically inject setting values into standard variables useful for the modules
_settings = load_settings()

REDIS_HOST = _settings.get("redis_host", "192.168.1.120")
REDIS_PORT = _settings.get("redis_port", 6379)
REDIS_PASSWORD = _settings.get("redis_password", None)
REDIS_DB = 0
REDIS_TIMEOUT = 2.0
SAMBA_PATH = _settings.get("samba_path", "Z:\\")

def refresh_config():
    global REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, SAMBA_PATH
    _s = load_settings()
    REDIS_HOST = _s.get("redis_host", "192.168.1.120")
    REDIS_PORT = _s.get("redis_port", 6379)
    REDIS_PASSWORD = _s.get("redis_password", None)
    SAMBA_PATH = _s.get("samba_path", "Z:\\")
