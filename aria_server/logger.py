"""
ARIA Server - Structured Logging System
Fase AS-3: Logging con console colorata + JSON strutturato
"""

import logging
import logging.handlers
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
import threading

# Context variables per propagazione automatica
job_id_var: ContextVar[Optional[str]] = ContextVar('job_id', default=None)
model_id_var: ContextVar[Optional[str]] = ContextVar('model_id', default=None)

class JSONFormatter(logging.Formatter):
    """Formatter JSON strutturato per file di log"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Campi base
        log_entry = {
            'ts': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'component': record.name,
            'message': record.getMessage(),
            'thread': record.threadName,
            'file': f"{record.filename}:{record.lineno}"
        }
        
        # Context propagation
        job_id = job_id_var.get()
        model_id = model_id_var.get()
        
        if job_id:
            log_entry['job_id'] = job_id
        if model_id:
            log_entry['model_id'] = model_id
            
        # Extra fields from record
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        if hasattr(record, 'error'):
            log_entry['error'] = str(record.error)
        if hasattr(record, 'event'):
            log_entry['event'] = record.event
            
        # Add any other extra attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'getMessage', 'message'] and not key.startswith('_'):
                if key not in log_entry:
                    log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)

class ColoredFormatter(logging.Formatter):
    """Formatter con colori per console"""
    
    # Colori ANSI
    COLORS = {
        'DEBUG': '\033[90m',    # Grigio
        'INFO': '\033[0m',      # Bianco (default)
        'WARNING': '\033[93m',  # Giallo
        'ERROR': '\033[91m',    # Rosso
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        # Colore per il livello
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Componente (abbreviata)
        component = record.name.replace('aria_server.', 'aria.')
        if len(component) > 15:
            component = component[:12] + '...'
        
        # Context
        context_parts = []
        job_id = job_id_var.get()
        model_id = model_id_var.get()
        
        if job_id:
            context_parts.append(f"job_id={job_id}")
        if model_id:
            context_parts.append(f"model={model_id}")
            
        context_str = ' ' + ' '.join(context_parts) if context_parts else ''
        
        # Messaggio
        message = record.getMessage()
        
        # Formato finale
        if record.levelno >= logging.WARNING:
            # Per warning/error: più visibile
            level_color = self.BOLD + color
            return f"{color}[{timestamp}] {level_color}{record.levelname:8}{reset}{color} [{component:15}]{context_str} {message}{reset}"
        else:
            # Per info/debug: più pulito
            return f"{color}[{timestamp}] [{component:15}]{context_str} {message}{reset}"

def setup_logging(log_level: str = "INFO", log_dir: str = "/app/logs") -> None:
    """Setup completo del sistema di logging"""
    
    # Crea directory logs se non esiste
    os.makedirs(log_dir, exist_ok=True)
    
    # Logger root
    logger = logging.getLogger('aria_server')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Evita propagazione a logger root di Python
    logger.propagate = False
    
    # Handler Console (colorato)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Handler File JSON con rotazione giornaliera
    json_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'aria.jsonl'),
        when='midnight',
        interval=1,
        backupCount=7,  # Mantiene 7 giorni
        encoding='utf-8'
    )
    json_handler.setLevel(logging.DEBUG)  # Logga tutto su file
    json_formatter = JSONFormatter()
    json_handler.setFormatter(json_formatter)
    logger.addHandler(json_handler)
    
    # Log iniziale
    logger.info("🚀 ARIA Logging System avviato", extra={
        'event': 'logging_initialized',
        'log_level': log_level,
        'log_dir': log_dir
    })

def get_logger(name: str) -> logging.Logger:
    """Ottieni logger per un componente specifico"""
    return logging.getLogger(f'aria_server.{name}')

def set_log_context(job_id: Optional[str] = None, model_id: Optional[str] = None) -> None:
    """Imposta context per propagazione automatica nei log"""
    if job_id is not None:
        job_id_var.set(job_id)
    if model_id is not None:
        model_id_var.set(model_id)

def clear_log_context() -> None:
    """Pulisci context corrente"""
    job_id_var.set(None)
    model_id_var.set(None)

# Context manager per logging con context
class log_context:
    """Context manager per logging con context automatico"""
    
    def __init__(self, job_id: Optional[str] = None, model_id: Optional[str] = None):
        self.job_id = job_id
        self.model_id = model_id
        self._tokens = []
    
    def __enter__(self):
        if self.job_id:
            self._tokens.append(job_id_var.set(self.job_id))
        if self.model_id:
            self._tokens.append(model_id_var.set(self.model_id))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ripristina context precedente
        for token in reversed(self._tokens):
            try:
                job_id_var.reset(token) if 'job_id' in str(token) else model_id_var.reset(token)
            except:
                pass
        self._tokens.clear()

# Inizializza logging all'import se ARIA_LOG_LEVEL è definito
if os.getenv('ARIA_LOG_LEVEL'):
    setup_logging(os.getenv('ARIA_LOG_LEVEL'))