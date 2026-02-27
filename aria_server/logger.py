import os
import sys
import logging
import logging.handlers
from datetime import datetime
import structlog
from pathlib import Path

# Paths
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Shared context variable definition (if we need to clear context vars across threads we might need contextvars directly)
# but structlog provides clear_contextvars(), bind_contextvars() automatically using contextvars under the hood.

def setup_logging(log_level_name: str = "INFO", console_only: bool = False):
    """
    Configures the global structlog and standard logging settings.
    - Console Output: Colored and human-readable using rich (if available) or ConsoleRenderer.
    - File Output: JSONL, rotated daily, stored in `./logs/`.
    """
    level = getattr(logging, log_level_name.upper(), logging.INFO)
    
    # 1. Clear existing handlers
    logging.root.handlers = []
    
    # 2. Add File Handler (JSONL Format)
    if not console_only:
        log_file = LOGS_DIR / f"aria-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        
        # Processor pipeline for standard logging when going to File Handler
        # Because we'll hook structlog to output to standard logging.
        # However, structlog has a ProcessorFormatter that can format standard logs cleanly into JSON
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=[]
        )
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

    # 3. Add Console Handler (Human Readable & Colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Setup rendering for console
    try:
        import rich
        console_renderer = structlog.dev.ConsoleRenderer(colors=True)
    except ImportError:
        console_renderer = structlog.dev.ConsoleRenderer(colors=False)
        
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=console_renderer,
        foreign_pre_chain=[]
    )
    console_handler.setFormatter(console_formatter)
    logging.root.addHandler(console_handler)

    logging.root.setLevel(level)

    # 4. Configure structlog pipeline
    shared_processors = [
        structlog.contextvars.merge_contextvars,          # Merge bound contextvars (like job_id)
        structlog.stdlib.add_logger_name,                 # Add "logger" field
        structlog.stdlib.add_log_level,                   # Add "level" field
        structlog.stdlib.PositionalArgumentsFormatter(),  # % formatting
        structlog.processors.TimeStamper(fmt="iso"),      # Add "timestamp"
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,             # Unpack exception info
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter, # Let stdlib deal with outputs
    ]

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Returns a structlog logger bound with the specific component name."""
    return structlog.get_logger(name)

# Expose contextvar helpers for easy access
bind_context = structlog.contextvars.bind_contextvars
clear_context = structlog.contextvars.clear_contextvars
unbind_context = structlog.contextvars.unbind_contextvars

# Pre-setup immediately so imports don't explode if setup_logging isn't called explicitly
if not logging.root.handlers:
    # Just basic fallback
    logging.basicConfig(level=logging.INFO)
