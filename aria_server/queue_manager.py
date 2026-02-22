"""
ARIA Queue Manager - Redis Bridge for Task Consumption
"""
import json
import time
import logging
from typing import Optional, Dict, Any, List
import redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from .config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD,
    QUEUE_PREFIX, MODEL_QUEUES, QUEUE_TIMEOUT, 
    REQUIRED_TASK_FIELDS, MAX_RETRIES, RETRY_BACKOFF
)

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Gestisce la lettura di task da Redis code con reconnect automatico
    """
    
    def __init__(self):
        self._redis_client = None
        self._connect()
    
    def _connect(self) -> None:
        """Connette a Redis con retry automatico"""
        for attempt in range(MAX_RETRIES):
            try:
                self._redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=10,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                # Test connessione
                self._redis_client.ping()
                logger.info(f"✅ Connesso a Redis: {REDIS_HOST}:{REDIS_PORT}")
                return
                
            except (ConnectionError, TimeoutError, RedisError) as e:
                wait_time = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"❌ Redis connessione fallita (tentativo {attempt + 1}): {e}")
                logger.info(f"🔄 Riprovo tra {wait_time}s...")
                time.sleep(wait_time)
        
        raise ConnectionError(f"❌ Impossibile connettersi a Redis dopo {MAX_RETRIES} tentativi")
    
    def next_task(self) -> Optional[Dict[str, Any]]:
        """
        Legge il prossimo task dalle code con BRPOP
        Ritorna None se nessun task disponibile
        """
        if not self._redis_client:
            self._connect()
        
        # Costruisci lista di code da controllare
        queues = []
        for model_type, models in MODEL_QUEUES.items():
            for model_id, queue_name in models.items():
                queues.append(queue_name)
        
        try:
            # BRPOP con timeout per non bloccare indefinitivamente
            result = self._redis_client.brpop(queues, timeout=QUEUE_TIMEOUT)
            
            if result:
                queue_name, task_json = result
                task = json.loads(task_json)
                
                # Valida task
                if self._validate_task(task):
                    logger.info(f"📥 Task ricevuto da {queue_name}: {task.get('job_id', 'unknown')}")
                    return task
                else:
                    logger.warning(f"⚠️ Task invalido scartato: {task}")
                    return None
            
            return None  # Nessun task disponibile
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"❌ Redis errore durante lettura task: {e}")
            self._connect()  # Riconnetti
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            return None
    
    def queue_lengths(self) -> Dict[str, int]:
        """
        Ritorna la lunghezza di tutte le code
        """
        lengths = {}
        try:
            for model_type, models in MODEL_QUEUES.items():
                for model_id, queue_name in models.items():
                    length = self._redis_client.llen(queue_name)
                    lengths[f"{model_type}:{model_id}"] = length
            
            return lengths
            
        except RedisError as e:
            logger.error(f"❌ Errore durante lettura lunghezze code: {e}")
            return {}
    
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Valida che il task abbia tutti i campi richiesti"""
        if not isinstance(task, dict):
            return False
        
        for field in REQUIRED_TASK_FIELDS:
            if field not in task:
                logger.warning(f"Campo mancante: {field}")
                return False
        
        # Valida campi specifici
        try:
            model_type = task.get("model_type")
            model_id = task.get("model_id")
            
            if model_type not in MODEL_QUEUES:
                logger.warning(f"Model type non supportato: {model_type}")
                return False
            
            if model_id not in MODEL_QUEUES[model_type]:
                logger.warning(f"Model ID non supportato: {model_id}")
                return False
            
            # Verifica timestamp (task non troppo vecchi)
            queued_at = task.get("queued_at", 0)
            current_time = time.time()
            timeout_seconds = task.get("timeout_seconds", 3600)
            
            if current_time - queued_at > timeout_seconds:
                logger.warning(f"Task scaduto: {current_time - queued_at}s > {timeout_seconds}s")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Errore validazione task: {e}")
            return False
    
    def health_check(self) -> bool:
        """Verifica che Redis sia raggiungibile"""
        try:
            return self._redis_client.ping()
        except:
            return False