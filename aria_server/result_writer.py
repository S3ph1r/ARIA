"""
ARIA Result Writer - Redis Bridge for Result Publishing
"""
import json
import time
import logging
from typing import Optional, Dict, Any, Union
import redis
from redis.exceptions import RedisError

from .config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD,
    RESULT_PREFIX, PROCESSING_PREFIX, RESULT_TTL, PROCESSING_TIMEOUT
)

logger = logging.getLogger(__name__)


class ResultWriter:
    """
    Gestisce la scrittura di risultati su Redis con TTL automatico
    """
    
    def __init__(self):
        self._redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        
        # Test connessione
        try:
            self._redis_client.ping()
            logger.info(f"✅ ResultWriter connesso a Redis: {REDIS_HOST}:{REDIS_PORT}")
        except RedisError as e:
            logger.error(f"❌ ResultWriter connessione fallita: {e}")
            raise
    
    def write_processing(self, job_id: str, worker_id: str) -> bool:
        """
        Segna un task come in elaborazione
        """
        try:
            processing_key = f"{PROCESSING_PREFIX}{job_id}"
            processing_data = {
                "status": "processing",
                "worker_id": worker_id,
                "started_at": time.time(),
                "job_id": job_id
            }
            
            # Salva con TTL
            self._redis_client.setex(
                processing_key, 
                PROCESSING_TIMEOUT,
                json.dumps(processing_data)
            )
            
            logger.info(f"🔄 Task {job_id} segnato come in elaborazione")
            return True
            
        except RedisError as e:
            logger.error(f"❌ Errore durante scrittura processing: {e}")
            return False
    
    def write_result(self, job_id: str, result_data: Dict[str, Any]) -> bool:
        """
        Scrive il risultato su Redis con TTL
        result_data deve contenere:
        - status: 'success' o 'error'
        - audio_base64: stringa base64 del file audio (se success)
        - error_message: stringa errore (se error)
        - processing_time: tempo in secondi
        """
        try:
            result_key = f"{RESULT_PREFIX}{job_id}"
            
            # Prepara risultato
            result = {
                "job_id": job_id,
                "status": result_data.get("status", "unknown"),
                "completed_at": time.time(),
                "processing_time": result_data.get("processing_time", 0),
                "audio_base64": result_data.get("audio_base64", ""),
                "audio_format": result_data.get("audio_format", "wav"),
                "error_message": result_data.get("error_message", ""),
                "metadata": result_data.get("metadata", {})
            }
            
            # Salva con TTL
            self._redis_client.setex(
                result_key,
                RESULT_TTL,
                json.dumps(result)
            )
            
            # Rimuovi processing key
            processing_key = f"{PROCESSING_PREFIX}{job_id}"
            self._redis_client.delete(processing_key)
            
            status = result["status"]
            logger.info(f"✅ Risultato {status} per job {job_id} salvato (TTL: {RESULT_TTL}s)")
            return True
            
        except RedisError as e:
            logger.error(f"❌ Errore durante scrittura risultato: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Errore generico durante scrittura risultato: {e}")
            return False
    
    def write_success(self, job_id: str, audio_base64: str, audio_format: str = "wav", 
                     processing_time: float = 0, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Helper per scrivere risultato di successo
        """
        result_data = {
            "status": "success",
            "audio_base64": audio_base64,
            "audio_format": audio_format,
            "processing_time": processing_time,
            "metadata": metadata or {}
        }
        return self.write_result(job_id, result_data)
    
    def write_error(self, job_id: str, error_message: str, processing_time: float = 0,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Helper per scrivere risultato di errore
        """
        result_data = {
            "status": "error",
            "error_message": error_message,
            "processing_time": processing_time,
            "metadata": metadata or {}
        }
        return self.write_result(job_id, result_data)
    
    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Legge un risultato (per test/debug)
        """
        try:
            result_key = f"{RESULT_PREFIX}{job_id}"
            result_json = self._redis_client.get(result_key)
            
            if result_json:
                return json.loads(result_json)
            return None
            
        except RedisError as e:
            logger.error(f"❌ Errore durante lettura risultato: {e}")
            return None
    
    def is_processing(self, job_id: str) -> bool:
        """
        Verifica se un job è in elaborazione
        """
        try:
            processing_key = f"{PROCESSING_PREFIX}{job_id}"
            return self._redis_client.exists(processing_key) > 0
            
        except RedisError as e:
            logger.error(f"❌ Errore durante verifica processing: {e}")
            return False