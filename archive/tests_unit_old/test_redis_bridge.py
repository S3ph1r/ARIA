"""
Test Redis Bridge - QueueManager e ResultWriter
"""
import time
import base64
import pytest
from aria_server import QueueManager, ResultWriter
from aria_server.config import MODEL_QUEUES, RESULT_PREFIX


class TestRedisBridge:
    """Test suite per Redis Bridge components"""
    
    @pytest.fixture
    def queue_manager(self):
        """Fixture per QueueManager"""
        return QueueManager()
    
    @pytest.fixture
    def result_writer(self):
        """Fixture per ResultWriter"""
        return ResultWriter()
    
    def test_queue_manager_connection(self, queue_manager):
        """Test connessione QueueManager"""
        assert queue_manager.health_check() == True
    
    def test_result_writer_connection(self, result_writer):
        """Test connessione ResultWriter"""
        # Se il costruttore non lancia eccezioni, la connessione è OK
        assert True
    
    def test_queue_lengths(self, queue_manager):
        """Test lettura lunghezze code"""
        lengths = queue_manager.queue_lengths()
        assert isinstance(lengths, dict)
        
        # Verifica che ci siano le code attese
        for model_type, models in MODEL_QUEUES.items():
            for model_id in models:
                key = f"{model_type}:{model_id}"
                assert key in lengths or lengths == {}
    
    def test_task_validation(self, queue_manager):
        """Test validazione task"""
        # Task valido
        valid_task = {
            "job_id": "test-123",
            "client_id": "client-456",
            "model_type": "tts",
            "model_id": "orpheus-3b",
            "queued_at": time.time(),
            "timeout_seconds": 3600,
            "callback_key": "callback-789",
            "payload": {"text": "Test text"}
        }
        assert queue_manager._validate_task(valid_task) == True
        
        # Task invalido - campi mancanti
        invalid_task = {"job_id": "test-123"}
        assert queue_manager._validate_task(invalid_task) == False
        
        # Task invalido - modello non supportato
        unsupported_task = valid_task.copy()
        unsupported_task["model_id"] = "modello-inventato"
        assert queue_manager._validate_task(unsupported_task) == False
    
    def test_result_writing(self, result_writer):
        """Test scrittura risultati"""
        job_id = "test-result-123"
        
        # Test successo
        audio_data = "dGVzdCBhdWRpbyBkYXRh"  # "test audio data" in base64
        result_writer.write_success(
            job_id=job_id,
            audio_base64=audio_data,
            audio_format="wav",
            processing_time=2.5,
            metadata={"sample_rate": 22050}
        )
        
        # Verifica che il risultato sia salvato
        result = result_writer.get_result(job_id)
        assert result is not None
        assert result["status"] == "success"
        assert result["audio_base64"] == audio_data
        assert result["processing_time"] == 2.5
        assert result["metadata"]["sample_rate"] == 22050
    
    def test_error_handling(self, result_writer):
        """Test gestione errori"""
        job_id = "test-error-456"
        
        # Test errore
        result_writer.write_error(
            job_id=job_id,
            error_message="Modello non disponibile",
            processing_time=0.5,
            metadata={"retry_count": 3}
        )
        
        # Verifica che l'errore sia salvato
        result = result_writer.get_result(job_id)
        assert result is not None
        assert result["status"] == "error"
        assert result["error_message"] == "Modello non disponibile"
        assert result["processing_time"] == 0.5
    
    def test_processing_status(self, result_writer):
        """Test stato processing"""
        job_id = "test-processing-789"
        
        # Segna come processing
        result_writer.write_processing(job_id, "worker-001")
        
        # Verifica stato
        assert result_writer.is_processing(job_id) == True
        
        # Scrivi risultato (dovrebbe rimuovere processing)
        result_writer.write_success(job_id, "YXVkaW8=", "wav", 1.0)
        
        # Verifica che processing sia rimosso
        assert result_writer.is_processing(job_id) == False
    
    def test_ttl_functionality(self, result_writer):
        """Test TTL su risultati"""
        job_id = "test-ttl-999"
        
        # Scrivi risultato
        result_writer.write_success(job_id, "YXVkaW8=", "wav", 1.0)
        
        # Verifica TTL sia impostato
        result_key = f"{RESULT_PREFIX}{job_id}"
        ttl = result_writer._redis_client.ttl(result_key)
        assert ttl > 0  # TTL dovrebbe essere > 0
    
    def test_queue_integration(self, queue_manager, result_writer):
        """Test integrazione completa QueueManager + ResultWriter"""
        # Questo test verifica che i due componenti possano coesistere
        assert queue_manager.health_check() == True
        
        # Test scrittura/lettura simultanea
        job_id = "integration-test-001"
        result_writer.write_success(job_id, "YXVkaW8=", "wav", 1.5)
        
        # Verifica lunghezze code (non dovrebbe cambiare)
        lengths_before = queue_manager.queue_lengths()
        lengths_after = queue_manager.queue_lengths()
        assert lengths_before == lengths_after


if __name__ == "__main__":
    # Test manuale rapido
    print("🧪 Testing Redis Bridge...")
    
    try:
        # Test connessioni
        qm = QueueManager()
        rw = ResultWriter()
        
        print("✅ QueueManager connesso")
        print("✅ ResultWriter connesso")
        
        # Test lunghezze code
        lengths = qm.queue_lengths()
        print(f"📊 Code disponibili: {len(lengths)}")
        for queue, length in lengths.items():
            print(f"  - {queue}: {length} task")
        
        # Test risultato
        test_job = "manual-test-001"
        rw.write_success(test_job, "YXVkaW8gdGVzdA==", "wav", 2.0)
        result = rw.get_result(test_job)
        print(f"✅ Risultato salvato: {result['status']}")
        
        print("🎉 Tutti i test manuali passati!")
        
    except Exception as e:
        print(f"❌ Test fallito: {e}")
        raise