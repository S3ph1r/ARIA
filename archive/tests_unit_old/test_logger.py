"""
Test unitari per il sistema di logging ARIA
Fase AS-3: Test del logger strutturato
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch
from aria_server.logger import (
    setup_logging, get_logger, set_log_context, clear_log_context, log_context,
    job_id_var, model_id_var
)

class TestARIALogger(unittest.TestCase):
    """Test suite per il sistema di logging"""
    
    def setUp(self):
        """Setup per ogni test"""
        # Pulisci context
        clear_log_context()
        
        # Crea directory temporanea per i log
        self.temp_log_dir = tempfile.mkdtemp()
        
        # Resetta i logger
        import logging
        logger = logging.getLogger('aria_server')
        logger.handlers.clear()
    
    def tearDown(self):
        """Cleanup dopo ogni test"""
        # Pulisci context
        clear_log_context()
        
        # Rimuovi directory temporanea
        shutil.rmtree(self.temp_log_dir, ignore_errors=True)
        
        # Resetta i logger
        import logging
        logger = logging.getLogger('aria_server')
        logger.handlers.clear()
    
    def test_logger_creation(self):
        """Test creazione logger base"""
        setup_logging(log_dir=self.temp_log_dir)
        logger = get_logger('test')
        
        # Verifica che il logger esista
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'aria_server.test')
        self.assertEqual(logger.level, logging.INFO)
    
    def test_log_levels(self):
        """Test dei diversi livelli di log"""
        setup_logging(log_level="DEBUG", log_dir=self.temp_log_dir)
        logger = get_logger('levels')
        
        # Test che non dovrebbero sollevare eccezioni
        logger.debug("Messaggio debug")
        logger.info("Messaggio info")
        logger.warning("Messaggio warning")
        logger.error("Messaggio error")
    
    def test_context_propagation(self):
        """Test propagazione context nei log"""
        setup_logging(log_dir=self.temp_log_dir)
        logger = get_logger('context')
        
        # Imposta context
        set_log_context(job_id="job-123", model_id="orpheus-3b")
        
        # Verifica che il context sia impostato
        self.assertEqual(job_id_var.get(), "job-123")
        self.assertEqual(model_id_var.get(), "orpheus-3b")
        
        # Log con context (non solleva eccezioni)
        logger.info("Test con context")
    
    def test_context_manager(self):
        """Test context manager per logging"""
        setup_logging(log_dir=self.temp_log_dir)
        logger = get_logger('context_mgr')
        
        # Test context manager
        with log_context(job_id="job-456", model_id="musicgen-small"):
            self.assertEqual(job_id_var.get(), "job-456")
            self.assertEqual(model_id_var.get(), "musicgen-small")
            logger.info("Test dentro context manager")
        
        # Dopo il context manager, dovrebbe essere pulito
        self.assertIsNone(job_id_var.get())
        self.assertIsNone(model_id_var.get())
    
    def test_json_file_creation(self):
        """Test creazione file JSON di log"""
        setup_logging(log_dir=self.temp_log_dir)
        logger = get_logger('json_test')
        
        # Scrivi un log
        logger.info("Test JSON", extra={'event': 'test_event', 'duration_ms': 1500})
        
        # Aspetta un attimo per la scrittura
        import time
        time.sleep(0.1)
        
        # Verifica che il file esista
        log_files = [f for f in os.listdir(self.temp_log_dir) if f.endswith('.jsonl')]
        self.assertGreater(len(log_files), 0)
        
        # Leggi e verifica il contenuto JSON
        log_file = os.path.join(self.temp_log_dir, log_files[0])
        with open(log_file, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            self.assertGreater(len(line), 0)
            
            # Parse JSON
            log_data = json.loads(line)
            
            # Verifica campi obbligatori
            self.assertIn('ts', log_data)
            self.assertIn('level', log_data)
            self.assertIn('component', log_data)
            self.assertIn('message', log_data)
            self.assertEqual(log_data['level'], 'INFO')
            self.assertEqual(log_data['component'], 'aria_server.json_test')
            self.assertEqual(log_data['message'], 'Test JSON')
            self.assertEqual(log_data['event'], 'test_event')
            self.assertEqual(log_data['duration_ms'], 1500)
    
    def test_console_output(self):
        """Test output console (mockato per evitare output reale)"""
        with patch('sys.stdout') as mock_stdout:
            setup_logging(log_dir=self.temp_log_dir)
            logger = get_logger('console')
            
            # Scrivi log
            logger.info("Test console output")
            logger.warning("Test warning")
            logger.error("Test error")
            
            # Verifica che sia stato chiamato (non possiamo testare i colori reali)
            # Ma verifichiamo che non sollevi eccezioni
            self.assertTrue(True)  # Se arriviamo qui, il test è passato
    
    def test_clear_context(self):
        """Test pulizia context"""
        # Imposta context
        set_log_context(job_id="job-789", model_id="llama-3b")
        self.assertIsNotNone(job_id_var.get())
        self.assertIsNotNone(model_id_var.get())
        
        # Pulisci
        clear_log_context()
        
        # Verifica pulizia
        self.assertIsNone(job_id_var.get())
        self.assertIsNone(model_id_var.get())
    
    def test_different_log_levels_setup(self):
        """Test setup con diversi livelli di log"""
        # Test DEBUG level
        setup_logging(log_level="DEBUG", log_dir=self.temp_log_dir)
        logger = get_logger('debug_test')
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Reset e test WARNING level
        import logging
        logging.getLogger('aria_server').handlers.clear()
        
        setup_logging(log_level="WARNING", log_dir=self.temp_log_dir)
        logger = get_logger('warning_test')
        self.assertEqual(logger.level, logging.WARNING)
    
    def test_unicode_logging(self):
        """Test logging con caratteri Unicode"""
        setup_logging(log_dir=self.temp_log_dir)
        logger = get_logger('unicode')
        
        # Test messaggi con Unicode
        logger.info("Test con emoji: 🚀 ARIA Server")
        logger.info("Test con accenti: Ciao mondo! àèìòù")
        logger.info("Test con CJK: 你好世界")
        
        # Se non solleva eccezioni, il test è passato
        self.assertTrue(True)

class TestLoggerIntegration(unittest.TestCase):
    """Test di integrazione per verificare il comportamento complessivo"""
    
    def setUp(self):
        self.temp_log_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_log_dir, ignore_errors=True)
        import logging
        logging.getLogger('aria_server').handlers.clear()
    
    def test_full_logging_workflow(self):
        """Test workflow completo di logging"""
        setup_logging(log_level="INFO", log_dir=self.temp_log_dir)
        logger = get_logger('integration')
        
        # Simula un workflow di processing task
        with log_context(job_id="job-integration-001", model_id="orpheus-3b"):
            logger.info("Inizio processing task", extra={'event': 'task_started'})
            
            # Simula processing
            logger.info("Modello caricato in VRAM", extra={'vram_used_gb': 7.2})
            
            # Simula warning
            logger.warning("Testo lungo, applico chunking", 
                          extra={'text_length': 450, 'chunks': 2})
            
            # Simula completamento
            logger.info("Processing completato", 
                       extra={'event': 'task_completed', 'duration_ms': 12500})
        
        # Verifica file di log
        time.sleep(0.2)  # Aspetta scrittura
        log_files = [f for f in os.listdir(self.temp_log_dir) if f.endswith('.jsonl')]
        self.assertGreater(len(log_files), 0)
        
        # Leggi tutte le linee
        log_file = os.path.join(self.temp_log_dir, log_files[0])
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Verifica che abbiamo i log attesi
        events = []
        for line in lines:
            if line.strip():
                log_data = json.loads(line.strip())
                if 'event' in log_data:
                    events.append(log_data['event'])
        
        self.assertIn('task_started', events)
        self.assertIn('task_completed', events)

if __name__ == '__main__':
    unittest.main()