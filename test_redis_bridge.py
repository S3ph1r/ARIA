#!/usr/bin/env python3
"""
Test Redis Bridge - Esecuzione diretta
"""
from aria_server import QueueManager, ResultWriter

def test_redis_bridge():
    """Test semplice Redis Bridge"""
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
        job_id = "container-test-001"
        rw.write_success(job_id, "YXVkaW8gdGVzdA==", "wav", 1.5)
        result = rw.get_result(job_id)
        print(f"✅ Risultato salvato: {result['status']}")
        
        print("🎉 Redis Bridge completamente funzionante!")
        
    except Exception as e:
        print(f"❌ Test fallito: {e}")
        raise

if __name__ == "__main__":
    test_redis_bridge()