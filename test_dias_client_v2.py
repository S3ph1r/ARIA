import redis
import json
import uuid
import time
from datetime import datetime, timezone

# Connect to CT120 Redis (Change to localhost if running locally)
r = redis.Redis(host='192.168.1.120', port=6379, db=0)

job_id = f"test-book-{str(uuid.uuid4())[:8]}"
client_id = "dias-minipc"

# Il nome del file che ci aspettiamo in risposta via HTTP Asset Server
expected_output_filename = f"{job_id}_scene-001.wav"

payload = {
    "job_id": job_id,
    "client_id": client_id,
    "model_type": "tts",
    "model_id": "qwen3-tts-1.7b",
    "queued_at": datetime.now(timezone.utc).isoformat(),
    "priority": 1,
    "timeout_seconds": 1800,
    "callback_key": f"gpu:result:{client_id}:{job_id}",
    "payload": {
        "text": "La pioggia cadeva fitta sui tetti di Neo-Milano. Correvo senza fermarmi, sentendo i droni alle mie spalle... il loro ronzio metallico era sempre più vicino.",
        "voice_id": "angelo",
        "output_format": "wav",
        "sample_rate": 24000
    }
}

queue_key = f"gpu:queue:tts:qwen3-tts-1.7b"
print(f"Injecting task {job_id} into {queue_key}...")

# 1. PUSH TASK
r.lpush(queue_key, json.dumps(payload))

# 2. WAIT FOR RESULT
print(f"Waiting on callback key: {payload['callback_key']}...")
start_time = time.time()

# BRPOP blocks until result arrives
result_tuple = r.brpop(payload['callback_key'], timeout=300)

if result_tuple:
    _, result_bytes = result_tuple
    result_json = json.loads(result_bytes.decode('utf-8'))
    elapsed = time.time() - start_time
    
    print(f"\n--- SUCCESS in {elapsed:.2f}s ---")
    print(json.dumps(result_json, indent=2))
    print(f"\nExpected audio URL at: {result_json.get('output', {}).get('audio_url')}")
    print("Verify this URL works or the local file exists in C:\\Users\\Roberto\\aria\\data\\outputs !")
else:
    print(f"\n--- TIMEOUT after 300s ---")
    print("No result received from ARIA Orchestrator.")
