import json
import uuid
import time
import redis
from datetime import datetime, timezone

# Redis Configuration
REDIS_HOST = "192.168.1.120"
REDIS_PORT = 6379

def test_gateway_cloud():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    job_id = f"test-cloud-{uuid.uuid4().hex[:6]}"
    client_id = "test_verify_script"
    callback_key = f"global:callback:{client_id}:{job_id}"
    
    # Matching AriaTaskPayload schema
    payload = {
        "job_id": job_id,
        "client_id": client_id,
        "model_type": "cloud",
        "provider": "google",
        "model_id": "gemini-1.5-flash",
        "callback_key": callback_key,
        "payload": {
            "contents": [
                {"role": "user", "parts": [{"text": "Ciao ARIA, rispondi con una breve rima sul fatto che sei diventata un Gateway potente."}]}
            ],
            "config": {
                "temperature": 0.7
            }
        },
        "schema_version": "1.0"
    }
    
    # Queue Key: global:queue:cloud:google:gemini-1.5-flash:test_verify_script
    queue_key = f"global:queue:cloud:google:gemini-1.5-flash:{client_id}"
    
    print(f"Sending task {job_id} to {queue_key}...")
    r.lpush(queue_key, json.dumps(payload))
    
    print(f"Waiting for result on {callback_key} (timeout 60s)...")
    res = r.brpop(callback_key, timeout=60)
    
    if res:
        _, raw_result = res
        result = json.loads(raw_result)
        print("\n=== GOT RESULT ===")
        print(json.dumps(result, indent=2))
        if result.get("status") == "done":
            print("\nSUCCESS: Gateway cloud task processed correctly!")
        else:
            print("\nERROR: Task failed.")
    else:
        print("\nTIMEOUT: No result received. Check ARIA logs on PC 139.")

if __name__ == "__main__":
    test_gateway_cloud()
