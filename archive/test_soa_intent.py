import redis
import json
import time
import uuid

def test_intent_resolution():
    r = redis.Redis(host='192.168.1.120', port=6379, decode_responses=True)
    
    job_id = f"test_soa_{uuid.uuid4().hex[:8]}"
    callback_key = f"callback:{job_id}"
    
    # Task payload following SOA v2.0 (Intent-only)
    task = {
        "job_id": job_id,
        "client_id": "surgical_test_lxc190",
        "model_type": "tts",
        "model_id": "fish-s1-mini",
        "callback_key": callback_key,
        "payload": {
            "text": "Questo è un test del nuovo sistema SOA di ARIA. Sto usando solo il voice ID narratore.",
            "voice_id": "narratore",
            "temperature": 0.7
        }
    }
    
    print(f"[*] Sending task {job_id} to Redis...")
    r.lpush("gpu:queue:tts:fish-s1-mini", json.dumps(task))
    
    print(f"[*] Waiting for result on {callback_key}...")
    result_raw = r.brpop(callback_key, timeout=60)
    
    if result_raw:
        result = json.loads(result_raw[1])
        print("\n[+] SUCCESS! Result received:")
        print(json.dumps(result, indent=2))
        
        if "output" in result and "audio_url" in result["output"]:
            url = result["output"]["audio_url"]
            print(f"\n[!] Audio URL: {url}")
            print(f"[*] You can try to download it: curl -I {url}")
    else:
        print("\n[!] TIMEOUT: No result received after 60 seconds.")

if __name__ == "__main__":
    test_intent_resolution()
