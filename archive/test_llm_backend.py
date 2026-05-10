import redis
import json
import uuid
import time
import sys

# === NH-MINI GROUNDED TEST ===
# Target: Qwen3.5 LLM Backend (Port 8085)
# Requirement: Build successful + llm_server.py running

def test_llm_task():
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    
    task_id = f"test-llm-{uuid.uuid4().hex[:8]}"
    client_id = "test-script"
    
    # Task Payload structure for ARIA
    task = {
        "job_id": task_id,
        "client_id": client_id,
        "model_type": "llm",
        "provider": "local",
        "model_id": "qwen3.5-35b-moe-q3ks",
        "payload": {
            "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain the importance of the Verification-First protocol in NH-Mini.<|im_end|>\n<|im_start|>assistant\n<thought>",
            "max_tokens": 512,
            "temperature": 0.7,
            "thinking": True
        }
    }
    
    queue_key = "global:queue:llm:local:qwen3.5-35b-moe-q3ks:test-script"
    result_key = f"gpu:result:{client_id}:{task_id}"
    
    print(f"Pushing task {task_id} to {queue_key}...")
    r.lpush(queue_key, json.dumps(task))
    
    print("Waiting for result (timeout 300s)...")
    start_time = time.time()
    while time.time() - start_time < 300:
        res = r.get(result_key)
        if res:
            data = json.loads(res)
            print("\n=== RESULT RECEIVED ===")
            print(f"Status: {data.get('status')}")
            if data.get('status') == 'done':
                output = data.get('output', {})
                print(f"Thinking Process: {output.get('thinking')[:200]}...")
                print(f"Final Text: {output.get('text')}")
                print(f"Usage: {output.get('usage')}")
            else:
                print(f"Error: {data.get('error')}")
            return
        time.sleep(2)
    
    print("Timeout reached.")

if __name__ == "__main__":
    test_llm_task()
