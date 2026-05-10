import redis
import json
import time

def send_mock_task():
    r = redis.Redis(host='192.168.1.120', port=6379, decode_responses=True)
    
    mock_payload = {
        "task_id": f"test-task-{int(time.time())}",
        "type": "tts",
        "parameters": {
            "text": "Hello world from the DIAS testing script!",
            "voice": "en_us"
        },
        "priority": 1,
        "timestamp": time.time()
    }
    
    # LPUSH is used so BRPOP on the other side can pop from the right
    r.lpush("aria:tasks", json.dumps(mock_payload))
    print(f"Sent mock task: {mock_payload['task_id']}")

if __name__ == "__main__":
    send_mock_task()
