import redis
import json
import time
from datetime import datetime, timezone

# Configurazioni
REDIS_HOST = "192.168.1.120"
CLIENT_ID = "dias"
MODEL_ID = "qwen3.5-35b-moe-q3ks"
QUEUE_KEY = f"global:queue:llm:local:{MODEL_ID}:{CLIENT_ID}"
CALLBACK_KEY = f"global:callback:{CLIENT_ID}:test-e2e-qwen35-mini"

# System Prompt (Stage C Director)
SYSTEM_PROMPT = """
Sei un DIRETTORE ARTISTICO esperto in audiolibri professionali di alta qualità.
Il tuo compito è trasformare un blocco di testo narrativo in una sequenza di MICRO-SCENE AUDIO (battute) ottimizzate per un motore TTS Zero-Shot (Qwen3-TTS 1.7B).

═══════════════════════════════════════════════════════════════
FASE 1: SEGMENTAZIONE IN MICRO-SCENE (MANDATORIO)
═══════════════════════════════════════════════════════════════
Dividi il testo in BATTUTE BREVI.
- Micro-scene DIALOGICHE: 5-40 parole
- Micro-scene NARRATIVE: 10-60 parole
- MAI superare 60 parole per micro-scena

FASE 2: PULIZIA TESTO (clean_text)
- NUMERI -> PAROLE per esteso
- ACCENTI FONETICI su parole ambigue
- Rimuovi tag e pulisci punteggiatura.

FASE 3: DIRETTIVA VOCALE (qwen3_instruct)
- 1-2 frasi in inglese su emozione, pacing e registro vocale.

Rispondi ESCLUSIVAMENTE con un JSON ARRAY. Formato:
[
  {
    "scene_label": "breve etichetta",
    "clean_text": "testo pulito",
    "qwen3_instruct": "instruction in english",
    "speaker": "Nome o null",
    "pause_after_ms": 200
  }
]
"""

# Input Data (Minimal Test)
TEXT_CONTENT = "Ciao, come ti chiami? Rispondi brevemente."
EMOTION_DESCRIPTION = "amichevole"

# Payload
task_payload = {
    "job_id": "test-e2e-qwen35-mini",
    "client_id": CLIENT_ID,
    "model_type": "llm",
    "model_id": MODEL_ID,
    "provider": "local",
    "callback_key": CALLBACK_KEY,
    "payload": {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": f"EMOZIONE DI BASE: {EMOTION_DESCRIPTION}\n\nTESTO DA ELABORARE:\n{TEXT_CONTENT}"}
        ],
        "thinking": False,
        "temperature": 0.0,
        "max_tokens": 4096
    }
}

def main():
    try:
        r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
        print(f"[*] Connessione a Redis ({REDIS_HOST}) OK.")
        
        # Pulizia callback precedente
        r.delete(CALLBACK_KEY)
        
        print(f"[*] Iniezione task in {QUEUE_KEY}...")
        r.lpush(QUEUE_KEY, json.dumps(task_payload))
        
        print(f"[*] Attesa risultato su {CALLBACK_KEY} (timeout 300s)...")
        start_time = time.time()
        result_raw = r.brpop(CALLBACK_KEY, timeout=300)
        
        if result_raw:
            elapsed = time.time() - start_time
            print(f"\n[+] RISULTATO RICEVUTO in {elapsed:.2f}s!")
            result = json.loads(result_raw[1].decode('utf-8'))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            if result.get("status") == "done":
                print("\n[SUCCESS] Qwen 3.5 E2E Test completato con successo.")
            else:
                print(f"\n[FAILED] ARIA ha restituito un errore: {result.get('error')}")
        else:
            print("\n[TIMEOUT] Nessun risultato ricevuto. Controlla i log di ARIA su PC 139.")
            
    except Exception as e:
        print(f"\n[ERROR] Errore nello script di test: {e}")

if __name__ == "__main__":
    main()
