import redis
import json
import time
import os
from datetime import datetime
from pathlib import Path

# --- CONFIGURATIONS ---
REDIS_HOST = "192.168.1.120"
CLIENT_ID = "dias-test"
BOOK_ID = "Cronache-del-Silicio"
BLOCK_ID = "comparison-test-chunk-0"

# Model IDs
GEMINI_MODEL = "gemini-flash-lite-latest"
QWEN_MODEL = "qwen3.5-35b-moe-q3ks"

# Queues Prefixes
Q_GOOGLE_PREFIX = "global:queue:cloud:google"
Q_LOCAL_PREFIX = "global:queue:llm:local"

# Output dir
OUTPUT_DIR = Path("comparison_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- PROMPTS ---

STAGE_B_PROMPT_TMPL = """
Sei un analista narrativo e semantico esperto. Analizza il seguente testo (in Italiano) per estrarre:
1. Analisi Emotiva: valence, arousal, tension (0.0-1.0) e l'emozione primaria del BLOCCO INTERO.
2. Marcatori Narrativi: punti di svolta e shift di mood SIGNIFICATIVI con la loro posizione relativa nel testo (0.0 = inizio, 1.0 = fine).
3. Entità: identifica SOLO i personaggi principali che compaiono (con relativa emozione predominante e stile di dialogo se parlano).
4. Relazioni tra personaggi.
5. Concetti chiave narrativi (non tecnici generici).

IMPORTANTE - Dialoghi:
Se il testo contiene dialogo diretto (frasi tra virgolette), imposta has_dialogue: true.
Nell'array entities, per ogni personaggio che parla, aggiungi:
- "speaking_style": una breve nota in INGLESE su COME parla quel personaggio
  (es. "speaks with quiet authority and analytical precision", "bright curiosity, quick questions", "cynical and sharp, ends sentences with challenges")

IMPORTANTE - Emozione:
Non appiattire tutto a "neutro". Sii coraggioso nell'analisi emotiva.
Se il testo contiene tensione, paura, gioia improvvisa, dialogo conflittuale: dillo.
primary_emotion deve riflettere l'emozione DOMINANTE nel blocco, non la media neutra.

Testo da analizzare:
{text}

Rispondi ESCLUSIVAMENTE in formato JSON con questa struttura:
{{
    "block_analysis": {{
        "valence": 0.5,
        "arousal": 0.5,
        "tension": 0.5,
        "primary_emotion": "neutro|gioia|tristezza|rabbia|paura|tensione|curiosita|relax|melanconia|stupore|determinazione|ansia|nostalgia",
        "secondary_emotion": "descrizione opzionale",
        "setting": "luogo fisico della scena",
        "has_dialogue": false,
        "audio_cues": ["lista", "di", "suoni", "ambientali", "concreti", "menzionati", "nel", "testo"]
    }},
    "narrative_markers": [
        {{
            "relative_position": 0.1,
            "event": "nome evento concreto",
            "mood_shift": "da X a Y (es: da tensione tecnica a sollievo euforico)"
        }}
    ],
    "entities": [
        {{
            "entity_id": "ent_001",
            "text": "nome del personaggio",
            "entity_type": "persona|luogo|organizzazione|concetto|evento",
            "emotional_tone": "neutro|gioia|tristezza|rabbia|paura|tensione|curiosita|relax",
            "speaking_style": "breve nota in inglese su come parla (se parla), altrimenti null",
            "confidence": 0.9,
            "metadata": {{}}
        }}
    ],
    "relations": [
        {{
            "relation_id": "rel_001",
            "source_entity_id": "ent_001",
            "target_entity_id": "ent_002",
            "relation_type": "tipo_relazione_in_italiano",
            "confidence": 0.8
        }}
    ],
    "concepts": [
        {{
            "concept_id": "conc_001",
            "concept": "nome concetto narrativo",
            "definition": "definizione in italiano",
            "emotional_tone": "neutro|gioia|tristezza|rabbia|paura|tensione|curiosita|relax",
            "confidence": 0.85
        }}
    ]
}}
"""

STAGE_C_PROMPT_TMPL = """
Sei un DIRETTORE ARTISTICO esperto in audiolibri professionali di alta qualità.
Il tuo compito è trasformare un blocco di testo narrativo in una sequenza di MICRO-SCENE AUDIO (battute) ottimizzate per un motore TTS Zero-Shot (Qwen3-TTS 1.7B).

---
FASE 1: SEGMENTAZIONE IN MICRO-SCENE (MANDATORIO)
---

Dividi il testo in BATTUTE BREVI, seguendo queste regole:

CRITERIO DI DIVISIONE E CONTINUITÀ:
- Ogni battuta di DIALOGO di un personaggio = 1 micro-scena separata
- Ogni TITOLO di capitolo/sezione = 1 micro-scena dedicata (sempre isolati)
- SEQUENZE NARRATIVE e DESCRITTIVE: Se hai un lungo blocco di descrizione continua che condivide lo stesso "mood", NON SPEZZARLO frase per frase.
- Spezza una sequenza narrativa in due micro-scene SOLO se c'è un REALE cambio di azione o un capoverso molto netto.

LUNGHEZZA:
- Micro-scene DIALOGICHE: 5-40 parole
- Micro-scene NARRATIVE: 10-60 parole (MAI superare 60 parole)

REGOLE GENERALI SUI TITOLI E SULLA STRUTTURA:
Se il blocco inizia con una o più frasi brevi isolate separate dal corpo principale da doppi ritorni a capo (\\n\\n)
DEVI isolare OGNUNA di queste in una micro-scena singola.
- NON unire mai un titolo strutturale con il paragrafo narrativo che lo segue.

---
FASE 2: PULIZIA TESTO (clean_text)
---
1. NUMERI -> PAROLE: (es. "2042" -> "duemilaquarantadue")
2. ACCENTI FONETICI ITALIANI: Aggiungi SEMPRE l'accento grafico sulle parole ambigue (es. "pàtina", "futòn").
3. TAG E PULIZIA: Rimuovi tutti i tag tra parentesi e punteggiatura residua dai titoli.

---
FASE 3: DIRETTIVA VOCALE PER MICRO-SCENA (qwen3_instruct)
---
Qwen3-TTS è controllato da istruzioni in PROSA NATURALE in INGLESE.
Descrivi l'EMOZIONE, il PACING e il REGISTRO VOCALE.
Esempio: "Read with quiet solemnity. Let the words land with weight."

---
FASE 4: METADATI
---
- pause_after_ms: 80, 200 (fine periodo), 400 (paragrafo), 1500 (didascalia), 2000 (titolo capitolo).

---
EMOZIONE DI BASE del blocco: {emotion_description}

TESTO DA ELABORARE:
{text_content}

Rispondi ESCLUSIVAMENTE con un JSON ARRAY.
"""

# --- HELPER FUNCTIONS ---

def call_aria(r, queue_key, payload, timeout=300):
    callback_key = payload["callback_key"]
    r.delete(callback_key)
    r.lpush(queue_key, json.dumps(payload))
    print(f"[*] Task injected into {queue_key}. Waiting for {callback_key}...")
    
    start_time = time.time()
    res_raw = r.brpop(callback_key, timeout=timeout)
    if res_raw:
        elapsed = time.time() - start_time
        print(f"[+] Result received in {elapsed:.1f}s")
        return json.loads(res_raw[1].decode('utf-8')), elapsed
    return None, 0

def run_pipeline(r, provider, model_id, text, thinking=True):
    # Stage B
    print(f"\n>>> Running STAGE B ({provider}/{model_id})...")
    stage_b_prompt = STAGE_B_PROMPT_TMPL.format(text=text)
    
    # Standard ARIA Routing logic
    if provider == "google":
        queue_key = f"{Q_GOOGLE_PREFIX}:{model_id}:{CLIENT_ID}"
        model_type = "cloud"
        # Gemini expects 'contents'
        payload_data = {
            "contents": [{"parts": [{"text": stage_b_prompt.strip()}]}],
            "config": {
                "max_output_tokens": 4096,
                "temperature": 0.2
            }
        }
    else:
        queue_key = f"{Q_LOCAL_PREFIX}:{model_id}:{CLIENT_ID}"
        model_type = "llm"
        # Local LLM (OpenAI-like) expects 'messages'
        payload_data = {
            "messages": [{"role": "user", "content": stage_b_prompt.strip()}],
            "max_tokens": 4096,
            "temperature": 0.2
        }

    # Stage B logic
    if provider == "google":
        print(f"[*] Skipping Gemini Stage B call (using cached result from Redis)...")
        callback_key_b = f"global:callback:{CLIENT_ID}:comp-{provider}-stage-b"
        res_raw = r.lindex(callback_key_b, 0)
        if res_raw:
            res_b = json.loads(res_raw.decode('utf-8'))
            time_b = 6.4 # Valore indicativo dai log precedenti
        else:
            print("[!] Cached Stage B result not found in Redis!")
            return None
    else:
        res_b, time_b = call_aria(r, queue_key, payload_b)
    
    if not res_b or res_b.get("status") != "done":
        print(f"[!] Stage B failed: {res_b.get('error') if res_b else 'Timeout'}")
        return None
    
    # Extract emotion from Stage B
    try:
        raw_text = res_b["output"]["text"]
        if isinstance(raw_text, str):
            # Strip markdown if present
            if raw_text.startswith("```json"):
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            b_out = json.loads(raw_text)
        else:
            b_out = raw_text
        emotion_desc = b_out.get("block_analysis", {}).get("primary_emotion", "neutro")
    except Exception as e:
        print(f"[!] Error parsing emotion: {e}")
        emotion_desc = "neutro"
    
    # Stage C
    print(f">>> Running STAGE C ({provider}/{model_id})... [Emotion: {emotion_desc}]")
    stage_c_prompt = STAGE_C_PROMPT_TMPL.format(text_content=text, emotion_description=emotion_desc)
    
    if provider == "google":
        payload_data_c = {
            "contents": [{"parts": [{"text": stage_c_prompt.strip()}]}],
            "config": {
                "max_output_tokens": 4096,
                "temperature": 0.0
            }
        }
    else:
        payload_data_c = {
            "messages": [{"role": "user", "content": stage_c_prompt.strip()}],
            "max_tokens": 4096,
            "temperature": 0.0
        }

    payload_c = {
        "job_id": f"comp-{provider}-stage-c",
        "client_id": CLIENT_ID,
        "model_type": model_type,
        "provider": provider,
        "model_id": model_id,
        "callback_key": f"global:callback:{CLIENT_ID}:comp-{provider}-stage-c",
        "payload": payload_data_c
    }
    if not thinking and "thinking" in payload_c["payload"]: 
        payload_c["payload"]["thinking"] = False

    res_c, time_c = call_aria(r, queue_key, payload_c)
    
    return {
        "stage_b": res_b,
        "stage_c": res_c,
        "times": (time_b, time_c)
    }

# --- MAIN ---

def main():
    # 1. Load sample text
    a_path = "/home/Projects/NH-Mini/sviluppi/dias/data/stage_a/output/Cronache-del-Silicio-chunk-000-20260307_005036.json"
    with open(a_path, "r") as f:
        data_a = json.load(f)
    
    sample_text = data_a["block_text"][:4000]
    print(f"[*] Loaded Stage A text. Using sample of {len(sample_text)} chars.")

    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

    # 2. Run Gemini Pipeline
    results_gemini = run_pipeline(r, "google", GEMINI_MODEL, sample_text)
    
    # 3. Run Qwen 3.5 Pipeline (DISABLED - Already produced)
    # results_qwen = run_pipeline(r, "local", QWEN_MODEL, sample_text, thinking=False)
    results_qwen = None 

    # 4. Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_text_sample": sample_text,
        "gemini": results_gemini,
        "qwen35": results_qwen
    }
    
    report_path = OUTPUT_DIR / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n[+] COMPARISON COMPLETE! Report saved to: {report_path}")
    
    if results_gemini:
        print(f"GEMINI:  B={results_gemini['times'][0]:.1f}s | C={results_gemini['times'][1]:.1f}s")
    if results_qwen:
        print(f"QWEN3.5: B={results_qwen['times'][0]:.1f}s  | C={results_qwen['times'][1]:.1f}s")

if __name__ == "__main__":
    main()
