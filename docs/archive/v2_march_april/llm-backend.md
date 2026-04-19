# ARIA LLM Backend (v2.1)

Il backend LLM di ARIA permette l'esecuzione di task di inferenza testuale (generazione, analisi, trasformazione) in locale o in cloud. 

---

## 1. Qwen 3.5 MoE (Backend Locale)

Il modello di riferimento attuale per l'inferenza locale è **Qwen2.5-35B-Instruct-Moe-GPTQ-Int4** (o versioni GGUF).

### Architettura Standalone
- **Engine**: `llama-server.exe` (OpenAI API Compatible).
- **Integrazione**: ARIA gestisce il ciclo di vita del server sulla porta `1234`.
- **Coda Redis**: `aria:q:local:llm:qwen3.5-35b-moe-q3ks:dias`

### Dettagli del Payload
Il backend supporta messaggi strutturati e il controllo del budget di "pensiero":

```json
{
  "messages": [
    {"role": "system", "content": "Sei un direttore artistico..."},
    {"role": "user", "content": "Analizza questo capitolo."}
  ],
  "max_tokens": 4096,
  "temperature": 0.2,
  "thinking": true
}
```

### Pensiero (Thinking Mode)
Qwen 3.5 supporta il ragionamento interno. ARIA estrae automaticamente il contenuto tra i tag `<thought>...</thought>` (o `<think>`) e lo isola nel campo `thinking` della risposta, consegnando il testo pulito nel campo `text`.

---

## 2. Gemini Flash Lite (Backend Cloud)
Gestito tramite il `CloudManager` e il `GeminiRateLimiter` per garantire l'aderenza alle quote API.

- **Modello**: `gemini-flash-lite-latest`
- **Coda**: `aria:q:cloud:google:gemini-1.5-flash-lite:dias`
- **Pacing**: Minimo 30s tra i task (configurabile in `GeminiRateLimiter`).
- **Quota Management**: In caso di errore 429, ARIA entra in modalit "Lockout" globale finch la quota non viene ripristinata.

---

## 3. Configurazione Ambiente (PC 139)

- **Conda Environment**: `nh-qwen35-llm`
- **Hardware**: Richiede GPU con almeno 16GB VRAM (es. RTX 5060 Ti / 4070+).
- **Path Modello**: `C:\Users\Roberto\aria\models\qwen3.5-35b-moe-q3ks.gguf`

### Avvio Backend
L'Orchestratore avvia il backend con:
```cmd
%ARIA_ROOT%\envs\nh-qwen35-llm\python.exe aria_node_controller/backends/qwen35_llm.py
```
*(Nota: il server llama-server deve essere già in esecuzione o avviato dallo script .bat principale)*

---

## 4. Backend Correlati

### Qwen3-TTS (Audio)
Mentre Qwen 3.5 MoE gestisce il testo, il modello **Qwen3-TTS 1.7B** gestisce la narrazione vocale basata su istruzioni (Instruct-TTS).
- **Doc**: [qwen3-tts-backend.md](qwen3-tts-backend.md)
- **Coda**: `aria:q:local:tts:qwen3-tts-1.7b:dias`
