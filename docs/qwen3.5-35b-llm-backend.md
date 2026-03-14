# Qwen3.5-35B-A3B-MoE — Backend LLM per ARIA

> **Aggiornato**: 2026-03-15
> **Ambiente**: `%ARIA_ROOT%\envs\nh-qwen35-llm` (Python 3.12)
> **Motore**: llama-cpp-python (GGUF)
> **Hardware**: RTX 5060 Ti 16GB (Blackwell sm_120)
> **Stato**: 🏗️ In fase di installazione

---

## 1. Panoramica

Questo backend integra il modello **Qwen3.5-35B-A3B-MoE** (quantizzato Q3_K_S) come motore di inferenza locale per ARIA. Sostituisce le chiamate cloud (Gemini) per i task pesanti di DIAS Stage B/C, garantendo:
- **Zero latenza di rete**: Inferenza diretta sulla GPU locale.
- **Privacy totale**: I dati non lasciano mai il PC Gaming.
- **Supporto "Thinking"**: Capacità di ragionamento esplicito (Chain of Thought) tramite tag `<thought>`.
- **Ottimizzazione VRAM**: Uso di MoE (Mixture of Experts) e KV Cache a 8-bit per far girare un modello da 35B su soli 16GB.

---

## 2. Architettura Tecnica

### GGUF + llama-cpp-python

A differenza dei backend TTS (basati su Transformers), abbiamo scelto **llama-cpp-python** per:
1. **Efficienza MoE**: Gestione superiore dei modelli Mixture of Experts.
2. **Quantizzazione Quantistica**: Il formato GGUF permette di caricare il modello Q3_K_S (~13.8 GB) lasciando spazio per il contesto.
3. **8-bit KV Cache**: Riduce drasticamente il consumo di VRAM durante l'elaborazione di sequenze lunghe (DIAS).

### Specifiche di Memoria (RTX 5060 Ti 16GB)

| Componente | Memoria Stimata | Note |
|------------|-----------------|------|
| **Modello (Q3_K_S)** | 13.8 GB | Caricato integralmente in VRAM |
| **8-bit KV Cache** | ~1.2 GB | Per un contesto di 32k token |
| **Overhead CUDA** | ~0.6 GB | Sistema e kernel |
| **Margine Libero** | **~0.4 GB** | Molto stretto, richiede JIT aggressivo |

---

## 3. Setup Ambiente (PC Gaming)

### Dipendenze Esterne
Prima di configurare l'ambiente, il PC deve avere:
1. **NVIDIA Studio Driver** (Recenti, serie 590+)
2. **CUDA Toolkit 13.2+** (Necessario per `nvcc`)
3. **Visual Studio 2022 Build Tools** (Componente "Sviluppo desktop con C++")

### Creazione Ambiente & Build
L'installazione richiede una compilazione "Native" per l'architettura Blackwell:

```cmd
:: 1. Creazione ambiente
conda create --prefix %ARIA_ROOT%\envs\nh-qwen35-llm python=3.12 -y

:: 2. Build ottimizzata sm_120 (da fare in "x64 Native Tools Command Prompt")
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_GENERATOR_PLATFORM=x64
%ARIA_ROOT%\envs\nh-qwen35-llm\python.exe -m pip install llama-cpp-python --no-cache-dir --verbose
```

---

## 4. Download Modello

Il modello viene scaricato da Hugging Face in formato GGUF:

```cmd
huggingface-cli download bartowski/Qwen_Qwen3.5-35B-A3B-GGUF ^
    --include "*Q3_K_S.gguf*" ^
    --local-dir %ARIA_ROOT%\data\models\Qwen3.5-35B-A3B-GGUF
```

---

## 5. Thinking Mode & Payload

Il backend supporta la modalità "ragionamento". Quando il flag `thinking: true` è presente, il server estrae il contenuto tra i tag `<thought>` e lo restituisce separatamente.

### Payload Esempio
```json
{
  "prompt": "Analizza la coerenza logica della scena 003...",
  "thinking": true,
  "max_tokens": 4096,
  "temperature": 0.7
}
```

### Risposta JSON
```json
{
  "thought": "L'utente chiede un'analisi... la scena 003 presenta un'incongruenza temporale...",
  "response": "La scena 003 è coerente, ma suggerisco di rivedere il timestamp..."
}
```

---

## 6. Integrazione nel Batch Processor

Per ottimizzare lo swap in VRAM, il modello viene gestito dal `batch_processor.py` con logica **Greedy Drain-First**:
1. ARIA accumula i task LLM nella coda Redis.
2. Il Batch Processor scarica eventuali modelli TTS (Fish/Qwen3).
3. Carica il modello 35B.
4. Svuota completamente la coda LLM prima di permettere il cambio modello.

---

## 7. Troubleshooting Blackwell

### Errore 'cudafe++' died with status 0xC0000005
Causato da un mismatch di architettura (compilazione x86 su sistema x64).
**Risoluzione**: Usare sempre **"x64 Native Tools Command Prompt"** per l'installazione di `llama-cpp-python`.

### Lentezza nel Context Processing
Assicurarsi che la KV Cache sia impostata su `cache_type_k="q8_0"` e `cache_type_v="q8_0"` nelle impostazioni del server.
