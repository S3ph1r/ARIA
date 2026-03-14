# NH-Mini Universal Inference Gateway
## Visione Architetturale e Roadmap (v1.0)
**Data:** 12 Marzo 2026
**Ecosistema:** NH-Mini (DIAS, ARIA, Future Apps)

---

## 1. Executive Summary

La visione a lungo termine per l'ecosistema **NH-Mini** (l'infrastruttura di automazione domestica/laboratorio IA) stabilisce un cambio di paradigma fondamentale: trasformare **ARIA** da un semplice esecutore di modelli locali (GPU Worker) a un **Universal Inference Gateway**.

In questa nuova architettura:
1. **Le Applicazioni (Client) sono "stude"**: Nessuna applicazione (es. DIAS) possiede API Key, logica di rate-limiting, o conoscenza dell'infrastruttura di esecuzione. Le app conoscono solo il "Cosa" (l'Intento), formattano un payload JSON agnostico (es. "Riassumi questo testo", "Genera questo audio") e lo inviano.
2. **ARIA è il "Guardiano delle Chiavi" (Gateway)**: ARIA espande le sue competenze oltre i modelli GPU in locale. Offre nuovi "Cloud Backends" che fungono da ponte verso Google Gemini, OpenAI, Claude, ecc. Custodisce le chiavi, gestisce le quote e regola il traffico.
3. **Redis (LXC 120) è il "Sistema Nervoso"**: Il punto di contatto universale e asincrono. Chiunque si scambia informazioni lo fa tramite code e publisher/subscriber su questo nodo.
4. **Indipendenza dei Repository**: Ogni progetto (DIAS, ARIA) è un mondo isolato su Git. Il deploy è banale: cloni, installi i requisiti, configuri l'indirizzo IP di Redis, e il servizio è vivo. I pezzi si incastrano dinamicamente perché comunicano solo tramite l'Event Bus (LXC 120).

---

## 2. Topologia dell'Infrastruttura

### 2.1 Il Bus Centrale (LXC 120 - "The Highway")
Il database Redis funge da Event Bus e Data Store temporaneo/In-Memory per tutta la casa.
- Applica il pattern **Pub/Sub** o **FIFO Queues**.
- Nessun client (DIAS, AppX) parla *direttamente* via HTTP/REST con ARIA per le inferenze. Tutto passa da Redis.
- Esempi di code globali: `gpu:queue:tts:...` (Locale), `gpu:queue:llm:gemini-flash` (Cloud).

### 2.2 Il Gateway (ARIA - PC Gaming Locale)
ARIA diventa un servizio di astrazione dell'Intelligenza Artificiale (AIaaS privato).
**Componenti di ARIA Core:**
- **Local GPU Backends**: Qwen3-TTS, Fish Speech, Image Generators. (Caricano i pesi in VRAM e usano CUDA).
- **NEW: Cloud Backends**: Moduli Python (es. `GeminiCloudBackend`) che non pesano sulla GPU ma sulla scheda di rete. Usano le SDK ufficiali per interrogare modelli hosted.
- **Global Rate Limiter (GeminiRateLimiter)**: Integrato nel cuore di ARIA. Quando 3 app diverse (es. DIAS, un Bot Telegram, un'app di mail parsing) chiedono l'uso di Gemini, il Rate Limiter di ARIA codifica le richieste rispettando i 30 secondi di cooldown globale per non farsi bannare l'API Key.
- **Quota Manager**: ARIA tiene traccia di quanti token sono stati consumati oggi e blocca le code se il budget preimpostato è esaurito.

### 2.3 I Client Applicativi (LXC 190 / 201 - "Dumb Clients")
Esempio: **DIAS** (Distributed Immersive Audiobook System).
- **Stato Attuale**: DIAS (Stage B e C) possiede la `GEMINI_API_KEY`, ha una logica complessa di "lockout" per evitare Errori 429 da Google.
- **Stato Futuro**: DIAS perde la chiave. Scrive semplicemente un task sulla coda `gpu:queue:llm:gemini-flash` di LXC 120 contenente il proprio JSON (MacroAnalyer / SceneDirector), ed entra in ascolto in `BRPOP` sulla coda di risposta `gpu:result:dias:{job_id}`. Esattamente come fa oggi per lo Stage D (TTS).
- **Vantaggi del "Dumb Client"**: Qualsiasi modifica commerciale o tecnica di OpenAI o Google richiederà di aggiornare **SOLO** il codice di ARIA.

---

## 3. Deployment e Filosofia Operativa

L'approccio modulare garantisce che ogni sistema sia rimpiazzabile, debuggabile offline, e scalabile.

### Deploy Workflow (Il Caso Ideale)
Scenario: Acquisto di un nuovo hardware (es. MiniPC per ospitare Agent Apps).
1. Si installa Proxmox e si crea un LXC base.
2. Si clona il repo (es. `git clone nh-mini/dias`).
3. Si configura il file `.env` dell'app **unicamente con le coordinate del "Mondo Esterno"**, ovvero l'IP di Redis (LXC 120) e i path dei mount fisici.
4. L'app si avvia (es. `python start_pipeline.py`).
5. **Autoregolazione**: L'app si accorge che c'è lavoro, lo scarica su LXC 120. Se il PC Gaming (ARIA) è spento in quel momento, il lavoro si accumula. Quando ARIA viene acceso e "montato", vede la coda, la consuma a scaglioni (rispettando il rate limit Google) e le risposte piovono gradualmente giù al MiniPC.
**Nulla si perde, nulla muore di Timeout.**

---

## 4. Nuovi Contratti di Servizio (Code e Naming Convention)

### Ristrutturazione delle Code Redis
Per abbracciare questa visione, i prefissi si "universalizzano":
- `global:queue:local:{model_type}:{model_id}` -> Inferenza su GPU interna ad ARIA.
- `global:queue:cloud:{provider}:{model_id}` -> Inferenza remota pre-filtrata da ARIA. (Esempio: `global:queue:cloud:google:gemini-3.1-flash`)

### Standardizzazione del Payload (Task di ARIA)
```json
{
  "job_id": "uuid-1234-abcd",
  "client_id": "dias-minipc", // Per routing della risposta
  "type": "cloud", // 'local' o 'cloud'
  "provider": "google",
  "model_id": "gemini-flash",
  "payload": {
    "system_instruction": "Sei l'analyzer...",
    "messages": [
      {"role": "user", "content": "Analizza questo blocco testuale..."}
    ],
    // Verrà mappato dinamicamente nel CloudBackend
    "temperature": 0.4 
  },
  "callback_key": "global:result:dias:uuid-1234-abcd"
}
```

---

## 5. Roadmap e Piano d'Azione (Sviluppo Futuro)

1. **Creazione Common Libs (Optional)**: Eliminata a favore della teoria dei monolitici isolati. Le librerie comuni (es. Redis Wrapper) verranno duplicate o fornite come package Python privato (`nh-mini-core`) se strettamente necessario, ma è preferibile mantenere ogni repository 100% autosufficiente e limitarsi al "Contratto JSON" come standard di interfaccia.
2. **Implementazione CloudBackend in ARIA**: Scrittura di un modulo in `aria_node_controller/backends/` che non chiama PyTorch, ma carica la libreria di `google-generativeai`.
3. **Migrazione GeminiRateLimiter**: Spostamento della logica avanzata di Lockout/Pacing scritta in DIAS all'interno del layer "QueueManager" di ARIA.
4. **Refactoring DIAS Stage B & Stage C**: Rimozione delle dipendenze API e adozione del client HTTP/Redis standard già usato per lo Stage D.
5. **Release v2.0**: Completo disaccoppiamento raggiunto. Nessuna app locale contatterà mai più il Cloud in autonomia.

---
*Documento tecnico preparato per il brainstorming multiprocesso (Claude / Studio AI).*
