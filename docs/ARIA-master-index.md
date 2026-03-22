# ARIA Master Index — Catalogo Documentazione e Asset

Questo documento fornisce una mappa completa dei file di configurazione, documentazione e ambienti del progetto ARIA.

---

## 1. Architettura e Visione (Core)
*   [**ARIA-blueprint.md**](ARIA-blueprint.md): Il documento "madre". Descrive l'architettura, i flussi asincroni e la filosofia agnostica.
*   [**master-roadmap.md**](master-roadmap.md): Stato di avanzamento del progetto e obiettivi futuri.
*   [**ARIA-network-interface.md**](ARIA-network-interface.md): Specifiche tecniche del bus Redis e dei protocolli di rete.

## 2. Configurazione Operativa
*   [**backends_manifest.json**](../aria_node_controller/config/backends_manifest.json): **L'Anagrafe dei Backend**. Contiene porte, comandi e percorsi env per ogni modello.
*   [**node_settings.json.example**](../node_settings.json.example): Template per la configurazione locale del nodo (Redis IP, VRAM limits).
*   [**Avvia_Tutti_Server_ARIA.bat**](../Avvia_Tutti_Server_ARIA.bat): Entry point principale per l'avvio su Windows.

## 3. Ambienti e Provisioning
*   [**bootstrap_aria.ps1**](../scripts/installer/bootstrap_aria.ps1): Lo script magico. Ripristina tutti gli ambienti Conda e le utility (SoX) in un clic.
*   [**environments-setup.md**](environments-setup.md): Guida tecnica alla gerarchia degli ambienti Python.
*   **Requirements (`requirements/*.txt`)**:
    *   `core.txt`: Dipendenze per l'Orchestratore (Pystray, Redis, PIL).
    *   `qwen3tts.txt`: Specifiche per il modello Qwen3.
    *   `cloud.txt`: Dipendenze per i worker Gemini/Cloud.
*   **Templates (`envs/templates/*.yml`)**: Definizioni dichiarative degli ambienti Conda per la ricostruzione.

## 4. Specifiche Backend (Deep Dives)
*   [**qwen3-tts-backend.md**](qwen3-tts-backend.md): Dettagli sull'integrazione di Qwen3-1.7B.
*   [**fish-tts-backend.md**](fish-tts-backend.md): Dettagli sull'integrazione di Fish Speech S1.
*   [**qwen3.5-35b-llm-backend.md**](qwen3.5-35b-llm-backend.md): Specifiche per il backend LLM locale.
*   [**hybrid-tts-architecture.md**](hybrid-tts-architecture.md): Come ARIA gestisce il mix tra modelli locali e cloud.

## 5. Script di Supporto
*   `scripts/qwen3/download_qwen3_model.bat`: Utility per scaricare i pesi da HuggingFace.
*   `scripts/voice_prepper.py`: Tool per normalizzare i campioni voce della libreria.

---

## Suggerimenti per l'Organizzazione
1.  **SOT (Source of Truth)**: LXC 190 deve essere sempre allineato con questo indice.
2.  **Manutenzione**: Ogni nuovo backend DEVE essere registrato nel `backends_manifest.json` e avere un suo template `.yml` in `envs/templates/`.
3.  **Portabilità**: Evitare assolutamente path assoluti `C:\...` nei documenti; usare sempre variabili d'ambiente o percorsi relativi alla root.
