# ARIA Master Index — Bussola della Documentazione

Questo documento è il punto di partenza per chiunque (Agent o Umano) voglia interagire con ARIA. Definisce la gerarchia dei documenti e distingue tra ciò che è **Realtà Operativa (SOT)** e ciò che è **Design Futuro (Blueprint)**.

---

## 1. Il Sacro Graal (Source of Truth - SOT)
Questi documenti descrivono il sistema **così come funziona oggi**. Ogni Agent deve attenersi a queste specifiche.

*   [**ARIA-API-Contract.md**](ARIA-API-Contract.md): **IL CONTRATTO**. Nomenclatura code Redis, schema dei payload JSON e ID modelli.
*   [**backends_manifest.json**](../aria_node_controller/config/backends_manifest.json): **L'ANAGRAFE BACKEND**. Porte, comandi e versioni degli ambienti reali.

---

## 2. Design e Visione (Blueprint)
Documentazione di alto livello che descrive la filosofia e le evoluzioni future.

*   [**ARIA-blueprint.md**](ARIA-blueprint.md): Descrive l'architettura macroscopica e la logica di business. 
    *   *Nota: Se una sezione del Blueprint contraddice il Contratto, il Contratto vince.*
*   [**master-roadmap.md**](master-roadmap.md): Obiettivi a breve e lungo termine.

---

## 3. Guide Operative (How-To)
Istruzioni pratiche per installazione e manutenzione.

*   [**environments-setup.md**](environments-setup.md): Guida alla creazione degli ambienti Conda isolati.
*   [**qwen3-tts-backend.md**](qwen3-tts-backend.md): Dettagli specifici sul backend Qwen3.
*   [**fish-tts-backend.md**](fish-tts-backend.md): Dettagli specifici sul backend Fish Speech.

---

## 4. Archivio (Legacy)
Documenti obsoleti o superati, mantenuti solo per riferimento storico. **NON USARE PER NUOVE IMPLEMENTAZIONI.**

*   `docs/ARIA-network-interface.md`: Sostituito integralmente da `ARIA-API-Contract.md`.

---
*Ultimo Aggiornamento: 2026-03-23*
