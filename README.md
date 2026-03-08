# ARIA - Adaptive Resource for Inference and AI
## Distributed GPU Inference Broker per Homelab

> Piattaforma di inferenza AI privata per reti domestiche.
> ARIA trasforma il PC Gaming con GPU in un servizio AI condiviso sulla LAN.

---

## Architettura "Linked-but-Independent"

Il sistema è progettato per essere **totalmente disaccoppiato** e **agnostico rispetto all'infrastruttura**:

```
BRAIN NODE                REDIS MESH            WORKER NODE (GPU)
(DIAS Engine)   ◄──►   Infrastructure   ◄──►   ARIA Orchestrator
                       (Message Bus)           (RTX 5060 Ti)
```

**Principio fondamentale**: La scoperta dei nodi è dinamica via Heartbeat. I client non hanno bisogno di conoscere l'IP del worker, solo l'indirizzo del bus Redis.

### 📜 Specifiche di Rete
Per il "Contratto" ufficiale di comunicazione (code Redis, Heartbeat, Payload), consultare:
👉 **[docs/ARIA-network-interface.md](docs/ARIA-network-interface.md)**

---

## Documentazione Principale

| File | Contenuto |
|---|---|
| `docs/ARIA-blueprint.md` | **Documento principale** — Filosofia di design e ciclo di vita modelli |
| `docs/ARIA-network-interface.md` | **Il Contratto** — Specifiche Redis, Heartbeat e Idempotenza |
| `docs/environments-setup.md` | **Guida Ambienti** — Setup Python isolato per GPU Windows |
| `docs/master-roadmap.md` | Stato completo del progetto e fasi future |

---
*ARIA Project — NH-Mini Philosophy*