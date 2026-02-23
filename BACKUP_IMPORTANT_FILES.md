# Backup File Importanti - ARIA Legacy

Questi file sono stati preservati per reference ma il progetto sta migrando a una nuova architettura:

## File da Conservare (per reference)
- `aria_server/queue_manager.py` - Redis bridge (potrebbe essere riutilizzato)
- `aria_server/result_writer.py` - Gestione risultati (potrebbe essere riutilizzato)
- `tests/` - Test unitari (da adattare)
- `docs/` - Documentazione

## File che Diventano Obsoleti
- `main.py` - Sarà rimpiazzato da Orpheus-FastAPI
- `Dockerfile*` - Tutti i Dockerfile attuali
- `docker-compose*.yml` - Tutti i compose attuali
- `requirements*.txt` - Requirements cambieranno completamente
- `aria_server/backends/` - Backend attuali non più necessari

## Nuova Architettura
Il progetto migrerà a:
- llama.cpp server come backend inferenza
- Orpheus-FastAPI come wrapper TTS
- 2 container Docker separati
- CUDA 12.8 per RTX 5060 Ti compatibility