# ARIA Sound Factory: Setup TorchCodec (NVIDIA Blackwell sm_120)

Questa guida documenta il processo riproducibile per installare `torchcodec` su sistemi Windows con GPU RTX Serie 50 (architettura Blackwell) evitando conflitti di dipendenze.

## Stack di Compatibilità (Aprile 2026)
| Componente | Versione | Nota |
| :--- | :--- | :--- |
| **Python** | 3.10+ | |
| **PyTorch** | 2.11.0+cu128 | Tassativo per supporto `sm_120` |
| **CUDA Toolkit** | 12.8 | |
| **FFmpeg** | 8.0.1+ | Deve essere la versione "shared" (via conda-forge) |
| **TorchCodec** | 0.11.0 | |
| **Quantizzazione** | `torchao 0.12.0` | **Indispensabile per stabilità DiT XL** |
| **Configurazione** | Flat TOML | Evita crash nel parsing CLI su sm_120 |

---

## 1. Preparazione Ambiente (Conda)
Assicurarsi che FFmpeg sia installato nel prefisso dell'ambiente per fornire le librerie condivise necessarie al decoder.

```cmd
conda install ffmpeg -c conda-forge -y
```

## 2. Installazione "Hardened"
Per evitare che Pip tenti di scaricare versioni di PyTorch non compatibili con Blackwell durante l'installazione di TorchCodec, usare il flag `--no-deps`.

```cmd
# Assicurarsi che torch sia già presente (es. 2.11.0+cu128)
pip install torchcodec --no-deps
```

## 3. Integrazione in Python (Windows DLL Pathing)
Su Windows, le DLL di FFmpeg installate via Conda si trovano in `Library\bin`. È necessario informare Python della loro posizione prima di importare `torchcodec`.

```python
import os
import sys

# Aggiunge il path delle DLL di Conda al search path di Windows
conda_bin = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.exists(conda_bin):
    os.add_dll_directory(conda_bin)

import torchcodec
from torchcodec.decoders import AudioDecoder
```

## 4. Stabilizzazione DiT (ACE-Step XL 4B)
Su GPU Blackwell da 16GB (Tier 6a), il solo uso di PyTorch 2.11 non garantisce la stabilità per modelli grandi. È necessario:

1.  **Quantizzazione INT8**: Ridurre il peso del modello DiT XL (~9GB) tramite `int8_weight_only`.
2.  **Configurazione Flat**: Evitare sezioni annidate nei file TOML (es. `[models]`) per garantire che i parametri siano letti correttamente nel namespace globale della CLI.
3.  **Ambiente**: `ACESTEP_COMPILE_MODEL="0"` per evitare fallimenti di compilazione Triton.

## 5. Troubleshooting
- **ImportError: DLL load failed**: Spesso indica che FFmpeg non è nel PATH o non è stato invocato `os.add_dll_directory`.
- **sm_120 Not Supported**: Indica che `torch` è stato degradato a una versione senza supporto Blackwell. Reinstallare torch con l'index URL `cu128`.
- **UnicodeEncodeError (Windows)**: Evitare caratteri non-ASCII (emoji) nei print finali di successo per prevenire crash su console CP1252.

---
*Status: Verified on RTX 5060 Ti (sm_120) - Aprile 2026*
