\# 📘 GUIDA TECNICA: PRODUZIONE E MIXAGGIO DI AUDIO DRAMA (STILE BBC/STAR WARS)  
\*Pipeline locale ottimizzata per 16 GB VRAM | Integrazione con time grid esistente | Asset modulari \+ mixaggio broadcast-ready\*

\---

\#\# 1\. 🗂️ ARCHITETTURA DEGLI ASSET & SINCRONIZZAZIONE CON TIME GRID

Per ottenere un risultato professionale, gli asset \*\*non devono essere generati come tracce complete\*\*, ma come \*\*mattoni indipendenti mappati sulla tua time grid\*\*. Questo permette al mixatore (automatico o manuale) di comporre dinamicamente la scena.

\#\#\# Struttura cartelle consigliata  
\`\`\`  
project/  
├── time\_grid.json          \# Tua griglia esistente (scene, durate, timestamp)  
├── voices/                 \# Output Qwen3TTS (narrazione \+ dialoghi)  
├── music\_stems/  
│   ├── {mood}\_pad.wav  
│   ├── {mood}\_bass.wav  
│   ├── {mood}\_harmony.wav  
│   ├── {mood}\_lead.wav  
│   └── {mood}\_perc.wav  
├── ambient/  
│   ├── {location}\_bed.wav  \# Loop di 15-30s crossfaddati  
│   └── {location}\_layer\_X.wav  
├── sfx/  
│   ├── diegetic/  
│   ├── non\_diegetic/  
│   └── transitions/  
└── stings/  
    ├── sting\_reveal.wav  
    ├── sting\_curtain.wav  
    └── sting\_emotional.wav  
\`\`\`

\#\#\# Metadata obbligatorio per ogni asset  
Aggiungi al tuo \`time\_grid.json\` un campo \`assets\` per ogni blocco:  
\`\`\`json  
{  
  "scene\_id": "SC04",  
  "start\_ms": 45200,  
  "duration\_ms": 18500,  
  "mood": "tension\_rising",  
  "intensity": 0.75,  
  "assets": {  
    "music": \["tension\_pad.wav", "tension\_bass.wav"\],  
    "ambient": \["corridor\_bed.wav"\],  
    "sfx": \["footsteps\_02.wav", "door\_heavy\_close.wav"\],  
    "sting": null,  
    "reverb\_preset": "large\_hall\_wet",  
    "spatial": {"dialogue\_L": 0.3, "dialogue\_R": \-0.2}  
  }  
}  
\`\`\`  
Questo schema permette al motore di mix di \*\*attivare/disattivare layer\*\* in modo preciso, senza sovrapposizioni caotiche.

\---

\#\# 2\. 🎵 PRODUZIONE MUSICALE: DA AI A COLONNA SONORA NARRATIVA

Nessun modello AI genera ore di musica coerente. La soluzione professionale è \*\*stem-based \+ variazione procedurale\*\*.

\#\#\# 🔹 Step 1: Generazione Stem (15-30s per mood)  
Usa \`MusicGen-small\` o \`ACE-Step\` con prompt strutturati per ruolo:  
| Stem | Prompt Template | Durata | Note |  
|------|----------------|--------|------|  
| \`pad\` | \`cinematic ambient pad, slow attack, no rhythm, sustained strings, mood: \[mood\]\` | 30s | Base armonica, loopabile |  
| \`bass\` | \`low cinematic drone, sub bass, slow movement, tension underscore\` | 20s | Fondale, non melodico |  
| \`harmony\` | \`orchestral strings section, legato, emotional progression, key: D minor\` | 25s | Corpo armonico |  
| \`lead\` | \`solo instrument motif, sparse, cinematic, leitmotif style\` | 15s | Tema riconoscibile |  
| \`perc\` | \`cinematic taiko hits, brushed snares, slow ostinato, no melody\` | 20s | Ritmo narrativo |

\*\*Genera 3 varianti per mood\*\* (es. \`tension\_v1.wav\`, \`tension\_v2.wav\`). Usa seed fissi per coerenza.

\#\#\# 🔹 Step 2: Variazione Procedurale (Python)  
Evita il "reinventare" la musica ad ogni scena. Applica trasformazioni matematiche agli stem:  
\`\`\`python  
\# music\_variator.py  
import librosa  
import soundfile as sf  
import numpy as np

def vary\_stem(input\_path, output\_path, pitch\_shift=0, time\_stretch=1.0, highcut\_freq=None):  
    y, sr \= librosa.load(input\_path, sr=32000)  
    if pitch\_shift \!= 0:  
        y \= librosa.effects.pitch\_shift(y, sr=sr, n\_steps=pitch\_shift)  
    if time\_stretch \!= 1.0:  
        y \= librosa.effects.time\_stretch(y, rate=time\_stretch)  
    if highcut\_freq:  
        \# EQ high-cut simulato via FFT (per intensità bassa)  
        D \= librosa.stft(y)  
        freqs \= librosa.fft\_frequencies(sr=sr)  
        mask \= freqs\[:, None\] \< highcut\_freq  
        D \= np.where(mask, D, D \* 0.3)  
        y \= librosa.istft(D)  
    sf.write(output\_path, y, sr)  
\`\`\`  
\*\*Regole di mapping:\*\*  
\- \`intensity \< 0.3\` → solo \`pad\` \+ \`bass\`, high-cut a 1kHz  
\- \`0.3 ≤ intensity \< 0.7\` → aggiungi \`harmony\`, pitch ±1 semitono  
\- \`intensity ≥ 0.7\` → tutti gli stem \+ \`lead\`, time-stretch \+5% per urgenza  
\- Cambio scena → crossfade 2-3s tra varianti

\#\#\# 🔹 Step 3: Leitmotif & Coerenza Tematica  
\- Scegli un \*\*motivo base\*\* (es. 4 note) e rigeneralo con \`lead\` prompt variando solo strumento/tonalità.  
\- Mantieni la \*\*tonalità di riferimento\*\* (es. D minor) per tutte le scene dello stesso arco narrativo.  
\- Usa \*\*silenzi strategici\*\* (200-500ms) prima dei dialoghi cruciali: la BBC li usa per aumentare l'attenzione.

\---

\#\# 3\. 🔊 SFX, AMBIENT & STING: PIPELINE DI CREAZIONE

\#\#\# 🌫️ Ambient Beds  
\- Genera con \`AudioGen\` o usa librerie (Freesound, BBC SFX).  
\- Struttura: \`room\_tone.wav\` \+ \`distant\_element.wav\` \+ \`weather/city.wav\`.  
\- \*\*Loop perfetto\*\*: taglia a zero-crossing, applica \`crossfade\` di 1-2s alle estremità.  
\- Mappa alla location: \`corridor\`, \`forest\`, \`cockpit\`, \`archive\`.

\#\#\# 💥 SFX (Diegetici & Non-Diegetici)  
| Tipo | Tecnica | Esempio |  
|------|---------|---------|  
| Diegetico | Layering: Impact \+ Body \+ Tail | \`door\_close\_impact.wav\` \+ \`wood\_creak.wav\` \+ \`room\_reverb\_tail.wav\` |  
| Non-diegetico | Sintetico/psicologico | \`whoosh\_transition.wav\`, \`sub\_drop.wav\` |  
| Foley-style | Registra oggetti reali o usa AI con prompt fisici | \`fabric\_rustle.wav\`, \`paper\_turn.wav\` |

\*\*Regola d'oro\*\*: ogni SFX deve avere \*\*spazializzazione corrispondente all'ambiente\*\* della scena.

\#\#\# 🎼 Sting (Transizioni Musicali)  
\- Durata: 1-3 secondi  
\- Uso: cambi scena, rivelazioni, colpi di scena  
\- Generazione: prompt mirati → \`sting\_reveal.wav\`, \`sting\_curtain.wav\`, \`sting\_emotional.wav\`  
\- Processing: applica \`reverse\` \+ \`lowpass swell\` \+ \`impact hit\` per transizioni cinematografiche.

\---

\#\# 4\. 🎚️ MIXAGGIO STILE BBC/STAR WARS: TECNICHE PROFESSIONALI

\#\#\# 📌 Principi Fondamentali  
1\. \*\*Dialogo sempre al centro\*\*: \-3dB a \-6dB sopra musica/SFX nei momenti quieti.  
2\. \*\*Ducking automatico\*\*: musica scende di 6-9dB durante la voce, torna su nelle pause.  
3\. \*\*Spazializzazione narrativa\*\*: reverb e panning definiscono distanza, luogo, stato emotivo.  
4\. \*\*Dynamic Range controllato\*\*: nessun clipping, loudness standardizzato, picchi gestiti.

\#\#\# ⚙️ Catena di Mix (Automatizzabile)  
\`\`\`  
Dialogue Track → High-Pass 80Hz → Compressor (3:1, \-18dB threshold) → Limiter  
Music Stems   → Sidechain (key: dialogue) → EQ (low-cut 60Hz, high-shelf \-3dB @12kHz) → Bus  
SFX/Ambient   → Reverb Send (per environment) → Panning → Bus  
Master Bus    → Multiband Compressor (leggero) → Loudness Normalization (-16 LUFS) → Limiter (-1dBTP)  
\`\`\`

\#\#\# 🎛️ Parametri Broadcast-Ready  
| Parametro | Valore | Note |  
|-----------|--------|------|  
| Loudness integrata | \-16 LUFS (podcast) / \-23 LUFS (broadcast) | ITU-R BS.1770-4 |  
| True Peak | ≤ \-1.0 dBTP | Evita clipping su DAC/codec |  
| Attack Ducking | 10-30 ms | Rapido per non coprire consonanti |  
| Release Ducking | 150-300 ms | Naturale, non "pompa" |  
| Reverb Pre-delay | 20-40 ms | Mantiene intelligibilità |  
| Stereo Width | ≤ 80% su master | Compatibilità mono |

\---

\#\# 5\. 🤖 AUTOMAZIONE DEL MIX FINALE (PYTHON \+ FFMPEG)

Dato che hai già \`time\_grid.json\` e i wav, ecco un mixer automatico pronto.

\`\`\`python  
\# drama\_mixer.py  
import json  
import subprocess  
import os  
from pathlib import Path

class DramaMixer:  
    def \_\_init\_\_(self, grid\_path, output\_dir="final"):  
        with open(grid\_path) as f: self.grid \= json.load(f)\["scenes"\]  
        Path(output\_dir).mkdir(exist\_ok=True)  
        self.output\_dir \= Path(output\_dir)

    def build\_ffmpeg\_command(self):  
        \# Costruisce complessivamente il filtro audio  
        inputs \= \[\]  
        filters \= \[\]  
          
        \# 1\. Carica tracce dialogo (già allineate temporalmente)  
        for i, scene in enumerate(self.grid):  
            for block in scene.get("blocks", \[\]):  
                if block\["type"\] in \["dialogue", "narration"\]:  
                    inputs.append(f"-i {block\['audio\_path'\]}")  
                    filters.append(f"\[{len(inputs)-1}:a\]adelay={block\['start\_ms'\]}|{block\['start\_ms'\]}\[d{i}\]")  
          
        \# 2\. Mixa dialoghi  
        dialogue\_mix \= "+".join(\[f"\[d{i}\]" for i in range(len(inputs))\])  
        filters.append(f"{dialogue\_mix}amix=inputs={len(inputs)}:normalize=0\[dialogue\]")  
          
        \# 3\. Musica (sidechain ducking)  
        inputs.append("-i music\_stems/mixed\_score.wav")  
        music\_idx \= len(inputs) \- 1  
        filters.append(f"\[dialogue\]\[{music\_idx}:a\]sidechaincompress=threshold=0.003:ratio=4:attack=0.02:release=0.25\[music\_ducked\]")  
          
        \# 4\. Unione finale \+ loudness  
        filters.append(f"\[music\_ducked\]\[dialogue\]amix=inputs=2:normalize=0\[master\]")  
        filters.append(f"\[master\]loudnorm=I=-16:TP=-1.5:LRA=11:print\_format=json\[out\]")  
          
        cmd \= f"ffmpeg {' '.join(inputs)} \-filter\_complex '{'; '.join(filters)}' \-map '\[out\]' \-c:a pcm\_s16le {self.output\_dir}/final\_drama.wav"  
        return cmd

    def run(self):  
        cmd \= self.build\_ffmpeg\_command()  
        print("🎛️ Eseguendo mixaggio...")  
        subprocess.run(cmd, shell=True, check=True)  
        print("✅ Mix completato.")  
\`\`\`  
\*\*Nota\*\*: per progetti \>2 ore, renderizza per capitolo e unisci con \`ffmpeg \-f concat \-safe 0 \-i list.txt \-c copy final\_full.wav\`.

\---

\#\# 6\. ✅ CHECKLIST QUALITÀ & OTTIMIZZAZIONE 16GB VRAM

\#\#\# 🔍 Validazione Pre-Export  
\- \[ \] Nessun dialogo coperto da musica \> \-6dB RMS  
\- \[ \] Transizioni senza click (zero-crossing o crossfade ≥500ms)  
\- \[ \] Loudness integrato: \-15.5 / \-16.5 LUFS  
\- \[ \] True Peak: ≤ \-1.0 dBTP  
\- \[ \] Reverb coerente per ambiente (non "wash" generico)  
\- \[ \] SFX diegetici hanno direzione/panning plausibile

\#\#\# 💾 Gestione VRAM (16GB)  
| Fase | Strategia |  
|------|-----------|  
| Generazione Stem | 1 mood alla volta, \`torch.cuda.empty\_cache()\` dopo ogni run |  
| TTS | Usa \`qwen3tts\` in modalità \`int4\` o CPU-offload per voci secondarie |  
| Mixing | 100% CPU/RAM. \`pydub\` \+ \`ffmpeg\` non usano GPU |  
| Cache | Salva tutti gli stem in \`stems/\`. Rigenera solo se cambi mood/theme |  
| Monitoraggio | \`nvtop\` o \`nvitop\` per verificare leak. Usa \`torch.no\_grad()\` |

\#\#\# 📈 Fallback & Sicurezza  
\- Se un stem AI è instabile: sostituisci con loop royalty-free \+ EQ match  
\- Se il ducking pompa: aumenta \`release\` a 300ms, riduci \`ratio\` a 3:1  
\- Se la loudness è inconsistente: normalizza prima del mix con \`ffmpeg \-af loudnorm=I=-16:TP=-1.5:linear=true\`

\---

\#\# 📎 APPENDICI

\#\#\# A. Prompt AI per Stem Musicali (MusicGen/Ace-Step)  
\`\`\`  
\[Role\] cinematic underscore, \[Mood\] tense/wonder/calm, \[Instrument\] strings/pads/brass,   
\[Structure\] sparse, no melody, loopable 30s, key: \[tonality\], tempo: \[bpm\],   
\[Production\] dry, no reverb, high quality, broadcast ready  
\`\`\`

\#\#\# B. Schema JSON Esteso (per il tuo time grid)  
\`\`\`json  
{  
  "project": "BBC\_Drama\_Project",  
  "target\_loudness": \-16,  
  "scenes": \[  
    {  
      "id": "SC01",  
      "start\_ms": 0,  
      "duration\_ms": 45000,  
      "location": "corridor",  
      "mood": "tension",  
      "intensity": 0.6,  
      "reverb\_preset": "large\_hall",  
      "blocks": \[  
        {"type": "narration", "start\_ms": 1200, "duration\_ms": 4200, "audio\_path": "voices/narr\_01.wav"},  
        {"type": "dialogue", "character": "Aldrin", "start\_ms": 6500, "duration\_ms": 3800, "audio\_path": "voices/ald\_01.wav"},  
        {"type": "sfx", "start\_ms": 44000, "audio\_path": "sfx/door\_heavy.wav", "pan": 0.5}  
      \]  
    }  
  \]  
}  
\`\`\`

\#\#\# C. Toolchain Consigliata (Locale)  
| Compito | Strumento | Note |  
|---------|-----------|------|  
| Generazione Stem | \`facebook/musicgen-small\` | 300M params, \~4.5GB VRAM |  
| Variazione Audio | \`librosa\`, \`numpy\` | CPU, leggero |  
| Mixing/Rendering | \`ffmpeg\`, \`pydub\` | CPU, broadcast-ready |  
| Loudness/Analisi | \`ffmpeg \-af loudnorm\`, \`ebuR128\` | Standard ITU |  
| Monitoraggio VRAM | \`nvitop\`, \`torch.cuda.empty\_cache()\` | Essenziale |

\---

\#\# 🚀 PROSSIMO PASSO OPERATIVO  
1\. \*\*Genera 3 stem per ogni mood\*\* presente nel tuo \`time\_grid.json\` (pad, bass, harmony).  
2\. \*\*Applica il variatore procedurale\*\* per mappare intensità → layer attivi.  
3\. \*\*Esegui il mixer automatico\*\* su 1 capitolo di test (5-10 min).  
4\. \*\*Valida loudness e ducking\*\*, regola parametri, poi scala all'intero libro.

Se vuoi, ti preparo:  
\- Uno script completo \`stem\_generator.py\` con gestione VRAM e cache  
\- Un template \`time\_grid.json\` popolato con un capitolo di esempio  
\- Una configurazione \`ffmpeg\` ottimizzata per loudness \-16 LUFS \+ sidechain ducking preciso

Dimmi quale preferisci e te lo invio pronto per il copia-incolla nella tua pipeline.  
