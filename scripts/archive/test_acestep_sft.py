import os
import sys
import time
import torch
import soundfile as sf
import numpy as np
import gc

# --- CONFIGURAZIONE PERCORSI ---
# Repository ufficiale clonato
acestep_root = r"C:\Users\Roberto\aria\ACE-Step-1.5"
# Pesi XL certificati
dit_path = r"C:\Users\Roberto\aria\data\assets\models\acestep-v15-xl-sft"
# LM Standard
lm_path = r"C:\Users\Roberto\aria\data\assets\models\acestep-5Hz-lm-1.7b\acestep-5Hz-lm-1.7B"
# Output
output_path = r"C:\Users\Roberto\aria\tmp\acestep_XL_blackwell_FINALE.wav"

# --- INIEZIONE SORGENTE UFFICIALE ---
if acestep_root not in sys.path:
    sys.path.append(acestep_root)

# Importazione della pipeline XL certificata dal repository ufficiale
try:
    # Registra i modelli Qwen3 dinamicamente se necessario
    from transformers import AutoModel, AutoTokenizer
    # Ora che nano-vllm è installato, questo non dovrebbe fallire
except ImportError as e:
    print(f"Errore inizializzazione stack Transformers/NanoVLLM: {e}")
    sys.exit(1)

# --- OTTIMIZZAZIONE BLACKWELL (sm_120) ---
os.environ["CUDA_FORCE_PTX_JIT"] = "1"
os.environ["ACESTEP_DISABLE_QUANTIZATION"] = "1" # Usiamo FP16 nativo su sm_120
torch.backends.cuda.matmul.allow_tf32 = True

def test_acestep_sft_sota_turbo_mode():
    prompt_text = "A cinematic ambient PAD with deep sub-bass and ethereal crystal textures, 44.1kHz high-fidelity sound."
    
    print("--- ARIA ACE-Step 1.5 XL (SOTA + Turbo Stack sm_120) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # 1. Caricamento LM (Stage A) su CPU
        print(f"Caricamento LM (Qwen3-1.7B) su CPU...")
        lm_tokenizer = AutoTokenizer.from_pretrained(lm_path)
        lm_model = AutoModel.from_pretrained(lm_path, torch_dtype=torch.float16, device_map="cpu")
        
        inputs = lm_tokenizer(prompt_text, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = lm_model(**inputs, output_hidden_states=True)
            text_hidden_states = outputs.hidden_states[-1].detach().clone()
            text_attention_mask = inputs["attention_mask"].to(torch.float16).detach().clone()
        
        del lm_model
        gc.collect()
        torch.cuda.empty_cache()
        print("LM completato. Hidden states pronti.")
        
        # 2. Caricamento DiT (Stage B) - SOTA + TURBO MODE
        print(f"Caricamento DiT XL (4B) in float16 nativo su {device}...")
        start_dit = time.time()
        
        # Ora AutoModel userà i moduli Qwen3 appena installati in nano-vllm
        dit_model = AutoModel.from_pretrained(
            dit_path, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            device_map=None,          # Caricamento manuale RAM per Blackwell
            low_cpu_mem_usage=False   
        )
        print(f"Modello istanziato. Trasferimento in VRAM...")
        dit_model = dit_model.to(device)
        print(f"DiT XL SOTA pronto in {time.time() - start_dit:.2f}s.")

        # 3. Generazione (Parametri Standard Pipeline XL)
        print("Inizio generazione audio (50 steps diffusion)...")
        start_gen = time.time()
        
        batch_size = text_hidden_states.shape[0]
        dummy_lyric = torch.zeros(batch_size, 1, 2048, dtype=torch.float16, device=device)
        dummy_lyric_mask = torch.zeros(batch_size, 1, dtype=torch.float16, device=device)
        dummy_refer = torch.zeros(batch_size, 1, 64, dtype=torch.float16, device=device)
        dummy_refer_mask = torch.zeros(batch_size, dtype=torch.long, device=device)
        dummy_latents = torch.zeros(batch_size, 250, 192, dtype=torch.float16, device=device) 
        dummy_chunk_mask = torch.ones(batch_size, 250, dtype=torch.float16, device=device)
        is_covers = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            audio_output = dit_model.generate_audio(
                text_hidden_states=text_hidden_states.to(device),
                text_attention_mask=text_attention_mask.to(device),
                lyric_hidden_states=dummy_lyric,
                lyric_attention_mask=dummy_lyric_mask,
                refer_audio_acoustic_hidden_states_packed=dummy_refer,
                refer_audio_order_mask=dummy_refer_mask,
                src_latents=dummy_latents,
                chunk_masks=dummy_chunk_mask,
                is_covers=is_covers,
                infer_steps=50,
                diffusion_guidance_scale=4.5 
            )
        
        print(f"Generazione completata in {time.time() - start_gen:.2f}s.")
        
        print("Esportazione audio (.wav)...")
        wav_data = dit_model.detokenize(audio_output)
        wav_np = wav_data.cpu().numpy().squeeze()
        sf.write(output_path, wav_np, 44100)
        print(f"[SUCCESS] Test SOTA + Turbo completato: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Test fallito: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_acestep_sft_sota_turbo_mode()
