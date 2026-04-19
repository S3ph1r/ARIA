"""
Test manuale: invia UN singolo payload ad ARIA e aspetta il callback.
Ordine consigliato: 1=AMB, 2=SFX stone, 3=SFX body, 4=SFX explosion, 5=STING, 0=PAD

Uso: python test_single_asset.py <indice>
"""
import json, sys, time, os
import redis

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = 6379
QUEUE      = "aria:q:mus:local:acestep-1.5-xl-sft:dias"

PAYLOADS = [
    # 0 — PAD (relay 470s, run_demucs=True, testare per ultimo)
    {
        "canonical_id": "pad_retro_scifi_tension_01",
        "type": "pad",
        "timeout_s": 7200,
        "callback_key": "aria:c:dias:d2-pad-1f227b79b8",
        "redis_task": {
            "job_id": "d2-pad-1f227b79b8",
            "client_id": "dias",
            "model_type": "mus",
            "model_id": "acestep-1.5-xl-sft",
            "callback_key": "aria:c:dias:d2-pad-1f227b79b8",
            "timeout_seconds": 7200,
            "payload": {
                "job_id": "d2-pad-1f227b79b8",
                "prompt": "cinematic underscore, no vocals, instrumental, 1970s orchestral, retro sci-fi, dark suspense, low strings, vintage analog synthesizer, dissonant brass ensemble, analog warmth, slow tempo, minor key, ominous, heavy metallic percussion, Bernard Herrmann influence, tense atmosphere",
                "lyrics": "[00:01.00] [Intro] Sparse cello drone, quiet vintage synthesizer textures, lonely atmosphere, no percussion\n[02:31.00] [Pre-Chorus] Layered analog strings, subtle dissonant brass swells, increasing tension, steady slow tempo\n[05:51.00] [Chorus] Rapidly thickening brass clusters, metallic industrial percussion builds, intense ominous harmony\n[07:01.00] [Outro] Fading metallic strikes, lingering low synthesizer note, decaying analog reverb, silence",
                "duration": 470.4,
                "seed": 42,
                "guidance_scale": 4.5,
                "inference_steps": 60,
                "output_style": "pad",
                "thinking": True,
                "run_demucs": True,
            }
        }
    },
    # 1 — AMB (single-shot 4s, primo test consigliato)
    {
        "canonical_id": "amb_enclosed_cave_01",
        "type": "amb",
        "timeout_s": 900,
        "callback_key": "aria:c:dias:d2-amb-8d5a86bbfb",
        "redis_task": {
            "job_id": "d2-amb-8d5a86bbfb",
            "client_id": "dias",
            "model_type": "mus",
            "model_id": "acestep-1.5-xl-sft",
            "callback_key": "aria:c:dias:d2-amb-8d5a86bbfb",
            "timeout_seconds": 900,
            "payload": {
                "job_id": "d2-amb-8d5a86bbfb",
                "prompt": "short transitional cue, establishing ambience, enclosed cave, stone walls, distant water dripping, low reverb, cold damp air, fade in 0.5s, fade out 1.5s, total 4s, non-looping, no music, no melody",
                "lyrics": "",
                "duration": 4.0,
                "seed": 42,
                "guidance_scale": 7.0,
                "inference_steps": 60,
                "output_style": "amb",
                "thinking": False,
                "run_demucs": False,
            }
        }
    },
    # 2 — SFX stone impact
    {
        "canonical_id": "sfx_impact_stone_01",
        "type": "sfx",
        "timeout_s": 900,
        "callback_key": "aria:c:dias:d2-sfx-74068b90c3",
        "redis_task": {
            "job_id": "d2-sfx-74068b90c3",
            "client_id": "dias",
            "model_type": "mus",
            "model_id": "acestep-1.5-xl-sft",
            "callback_key": "aria:c:dias:d2-sfx-74068b90c3",
            "timeout_seconds": 900,
            "payload": {
                "job_id": "d2-sfx-74068b90c3",
                "prompt": "sound effect, isolated, sharp, hand hitting solid rock, stone impact, deep resonance, no music, no reverb, no ambient, no echo, mono",
                "lyrics": "",
                "duration": 0.4,
                "seed": 42,
                "guidance_scale": 7.0,
                "inference_steps": 60,
                "output_style": "sfx",
                "thinking": False,
                "run_demucs": False,
            }
        }
    },
    # 3 — SFX body fall
    {
        "canonical_id": "sfx_impact_body_fall_01",
        "type": "sfx",
        "timeout_s": 900,
        "callback_key": "aria:c:dias:d2-sfx-5cc9973ff9",
        "redis_task": {
            "job_id": "d2-sfx-5cc9973ff9",
            "client_id": "dias",
            "model_type": "mus",
            "model_id": "acestep-1.5-xl-sft",
            "callback_key": "aria:c:dias:d2-sfx-5cc9973ff9",
            "timeout_seconds": 900,
            "payload": {
                "job_id": "d2-sfx-5cc9973ff9",
                "prompt": "sound effect, isolated, sharp, hard impact of skull against rock, dull thud, no music, no reverb, no ambient, no echo, mono",
                "lyrics": "",
                "duration": 0.3,
                "seed": 42,
                "guidance_scale": 7.0,
                "inference_steps": 60,
                "output_style": "sfx",
                "thinking": False,
                "run_demucs": False,
            }
        }
    },
    # 4 — SFX explosion/creature
    {
        "canonical_id": "sfx_impact_explosion_01",
        "type": "sfx",
        "timeout_s": 900,
        "callback_key": "aria:c:dias:d2-sfx-9e3bc3929e",
        "redis_task": {
            "job_id": "d2-sfx-9e3bc3929e",
            "client_id": "dias",
            "model_type": "mus",
            "model_id": "acestep-1.5-xl-sft",
            "callback_key": "aria:c:dias:d2-sfx-9e3bc3929e",
            "timeout_seconds": 900,
            "payload": {
                "job_id": "d2-sfx-9e3bc3929e",
                "prompt": "sound effect, isolated, sharp, ground breaking open, muffled deep boom, creature hissing and skittering, no music, no reverb, no ambient, no echo, mono",
                "lyrics": "",
                "duration": 1.5,
                "seed": 42,
                "guidance_scale": 7.0,
                "inference_steps": 60,
                "output_style": "sfx",
                "thinking": False,
                "run_demucs": False,
            }
        }
    },
    # 5 — STING tragedy
    {
        "canonical_id": "sting_tragedy_01",
        "type": "sting",
        "timeout_s": 900,
        "callback_key": "aria:c:dias:d2-sti-dd45a09fb5",
        "redis_task": {
            "job_id": "d2-sti-dd45a09fb5",
            "client_id": "dias",
            "model_type": "mus",
            "model_id": "acestep-1.5-xl-sft",
            "callback_key": "aria:c:dias:d2-sti-dd45a09fb5",
            "timeout_seconds": 900,
            "payload": {
                "job_id": "d2-sti-dd45a09fb5",
                "prompt": "dramatic sting, orchestral accent, sharp attack, short, tragic revelation, low strings, ominous brass, dark, no vocals, no singing, no sustained pad, retro sci-fi",
                "lyrics": "",
                "duration": 2.0,
                "seed": 42,
                "guidance_scale": 4.5,
                "inference_steps": 60,
                "output_style": "sting",
                "thinking": False,
                "run_demucs": False,
            }
        }
    },
]


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if idx < 0 or idx >= len(PAYLOADS):
        print(f"Indice non valido. Usa 0-{len(PAYLOADS)-1}.")
        print("  0=PAD  1=AMB  2=SFX-stone  3=SFX-body  4=SFX-explosion  5=STING")
        sys.exit(1)

    item    = PAYLOADS[idx]
    cid     = item["canonical_id"]
    cbk     = item["callback_key"]
    timeout = item["timeout_s"]
    task    = json.dumps(item["redis_task"])
    style   = item["type"]
    dur     = item["redis_task"]["payload"]["duration"]

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    print(f"\n{'='*60}")
    print(f"  Asset     : {cid}")
    print(f"  Tipo      : {style.upper()}")
    print(f"  Durata    : {dur}s")
    print(f"  Queue     : {QUEUE}")
    print(f"  Callback  : {cbk}")
    print(f"  Timeout   : {timeout}s")
    print(f"{'='*60}\n")

    # Cleanup eventuale callback residuo
    r.delete(cbk)

    print("Invio task a Redis...")
    r.lpush(QUEUE, task)

    print(f"Attesa callback (max {timeout}s)...")
    t_start = time.time()
    result = r.brpop(cbk, timeout=timeout)
    elapsed = time.time() - t_start

    if not result:
        print(f"\nTimeout ({timeout}s) — nessun callback ricevuto.")
        return

    _, data = result
    resp = json.loads(data)
    status = resp.get("status", "?")
    print(f"\nCallback ricevuto in {elapsed:.1f}s — status: {status}")
    print(json.dumps(resp, indent=2, ensure_ascii=False))

    if status == "done":
        out = resp.get("output", {})
        print(f"\nAudio URL : {out.get('audio_url', 'N/A')}")
        print(f"Durata    : {out.get('duration_seconds', '?')}s")
        if out.get("stems"):
            print(f"Stems     : {list(out['stems'].keys())}")


if __name__ == "__main__":
    main()
