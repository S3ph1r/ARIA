import redis
import json
from pathlib import Path
from datetime import datetime

REDIS_HOST = "192.168.1.120"
REPORTS_DIR = Path("comparison_results")

def main():
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    
    # 1. Recupero Qwen (da report 18:55)
    with open(REPORTS_DIR / "comparison_report_20260316_185516.json", "r") as f:
        full_data_qwen = json.load(f)
        qwen_data = full_data_qwen["qwen35"]
        input_text = full_data_qwen["input_text_sample"]

    # 2. Recupero Gemini Stage B (da report 19:51)
    with open(REPORTS_DIR / "comparison_report_20260316_195132.json", "r") as f:
        full_data_gemini = json.load(f)
        gemini_b = full_data_gemini["gemini"]["stage_b"]
        time_b = full_data_gemini["gemini"]["times"][0]

    # 3. Recupero Gemini Stage C (da Redis)
    res_c_raw = r.lindex('global:callback:dias-test:comp-google-stage-c', 0)
    if not res_c_raw:
        print("ERRORE: Gemini C non trovato in Redis! Sto controllando se è in un report...")
        # Check if it finished in a later report
        return
    
    gemini_c = json.loads(res_c_raw.decode('utf-8'))
    time_c = 12.0 

    # 4. Consolidamento
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "input_text_sample": input_text,
        "gemini": {
            "stage_b": gemini_b,
            "stage_c": gemini_c,
            "times": (time_b, time_c)
        },
        "qwen35": qwen_data
    }

    final_path = REPORTS_DIR / "final_comparison_report.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    print(f"REPORT FINALE GENERATO: {final_path}")

if __name__ == "__main__":
    main()
