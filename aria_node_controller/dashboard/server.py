"""
ARIA Dashboard — porta 8089  (server-side rendering, no JavaScript)
Auto-refresh via <meta http-equiv="refresh" content="5">
"""

import os
import re
import sys
import sqlite3
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

_THIS    = Path(__file__).resolve()
ARIA_ROOT = _THIS.parent.parent.parent

sys.path.insert(0, str(ARIA_ROOT))

import redis as redis_lib
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

# ── Config ───────────────────────────────────────────────────────────────────

DASHBOARD_PORT    = 8089
REDIS_HOST        = os.getenv("REDIS_HOST", "192.168.1.120")
REDIS_PORT        = int(os.getenv("REDIS_PORT", 6379))
TELEMETRY_DB_PATH = ARIA_ROOT / "logs" / "aria-telemetry.db"
ORCHESTRATOR_LOG  = ARIA_ROOT / "logs" / "aria_orchestrator.log"

GOOGLE_LIMITS  = {"rpm": 15, "tpm": 250_000, "rpd": 500}
GOOGLE_RESET_TZ = timezone(timedelta(hours=-7))
DAILY_PREFIX   = "aria:rate_limit:google:daily_count:"
REDIS_KEYS     = {
    "lockout":    "aria:rate_limit:google:lockout_until",
    "rpm_window": "aria:rate_limit:google:rpm_window",
    "tpm_window": "aria:rate_limit:google:tpm_window",
}

BACKENDS = {
    "qwen3-tts-1.7b":      "http://127.0.0.1:8083/health",
    "voice-cloning":       "http://127.0.0.1:8081/health",
    "fish-s1-mini":        "http://127.0.0.1:8080/v1/health",
    "acestep-1.5-xl-sft":  "http://127.0.0.1:8084/health",
    "asset-server":        "http://127.0.0.1:8082/",
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# ── Redis ─────────────────────────────────────────────────────────────────────

_redis_conn = None

def get_redis():
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_conn

# ── Data helpers ──────────────────────────────────────────────────────────────

def _rpm(r):
    try:
        now_ms = int(time.time() * 1000)
        r.zremrangebyscore(REDIS_KEYS["rpm_window"], "-inf", now_ms - 60_000)
        return r.zcard(REDIS_KEYS["rpm_window"])
    except Exception:
        return 0

def _tpm(r):
    try:
        now_ms = int(time.time() * 1000)
        r.zremrangebyscore(REDIS_KEYS["tpm_window"], "-inf", now_ms - 60_000)
        entries = r.zrange(REDIS_KEYS["tpm_window"], 0, -1)
        return sum(int(e.split(":")[1]) for e in entries if ":" in e)
    except Exception:
        return 0

def _rpd(r):
    try:
        key = DAILY_PREFIX + datetime.now().strftime("%Y-%m-%d")
        val = r.get(key)
        return int(val) if val else 0
    except Exception:
        return 0

def _lockout(r):
    try:
        val = r.get(REDIS_KEYS["lockout"])
        if val:
            until = datetime.fromisoformat(val)
            remaining = (until - datetime.now()).total_seconds()
            if remaining > 0:
                h, rem = divmod(int(remaining), 3600)
                m = rem // 60
                label = f"{h}h {m:02d}m" if h else f"{m}m {rem % 60:02d}s"
                return {"active": True, "label": label}
    except Exception:
        pass
    return {"active": False, "label": ""}

def _semaphore(r):
    try:
        val = r.get("aria:gpu:semaphore") or "unknown"
        if val == "green":
            return ("green", "GPU Libera - Workflow AI Completo")
        if val == "red":
            return ("red", "GPU Occupata - Solo Cloud Gateway")
    except Exception:
        pass
    return ("unknown", "Semaforo non disponibile")

def _reset_eta():
    now   = datetime.now(GOOGLE_RESET_TZ)
    reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    reset_it = reset.astimezone(timezone(timedelta(hours=2)))
    delta    = reset - now
    h, rem   = divmod(int(delta.total_seconds()), 3600)
    m        = rem // 60
    return f"fra {h}h {m:02d}m ({reset_it.strftime('%H:%M')} IT)"

def _queues(r):
    try:
        keys = list(r.scan_iter("aria:q:*"))
        out  = []
        for k in sorted(keys):
            try:
                t = r.type(k)
                n = r.llen(k) if t == "list" else r.zcard(k) if t == "zset" else None
                if n:
                    out.append((k, t, n))
            except Exception:
                pass
        return out
    except Exception:
        return []

def _today_stats():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        conn  = sqlite3.connect(str(TELEMETRY_DB_PATH))
        c     = conn.cursor()
        c.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN status='done'  THEN 1 ELSE 0 END),
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END),
                   ROUND(AVG(CASE WHEN status='done' THEN processing_s END), 1)
            FROM task_log WHERE ts >= ?
        """, (today,))
        row = c.fetchone()
        conn.close()
        return {"total": row[0] or 0, "ok": row[1] or 0, "err": row[2] or 0, "avg_s": row[3]}
    except Exception:
        return {"total": 0, "ok": 0, "err": 0, "avg_s": None}

def _recent_tasks(limit=20):
    try:
        conn = sqlite3.connect(str(TELEMETRY_DB_PATH))
        c    = conn.cursor()
        c.execute("""
            SELECT ts, job_id, model_id, status,
                   ROUND(processing_s,1), error_code, input_tokens, output_tokens
            FROM task_log ORDER BY ts DESC LIMIT ?
        """, (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception:
        return []

def _orch_logs(n=20):
    try:
        if not ORCHESTRATOR_LOG.exists():
            return ["(log non ancora creato)"]
        with open(ORCHESTRATOR_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [_ANSI_RE.sub("", l).rstrip("\n\r") for l in lines[-n:]]
    except Exception as e:
        return [f"(errore: {e})"]

# ── Backend cache (background thread) ────────────────────────────────────────

_backend_cache = {name: {"up": None} for name in BACKENDS}
_backend_lock  = threading.Lock()

def _check_one(name_url):
    name, url = name_url
    try:
        resp = requests.get(url, timeout=2)
        return name, {"up": resp.status_code == 200}
    except Exception:
        return name, {"up": False}

def _backend_loop():
    while True:
        with ThreadPoolExecutor(max_workers=len(BACKENDS)) as ex:
            results = dict(ex.map(_check_one, BACKENDS.items()))
        with _backend_lock:
            _backend_cache.update(results)
        time.sleep(15)

threading.Thread(target=_backend_loop, daemon=True).start()

def _backends():
    with _backend_lock:
        return dict(_backend_cache)

# ── HTML rendering ────────────────────────────────────────────────────────────

def _h(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _bar(pct):
    cls = "crit" if pct >= 90 else "warn" if pct >= 70 else "ok"
    return (f'<div class="bar"><div class="fill fill-{cls}" style="width:{pct}%"></div></div>'
            f'<span class="val">{pct}%</span>')

def _dot(up):
    if up is None:
        return '<span style="color:#666">? ?</span>'
    return '<span style="color:#2ecc71">UP</span>' if up else '<span style="color:#e74c3c">DOWN</span>'

def render_page():
    r   = get_redis()
    rpm_val = _rpm(r)
    tpm_val = _tpm(r)
    rpd_val = _rpd(r)
    lk      = _lockout(r)
    sem     = _semaphore(r)
    eta     = _reset_eta()
    queues  = _queues(r)
    today   = _today_stats()
    tasks   = _recent_tasks(20)
    logs    = _orch_logs(20)
    backends = _backends()
    ts      = datetime.now().strftime("%H:%M:%S")

    sem_color = {"green": "#2ecc71", "red": "#e74c3c"}.get(sem[0], "#888")
    lk_color  = "#f39c12" if lk["active"] else "#2ecc71"
    lk_text   = f"LOCKOUT - ripresa fra {lk['label']}" if lk["active"] else "Nessun lockout"

    def gauge(label, val, max_val):
        pct = min(100, round(val / max_val * 100))
        return f"""
        <div class="gauge-row">
          <span class="glabel">{label}</span>
          {_bar(pct)}
          <span class="gval">{val} / {max_val}</span>
        </div>"""

    # Queue rows
    q_rows = ""
    if queues:
        for key, typ, length in queues:
            sk = ("..." + key[-42:]) if len(key) > 45 else key
            q_rows += f"<tr><td>{_h(sk)}</td><td>{typ}</td><td class='num'>{length}</td></tr>"
    else:
        q_rows = "<tr><td colspan='3' class='muted'>Nessuna coda attiva</td></tr>"

    # Task rows
    t_rows = ""
    for row in tasks:
        ts_val, job, model, status, proc_s, err, in_tok, out_tok = row
        sc = "ok" if status == "done" else "err"
        t_rows += (f"<tr>"
                   f"<td>{_h(ts_val or '')[:19]}</td>"
                   f"<td class='muted'>{_h((job or '')[:16])}</td>"
                   f"<td>{_h(model or '')}</td>"
                   f"<td class='{sc}'>{_h(status)}</td>"
                   f"<td>{proc_s or '-'}</td>"
                   f"<td>{in_tok or '-'}</td>"
                   f"<td>{out_tok or '-'}</td>"
                   f"<td class='err'>{_h(err or '')}</td>"
                   f"</tr>")

    # Backend rows
    b_rows = ""
    for name, info in backends.items():
        b_rows += f"<tr><td>{_h(name)}</td><td>{_dot(info['up'])}</td></tr>"
    cloud_up = not lk["active"] and rpd_val < GOOGLE_LIMITS["rpd"]
    b_rows += f"<tr><td>gemini-api (cloud)</td><td>{_dot(cloud_up)}</td></tr>"

    # Log lines
    log_html = _h("\n".join(logs))

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="5">
<title>ARIA Dashboard</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e0e0e0;font-family:Consolas,monospace;font-size:13px}}
header{{background:#111;border-bottom:1px solid #222;padding:10px 20px;display:flex;align-items:center;gap:16px}}
header h1{{font-size:16px;color:#7ecfff;letter-spacing:1px}}
.ts{{color:#555;font-size:11px;margin-left:auto}}
.sem{{padding:3px 12px;border-radius:12px;font-size:12px;border:1px solid {sem_color};color:{sem_color};background:rgba(0,0,0,.4)}}
.lk{{display:inline-block;padding:4px 10px;border-radius:12px;font-size:12px;border:1px solid {lk_color};color:{lk_color};background:rgba(0,0,0,.4);margin-bottom:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:14px}}
.card{{background:#151515;border:1px solid #222;border-radius:6px;padding:12px}}
.card h2{{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}}
.full{{grid-column:1/-1}}
.gauge-row{{display:flex;align-items:center;gap:8px;margin-bottom:8px}}
.glabel{{width:40px;color:#aaa;font-size:12px}}
.bar{{flex:1;height:12px;background:#222;border-radius:3px;overflow:hidden}}
.fill{{height:100%;border-radius:3px}}
.fill-ok{{background:#2ecc71}}.fill-warn{{background:#f39c12}}.fill-crit{{background:#e74c3c}}
.val,.gval{{width:60px;text-align:right;color:#ddd;font-size:11px}}
table{{width:100%;border-collapse:collapse}}
th{{color:#555;font-weight:normal;text-align:left;font-size:11px;padding:4px 6px;border-bottom:1px solid #222}}
td{{padding:3px 6px;border-bottom:1px solid #1a1a1a;font-size:11px}}
tr:hover td{{background:#1c1c1c}}
.num,.ok,.err,.muted{{}}
.num{{font-weight:bold;color:#7ecfff;text-align:right}}
.ok{{color:#2ecc71}}.err{{color:#e74c3c;font-size:10px}}.muted{{color:#555}}
.stats{{display:flex;gap:12px;margin-bottom:8px}}
.stat{{text-align:center}}
.stat-n{{font-size:28px;color:#7ecfff}}
.stat-l{{color:#555;font-size:11px}}
pre{{color:#7ecfff;font-size:11px;line-height:1.6;white-space:pre-wrap;max-height:320px;
     overflow-y:auto;background:#0a0a0a;padding:10px;border-radius:4px;border:1px solid #1a1a1a}}
.eta{{color:#555;font-size:11px;margin-top:6px}}
</style>
</head>
<body>
<header>
  <h1>ARIA - Adaptive Resource for Inference and AI</h1>
  <span class="sem">{_h(sem[1])}</span>
  <span class="ts">{ts} | auto-refresh 5s | porta 8089</span>
</header>
<div class="grid">

  <div class="card">
    <h2>Limiti Google API</h2>
    <div class="lk">{_h(lk_text)}</div>
    {gauge("RPD", rpd_val, GOOGLE_LIMITS["rpd"])}
    {gauge("RPM", rpm_val, GOOGLE_LIMITS["rpm"])}
    {gauge("TPM", tpm_val, GOOGLE_LIMITS["tpm"])}
    <div class="eta">Reset quota: {_h(eta)}</div>
  </div>

  <div class="card">
    <h2>Oggi</h2>
    <div class="stats">
      <div class="stat"><div class="stat-n">{today["total"]}</div><div class="stat-l">Totali</div></div>
      <div class="stat"><div class="stat-n" style="color:#2ecc71">{today["ok"]}</div><div class="stat-l">OK</div></div>
      <div class="stat"><div class="stat-n" style="color:#e74c3c">{today["err"]}</div><div class="stat-l">Errori</div></div>
    </div>
    {"<div class='muted'>Tempo medio: " + str(today["avg_s"]) + "s</div>" if today["avg_s"] else ""}
  </div>

  <div class="card">
    <h2>Backend</h2>
    <table><tr><th>Servizio</th><th>Stato</th></tr>{b_rows}</table>
  </div>

  <div class="card">
    <h2>Code Redis</h2>
    <table><tr><th>Queue</th><th>Tipo</th><th>N</th></tr>{q_rows}</table>
  </div>

  <div class="card full">
    <h2>Ultimi task</h2>
    <table>
      <tr><th>Timestamp</th><th>Job ID</th><th>Modello</th><th>Status</th>
          <th>Proc(s)</th><th>Tok in</th><th>Tok out</th><th>Errore</th></tr>
      {t_rows if t_rows else "<tr><td colspan='8' class='muted'>Nessun task</td></tr>"}
    </table>
  </div>

  <div class="card full">
    <h2>Log Orchestrator ARIA (ultime 20 righe)</h2>
    <pre>{log_html}</pre>
  </div>

</div>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────

app = FastAPI(title="ARIA Dashboard", docs_url=None, redoc_url=None)

@app.get("/")
def dashboard():
    html = render_page()
    return Response(
        content=html,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

@app.get("/api/status")
def api_status():
    r       = get_redis()
    rpm_val = _rpm(r)
    tpm_val = _tpm(r)
    rpd_val = _rpd(r)
    lk      = _lockout(r)
    sem     = _semaphore(r)
    return {
        "ts":        datetime.now().strftime("%H:%M:%S"),
        "semaphore": {"state": sem[0], "label": sem[1]},
        "limits": {
            "rpm": {"current": rpm_val, "max": GOOGLE_LIMITS["rpm"]},
            "tpm": {"current": tpm_val, "max": GOOGLE_LIMITS["tpm"]},
            "rpd": {"current": rpd_val, "max": GOOGLE_LIMITS["rpd"]},
        },
        "lockout":  lk,
        "today":    _today_stats(),
        "logs":     _orch_logs(20),
    }

# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"ARIA Dashboard (SSR) avviata su http://0.0.0.0:{DASHBOARD_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="warning")
