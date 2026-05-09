"""
ARIA Dashboard v2.3 Pro — porta 8089 
Live Task Tracking & Metrics Breakdown
"""

import os
import re
import sys
import sqlite3
import time
import threading
import json
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
from fastapi.responses import HTMLResponse

# ── Config ───────────────────────────────────────────────────────────────────

DASHBOARD_PORT    = 8089
REDIS_HOST        = os.getenv("REDIS_HOST", "192.168.1.120")
REDIS_PORT        = int(os.getenv("REDIS_PORT", 6379))
TELEMETRY_DB_PATH = ARIA_ROOT / "logs" / "aria-telemetry.db"
ORCHESTRATOR_LOG  = ARIA_ROOT / "logs" / "aria_orchestrator.log"
MANIFEST_PATH     = ARIA_ROOT / "aria_node_controller" / "config" / "backends_manifest.json"

GOOGLE_LIMITS  = {"rpm": 15, "tpm": 250_000, "rpd": 500}
REDIS_KEYS     = {
    "lockout":    "aria:rate_limit:google:lockout_until",
    "rpm_window": "aria:rate_limit:google:rpm_window",
    "tpm_window": "aria:rate_limit:google:tpm_window",
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

def _get_manifest():
    try:
        if MANIFEST_PATH.exists():
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f).get("backends", {})
    except: pass
    return {}

def _rpm(r):
    try:
        now_ms = int(time.time() * 1000)
        r.zremrangebyscore(REDIS_KEYS["rpm_window"], "-inf", now_ms - 60_000)
        return r.zcard(REDIS_KEYS["rpm_window"])
    except: return 0

def _tpm(r):
    try:
        now_ms = int(time.time() * 1000)
        r.zremrangebyscore(REDIS_KEYS["tpm_window"], "-inf", now_ms - 60_000)
        entries = r.zrange(REDIS_KEYS["tpm_window"], 0, -1)
        return sum(int(e.split(":")[1]) for e in entries if ":" in e)
    except: return 0

def _rpd(r):
    try:
        key = "aria:rate_limit:google:daily_count:" + datetime.now().strftime("%Y-%m-%d")
        val = r.get(key)
        return int(val) if val else 0
    except: return 0

def _lockout(r):
    try:
        val = r.get(REDIS_KEYS["lockout"])
        if val:
            until = datetime.fromisoformat(val)
            rem = (until - datetime.now()).total_seconds()
            if rem > 0:
                h, m = divmod(int(rem), 3600)
                return {"active": True, "label": f"{h}h {m//60}m" if h else f"{m//60}m {int(rem%60)}s"}
    except: pass
    return {"active": False, "label": ""}

def _semaphore(r):
    val = r.get("aria:gpu:semaphore") or "unknown"
    return (val, "GPU Libera" if val == "green" else "GPU Occupata")

def _node_status(r):
    try:
        keys = list(r.scan_iter("aria:global:node:*:status"))
        if not keys: return {}
        # Prende il primo (assumiamo un solo nodo attivo per ora)
        val = r.get(keys[0])
        return json.loads(val) if val else {}
    except: return {}

def _today_stats():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(str(TELEMETRY_DB_PATH))
        c = conn.cursor()
        c.execute("""
            SELECT 
                CASE 
                    WHEN provider = 'google' THEN 'Cloud LLM'
                    WHEN model_id LIKE '%tts%' THEN 'Local TTS'
                    WHEN model_id LIKE '%acestep%' OR model_id LIKE '%audiocraft%' THEN 'Audio Engine'
                    ELSE 'Other'
                END as cat,
                COUNT(*),
                SUM(CASE WHEN status='done' THEN 1 ELSE 0 END),
                SUM(CASE WHEN status='error' THEN 1 ELSE 0 END)
            FROM task_log WHERE ts >= ?
            GROUP BY cat
        """, (today,))
        rows = c.fetchall()
        conn.close()
        return {r[0]: {"total": r[1], "ok": r[2], "err": r[3]} for r in rows}
    except: return {}

def _orch_logs(n=100, offset=0):
    try:
        if not ORCHESTRATOR_LOG.exists(): return ["(log vuoto)"]
        # Usiamo utf-8 con replace per evitare mangling estremo
        with open(ORCHESTRATOR_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(0, total - offset - n)
        end = total - offset
        if end <= 0: return []
        # Pulizia caratteri di controllo ANSI
        return [_ANSI_RE.sub("", l).rstrip("\n\r") for l in lines[start:end]]
    except: return ["(errore lettura log)"]

# ── Rendering ────────────────────────────────────────────────────────────────

def render_page():
    ts = datetime.now().strftime("%H:%M:%S")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ARIA Pro Dashboard v2.3</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter','Segoe UI',sans-serif;background:#050505;color:#eee;font-size:12px;overflow-x:hidden}}
header{{background:rgba(20,20,20,0.9);padding:10px 20px;display:flex;align-items:center;gap:15px;border-bottom:1px solid #222;position:sticky;top:0;z-index:100;backdrop-filter:blur(5px)}}
h1{{font-size:14px;color:#7ecfff;text-transform:uppercase;letter-spacing:2px}}
.sem{{padding:2px 10px;border-radius:10px;font-size:11px;border:1px solid #444;text-transform:uppercase}}
.grid{{display:grid;grid-template-columns:repeat(12, 1fr);gap:10px;padding:10px}}
.card{{background:#111;border:1px solid #222;border-radius:4px;padding:12px;grid-column:span 4}}
.card h2{{font-size:10px;color:#555;text-transform:uppercase;margin-bottom:10px;border-left:2px solid #7ecfff;padding-left:8px}}
.full{{grid-column:span 12}}
.stat-box{{display:flex;justify-content:space-between;margin-bottom:5px;background:#181818;padding:6px;border-radius:3px}}
.stat-val{{font-weight:bold;color:#7ecfff}}
.stat-err{{color:#e74c3c}}
.stat-ok{{color:#2ecc71}}
.stat-label{{color:#777}}
table{{width:100%;border-collapse:collapse}}
th{{text-align:left;color:#444;font-size:10px;padding:5px;border-bottom:1px solid #222}}
td{{padding:6px 5px;border-bottom:1px solid #181818}}
.status-tag{{padding:2px 6px;border-radius:3px;font-size:9px;font-weight:bold}}
.tag-running{{background:rgba(46,204,113,0.2);color:#2ecc71}}
.tag-standby{{background:rgba(243,156,18,0.2);color:#f39c12}}
.tag-offline{{background:rgba(231,76,60,0.2);color:#e74c3c}}
#log-container{{
    height:500px;overflow-y:auto;background:#000;border:1px solid #222;
    padding:10px;font-family:'Fira Code','Consolas',monospace;font-size:11px;color:#8ccfff;
}}
.current-task{{color:#f1c40f;font-style:italic}}
.lk-active{{color:#f39c12;border:1px solid #f39c12;padding:2px 8px;border-radius:10px;font-size:10px;margin-bottom:10px;display:inline-block}}
.gauge-bar{{height:6px;background:#222;border-radius:3px;margin:4px 0 8px 0;overflow:hidden}}
.gauge-fill{{height:100%;transition:width 0.5s ease}}
</style>
</head>
<body>
<header>
  <h1>ARIA 2.3 Pro</h1>
  <div id="sem-indicator" class="sem">...</div>
  <div style="margin-left:auto; color:#444" id="clock">{ts}</div>
</header>

<div class="grid">
  <!-- Colonna 1: Quota Cloud -->
  <div class="card">
    <h2>Cloud Gateway (Gemini)</h2>
    <div id="lockout-status"></div>
    <div id="google-gauges"></div>
  </div>

  <!-- Colonna 2: Backend Active Status -->
  <div class="card" style="grid-column:span 8">
    <h2>Service Topology & Live Tracking</h2>
    <table id="backends-table">
      <thead><tr><th>Service</th><th>Status</th><th>Instance</th><th>Current Task</th><th>Port</th></tr></thead>
      <tbody id="backends-body"></tbody>
    </table>
  </div>

  <!-- Colonna 3: Performance Breakdown -->
  <div class="card">
    <h2>Cloud LLM Metrics</h2>
    <div id="stats-cloud"></div>
  </div>
  <div class="card">
    <h2>Local TTS Metrics</h2>
    <div id="stats-tts"></div>
  </div>
  <div class="card">
    <h2>Audio Engine Metrics</h2>
    <div id="stats-audio"></div>
  </div>

  <!-- Log Terminal -->
  <div class="card full">
    <h2>System Orchestrator Stream</h2>
    <div id="log-container"><div id="log-content"></div></div>
  </div>
</div>

<script>
let logOffset = 0;
let autoScroll = true;
const logContainer = document.getElementById('log-container');
const logContent = document.getElementById('log-content');

function getLineId(line) {{
    let hash = 0;
    for (let i = 0; i < line.length; i++) {{ hash = ((hash << 5) - hash) + line.charCodeAt(i); hash |= 0; }}
    return 'L' + Math.abs(hash).toString(36) + line.length;
}}

async function update() {{
    try {{
        const r = await fetch('/api/data');
        const d = await r.json();
        
        document.getElementById('clock').innerText = d.ts;
        const sem = document.getElementById('sem-indicator');
        sem.innerText = d.semaphore.label;
        sem.style.color = d.semaphore.state === 'green' ? '#2ecc71' : '#e74c3c';
        sem.style.borderColor = d.semaphore.state === 'green' ? '#2ecc71' : '#e74c3c';

        // Google Gauges
        let gHtml = "";
        if (d.lockout.active) {{
            gHtml += `<div class="lk-active">LOCKOUT: ripresa in ${{d.lockout.label}}</div>`;
        }}
        for (let k in d.limits) {{
            const lim = d.limits[k];
            const pct = Math.min(100, Math.round(lim.current / lim.max * 100));
            const color = pct > 80 ? '#e74c3c' : (pct > 50 ? '#f39c12' : '#2ecc71');
            gHtml += `<div style="display:flex; justify-content:space-between"><span>${{k.toUpperCase()}}</span><span>${{lim.current}}/${{lim.max}}</span></div>
                      <div class="gauge-bar"><div class="gauge-fill" style="width:${{pct}}%; background:${{color}}"></div></div>`;
        }}
        document.getElementById('google-gauges').innerHTML = gHtml;

        // Metrics Breakdown
        const renderStat = (cat, elId) => {{
            const s = d.today[cat] || {{total:0, ok:0, err:0}};
            document.getElementById(elId).innerHTML = `
                <div class="stat-box"><span class="stat-label">Totali</span><span class="stat-val">${{s.total}}</span></div>
                <div class="stat-box"><span class="stat-label">Successi</span><span class="stat-ok">${{s.ok}}</span></div>
                <div class="stat-box"><span class="stat-label">Falliti</span><span class="stat-err">${{s.err}}</span></div>
            `;
        }};
        renderStat('Cloud LLM', 'stats-cloud');
        renderStat('Local TTS', 'stats-tts');
        renderStat('Audio Engine', 'stats-audio');

        // Backends Table
        let bHtml = "";
        const activeBackends = d.node.active_backends || [];
        const currentTasks = d.node.current_tasks || {{}};
        
        // Add Gemini fixed
        bHtml += `<tr><td>Google Gemini LLM</td><td><span class="status-tag tag-running">CLOUD</span></td><td>External</td><td>-</td><td>N/A</td></tr>`;
        
        for (let id in d.manifest) {{
            const m = d.manifest[id];
            const isRunning = activeBackends.includes(id);
            const statusLabel = isRunning ? "RUNNING" : "STANDBY";
            const statusClass = isRunning ? "tag-running" : "tag-standby";
            const task = currentTasks[id] || "-";
            bHtml += `<tr>
                <td>${{m.metadata.display_name}}</td>
                <td><span class="status-tag ${{statusClass}}">${{statusLabel}}</span></td>
                <td>${{isRunning ? "Active" : "JIT Ready"}}</td>
                <td class="current-task">${{task}}</td>
                <td>${{m.port}}</td>
            </tr>`;
        }}
        document.getElementById('backends-body').innerHTML = bHtml;

    }} catch(e) {{ console.error(e); }}
}}

async function fetchLogs() {{
    try {{
        const r = await fetch('/api/logs?limit=30');
        const d = await r.json();
        let added = false;
        d.logs.forEach(line => {{
            const id = getLineId(line);
            if (!document.getElementById(id)) {{
                const div = document.createElement('div');
                div.id = id; div.innerText = line;
                div.style.padding = "1px 0";
                logContent.appendChild(div);
                added = true;
            }}
        }});
        if (added && autoScroll) {{ logContainer.scrollTop = logContainer.scrollHeight; }}
        if (logContent.children.length > 1000) {{ logContent.removeChild(logContent.firstChild); }}
    }} catch(e) {{}}
}}

logContainer.addEventListener('scroll', () => {{
    autoScroll = (logContainer.scrollHeight - logContainer.scrollTop - logContainer.clientHeight) < 30;
}});

setInterval(update, 2000);
setInterval(fetchLogs, 2000);
update(); fetchLogs();
</script>
</body>
</html>"""
    return html

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def dashboard(): return render_page()

@app.get("/api/data")
def api_data():
    r = get_redis()
    return {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "semaphore": dict(zip(["state","label"], _semaphore(r))),
        "limits": {
            "rpm": {"current": _rpm(r), "max": 15},
            "tpm": {"current": _tpm(r), "max": 250000},
            "rpd": {"current": _rpd(r), "max": 500}
        },
        "lockout": _lockout(r),
        "today": _today_stats(),
        "manifest": _get_manifest(),
        "node": _node_status(r)
    }

@app.get("/api/logs")
def api_logs(limit: int = 50, offset: int = 0):
    return {"logs": _orch_logs(limit, offset)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089, log_level="warning")
