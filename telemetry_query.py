import sqlite3
conn = sqlite3.connect('C:/Users/roberto/aria/logs/aria-telemetry.db')
c = conn.cursor()

c.execute("SELECT status, COUNT(*), ROUND(AVG(processing_s),1), ROUND(AVG(queue_wait_s),1) FROM task_log WHERE ts >= '2026-05-02' GROUP BY status")
print('=== STATUS oggi ===')
for r in c.fetchall():
    print(f"  {r[0]:10} | count={r[1]:4} | avg_proc={r[2]}s | avg_wait={r[3]}s")

c.execute("SELECT model_id, status, COUNT(*) FROM task_log WHERE ts >= '2026-05-02' GROUP BY model_id, status")
print('=== MODELLO oggi ===')
for r in c.fetchall():
    print(f"  {r[0]:45} | {r[1]:10} | {r[2]}")

c.execute("SELECT substr(ts,12,2), COUNT(*), SUM(CASE WHEN status='done' THEN 1 ELSE 0 END), SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) FROM task_log WHERE ts >= '2026-05-02' GROUP BY substr(ts,12,2) ORDER BY 1")
print('=== ORA (UTC) ===')
for r in c.fetchall():
    print(f"  {r[0]}:xx | tot={r[1]:4} | ok={r[2]:4} | err={r[3]:4}")

c.execute("SELECT ts, error_code, model_id FROM task_log WHERE ts >= '2026-05-02' AND status='error' ORDER BY ts DESC LIMIT 10")
print('=== ERRORI ultimi 10 ===')
for r in c.fetchall():
    print(f"  {r[0][:19]} | {r[2]} | {r[1]}")
