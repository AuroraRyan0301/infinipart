#!/usr/bin/env python3
"""
Infinigen-Sim Pipeline Dashboard — event-driven monitoring for the "丹炉".

Reads ONLY:
  - /mnt/data_ssd/infinigen-sim/.stats/*.json (written by pipeline components)
  - nvidia-smi (fast subprocess)
  - tail of log files (last ~20KB, no scanning)

Zero directory scanning. Instant response.

Usage:
    python dashboard.py [--port 8501]
"""
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from string import Template

# ======================================================================
# Config
# ======================================================================

STATS_DIR = "/mnt/data_ssd/infinigen-sim/.stats"
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILES = {
    "train": os.path.join(REPO_DIR, "train_log_dynamic.txt"),
    "gen": os.path.join(REPO_DIR, "gen_log.txt"),
    "encode": os.path.join(REPO_DIR, "encode_watch_log.txt"),
}


# ======================================================================
# Data collection — all fast, no scanning
# ======================================================================

def read_stats_file(name):
    """Read a JSON stats file. Returns {} on any error."""
    path = os.path.join(STATS_DIR, name)
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        )
        gpus = []
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "id": int(parts[0]), "name": parts[1],
                    "mem_used": int(parts[2]), "mem_total": int(parts[3]),
                    "util": int(parts[4]), "temp": int(parts[5]),
                })
        return gpus
    except Exception:
        return []


def tail_log(name, nbytes=20000):
    path = LOG_FILES.get(name)
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", errors="replace").split("\n")
    except Exception:
        return []


def parse_train_from_log():
    lines = tail_log("train", 30000)
    info = {"status": "stopped", "step": 0, "loss": 0, "lr": "—",
            "n_train": 0, "recent_losses": []}

    for line in reversed(lines):
        m = re.search(r"Training:.*?(\d+)/(\d+).*?loss=([0-9.]+).*?lr=([0-9.e+-]+).*?n=(\d+)", line)
        if m:
            info.update({
                "status": "training",
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "loss": float(m.group(3)),
                "lr": m.group(4),
                "n_train": int(m.group(5)),
            })
            break

    for line in reversed(lines[-300:]):
        m = re.search(r"Step (\d+)/\d+ \| avg100: ([0-9.]+)", line)
        if m:
            info["recent_losses"].append({"step": int(m.group(1)), "avg100": float(m.group(2))})
            if len(info["recent_losses"]) >= 20:
                break
    info["recent_losses"].reverse()

    if info["status"] == "stopped":
        try:
            out = subprocess.check_output(["pgrep", "-f", "train_partnet_vjepa"],
                                          text=True, timeout=3)
            if out.strip():
                info["status"] = "starting"
        except Exception:
            pass

    return info


def parse_encode_from_log():
    lines = tail_log("encode", 15000)
    info = {"status": "stopped", "vae_progress": "—", "jepa_progress": "—", "iter": "—"}

    for line in reversed(lines):
        if "VAE:" in line and info["vae_progress"] == "—":
            m = re.search(r"VAE:\s+(\d+)%.*?(\d+)/(\d+)", line)
            if m:
                info["vae_progress"] = f"{m.group(2)}/{m.group(3)} ({m.group(1)}%)"
        if "JEPA:" in line and info["jepa_progress"] == "—":
            m = re.search(r"JEPA:\s+(\d+)%.*?(\d+)/(\d+)", line)
            if m:
                info["jepa_progress"] = f"{m.group(2)}/{m.group(3)} ({m.group(1)}%)"
        if info["vae_progress"] != "—" and info["jepa_progress"] != "—":
            break

    for line in reversed(lines):
        m = re.search(r"\[Iter (\d+)\] Found (\d+) new animodes.*?(\d+) already done", line)
        if m:
            info["iter"] = int(m.group(1))
            info["new_animodes"] = int(m.group(2))
            info["already_done"] = int(m.group(3))
            break

    try:
        out = subprocess.check_output(["pgrep", "-f", "encode_for_training"],
                                      text=True, timeout=3)
        info["status"] = "running" if out.strip() else "stopped"
    except Exception:
        pass

    return info


def get_gen_status():
    stats = read_stats_file("gen.json")
    if not stats:
        try:
            out = subprocess.check_output(["pgrep", "-f", "cluster_launch"],
                                          text=True, timeout=3)
            if out.strip():
                return {"status": "running (waiting for stats)", "precompute_ok": 0,
                        "render_ok": 0, "render_fail": 0, "precompute_fail": 0,
                        "total_objects": 0, "recent_events": []}
        except Exception:
            pass
        return {"status": "stopped"}

    stats["status"] = "finished" if stats.get("finished") else "running"
    return stats


def fmt_elapsed(seconds):
    if not seconds:
        return "—"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def fmt_eta(seconds):
    if not seconds or seconds <= 0:
        return "—"
    seconds = int(seconds)
    if seconds > 86400:
        d = seconds // 86400
        h = (seconds % 86400) // 3600
        return f"~{d}d {h}h"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    if h > 0:
        return f"~{h}h {m}m"
    return f"~{m}m"


# ======================================================================
# HTML template
# ======================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>丹炉 Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0f; color: #e0e0e0; padding: 16px; }
  h1 { font-size: 20px; color: #ff6b35; margin-bottom: 12px; }
  h2 { font-size: 14px; color: #888; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 2px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .card { background: #12121a; border: 1px solid #222; border-radius: 8px; padding: 14px; }
  .card.wide { grid-column: 1 / -1; }
  .status { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
  .status.running { background: #1a3a1a; color: #4ade80; }
  .status.stopped { background: #3a1a1a; color: #f87171; }
  .status.training { background: #1a2a3a; color: #60a5fa; }
  .status.starting { background: #3a3a1a; color: #fbbf24; }
  .status.finished { background: #1a3a3a; color: #22d3ee; }
  .metric { display: inline-block; margin-right: 16px; margin-bottom: 4px; }
  .metric .label { font-size: 10px; color: #666; text-transform: uppercase; }
  .metric .value { font-size: 18px; font-weight: bold; color: #fff; }
  .metric .value.highlight { color: #ff6b35; }
  .metric .value.sm { font-size: 13px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #1a1a24; }
  th { color: #666; font-size: 10px; text-transform: uppercase; }
  .gpu-bar { height: 16px; background: #1a1a24; border-radius: 3px; overflow: hidden; position: relative; }
  .gpu-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .gpu-fill.low { background: #22c55e; }
  .gpu-fill.mid { background: #eab308; }
  .gpu-fill.high { background: #ef4444; }
  .gpu-label { position: absolute; right: 4px; top: 0; line-height: 16px; font-size: 10px; color: #fff; }
  .event { font-size: 11px; padding: 2px 0; }
  .event .ok { color: #4ade80; }
  .event .fail { color: #f87171; }
  .event .type { color: #60a5fa; }
  .event .dur { color: #888; font-size: 10px; }
  .loss-chart { display: flex; align-items: end; height: 60px; gap: 2px; margin-top: 8px; }
  .loss-bar { background: #3b82f6; min-width: 8px; border-radius: 2px 2px 0 0; flex: 1; }
  .timestamp { font-size: 10px; color: #444; text-align: right; margin-top: 8px; }
  .pbar { height: 8px; background: #1a1a24; border-radius: 4px; overflow: hidden; margin-top: 4px; margin-bottom: 2px; }
  .pbar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .pbar-fill.orange { background: linear-gradient(90deg, #ff6b35, #f59e0b); }
  .pbar-fill.blue { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
  .pbar-fill.green { background: linear-gradient(90deg, #22c55e, #4ade80); }
  .pbar-info { font-size: 10px; color: #666; display: flex; justify-content: space-between; }
  .phase-section { margin-top: 10px; padding: 8px; background: #0e0e16; border-radius: 6px; border: 1px solid #1a1a24; }
  .phase-title { font-size: 11px; color: #aaa; font-weight: bold; margin-bottom: 4px; }
  .active-item { font-size: 11px; color: #4ade80; padding: 1px 0; }
  .active-item .gpu { color: #60a5fa; font-weight: bold; }
  .active-item .elapsed { color: #888; font-size: 10px; }
</style>
</head>
<body>

<h1>丹炉 Pipeline Dashboard</h1>

<!-- GPU Status -->
<div class="grid">
<div class="card wide">
<h2>GPU Status</h2>
<table>
<tr><th>GPU</th><th>VRAM</th><th>Utilization</th><th>Temp</th><th>Role</th></tr>
$gpu_rows
</table>
</div>
</div>

<!-- Main panels -->
<div class="grid">

<!-- Training -->
<div class="card">
<h2>Training <span class="status $train_status_class">$train_status</span></h2>
<div style="margin-top:8px">
  <div class="metric"><div class="label">Step</div><div class="value highlight">$train_step</div></div>
  <div class="metric"><div class="label">Loss</div><div class="value">$train_loss</div></div>
  <div class="metric"><div class="label">LR</div><div class="value">$train_lr</div></div>
  <div class="metric"><div class="label">Data (views)</div><div class="value">$train_n</div></div>
</div>
$loss_chart
</div>

<!-- Generation Pipeline -->
<div class="card">
<h2>Generation <span class="status $gen_status_class">$gen_status</span></h2>
<div style="margin-top:8px">
  <div class="metric"><div class="label">Total Objects</div><div class="value highlight">$gen_total</div></div>
  <div class="metric"><div class="label">Elapsed</div><div class="value">$gen_elapsed</div></div>
  <div class="metric"><div class="label">Phase</div><div class="value sm">$gen_phase</div></div>
</div>

<!-- Precompute progress -->
<div class="phase-section">
  <div class="phase-title">Precompute (CPU)</div>
  <div class="pbar"><div class="pbar-fill orange" style="width:$pc_pct%"></div></div>
  <div class="pbar-info">
    <span>$pc_done / $gen_total ($pc_pct%) — OK: $pc_ok, Fail: $pc_fail</span>
    <span>Avg: $pc_avg | ETA: $pc_eta</span>
  </div>
  $pc_active
</div>

<!-- Render progress -->
<div class="phase-section">
  <div class="phase-title">Render (GPU)</div>
  <div class="pbar"><div class="pbar-fill blue" style="width:$rn_pct%"></div></div>
  <div class="pbar-info">
    <span>$rn_done / $rn_total ($rn_pct%) — OK: $rn_ok, Fail: $rn_fail</span>
    <span>Avg: $rn_avg | ETA: $rn_eta | Queue: $rn_queue</span>
  </div>
  $rn_active
</div>
</div>

<!-- Encoding -->
<div class="card">
<h2>Encoding <span class="status $enc_status_class">$enc_status</span></h2>
<div style="margin-top:8px">
  <div class="metric"><div class="label">VAE</div><div class="value">$enc_vae</div></div>
  <div class="metric"><div class="label">JEPA</div><div class="value">$enc_jepa</div></div>
  <div class="metric"><div class="label">Iteration</div><div class="value">$enc_iter</div></div>
  <div class="metric"><div class="label">New Animodes</div><div class="value">$enc_new</div></div>
  <div class="metric"><div class="label">Already Done</div><div class="value">$enc_done</div></div>
</div>
</div>

<!-- Recent Events -->
<div class="card">
<h2>Recent Events</h2>
<div style="margin-top:4px; max-height: 200px; overflow-y: auto;">
$recent_events
</div>
</div>

</div>

<div class="timestamp">Updated: $timestamp | Auto-refresh: 10s | Event-driven (zero scanning)</div>

</body>
</html>"""


# ======================================================================
# Render
# ======================================================================

def render_dashboard():
    gpus = get_gpu_status()
    train = parse_train_from_log()
    gen = get_gen_status()
    enc = parse_encode_from_log()
    now_ts = time.time()

    # GPU rows
    gpu_roles = {0: "Train", 1: "Train", 2: "Render+Encode", 3: "Render"}
    gpu_rows = ""
    for g in gpus:
        pct = int(g["mem_used"] / g["mem_total"] * 100) if g["mem_total"] > 0 else 0
        fill_class = "low" if g["util"] < 50 else ("mid" if g["util"] < 80 else "high")
        role = gpu_roles.get(g["id"], "")
        gpu_rows += (
            f'<tr><td>GPU {g["id"]}</td>'
            f'<td>{g["mem_used"]}MB / {g["mem_total"]}MB ({pct}%)</td>'
            f'<td><div class="gpu-bar"><div class="gpu-fill {fill_class}" style="width:{g["util"]}%"></div>'
            f'<span class="gpu-label">{g["util"]}%</span></div></td>'
            f'<td>{g["temp"]}&deg;C</td><td>{role}</td></tr>'
        )

    # Loss chart
    loss_chart = ""
    if train.get("recent_losses"):
        losses = [x["avg100"] for x in train["recent_losses"]]
        max_loss = max(losses) if losses else 1
        bars = ""
        for val in losses:
            h = max(2, int(val / max_loss * 56))
            bars += f'<div class="loss-bar" style="height:{h}px" title="{val:.4f}"></div>'
        loss_chart = f'<div class="loss-chart">{bars}</div>'

    # Generation stats
    gen_total = gen.get("total_objects", 0)
    if not isinstance(gen_total, int):
        gen_total = 0

    # Precompute progress
    pc_ok = gen.get("precompute_ok", 0)
    pc_fail = gen.get("precompute_fail", 0) + gen.get("precompute_skip", 0)
    pc_done = gen.get("precompute_done", pc_ok + pc_fail)
    pc_pct = int(pc_done / gen_total * 100) if gen_total else 0
    pc_avg = f'{gen.get("precompute_avg_s", 0):.1f}s' if gen.get("precompute_avg_s") else "—"
    pc_eta = fmt_eta(gen.get("precompute_eta_s"))

    # Active precomputes
    pc_active_items = gen.get("active_precomputes", [])
    pc_active = ""
    if pc_active_items:
        items_str = ", ".join(pc_active_items[:8])
        extra = f" +{len(pc_active_items) - 8}" if len(pc_active_items) > 8 else ""
        pc_active = f'<div style="font-size:10px;color:#888;margin-top:3px">Active: {items_str}{extra}</div>'

    # Render progress
    rn_ok = gen.get("render_ok", 0)
    rn_fail = gen.get("render_fail", 0)
    rn_done = gen.get("render_done", rn_ok + rn_fail)
    rn_total = gen.get("render_total", pc_ok)  # renderable = precompute_ok
    rn_pct = int(rn_done / rn_total * 100) if rn_total else 0
    rn_avg = f'{gen.get("render_avg_s", 0):.0f}s' if gen.get("render_avg_s") else "—"
    rn_eta = fmt_eta(gen.get("render_eta_s"))
    rn_queue = gen.get("queue_size", 0)

    # Active renders with per-GPU elapsed time
    active_renders = gen.get("active_renders", {})
    rn_active = ""
    if active_renders:
        for gpu_id, info in sorted(active_renders.items()):
            label = info.get("label", "?")
            start = info.get("start_ts", 0)
            elapsed = fmt_elapsed(now_ts - start) if start else "?"
            rn_active += (
                f'<div class="active-item">'
                f'<span class="gpu">GPU{gpu_id}</span> {label} '
                f'<span class="elapsed">({elapsed})</span></div>'
            )

    # Recent events (reversed = newest first)
    recent_events = ""
    events = list(reversed(gen.get("recent_events", [])[-15:]))
    for ev in events:
        ts_str = ""
        if ev.get("ts"):
            ts_str = datetime.fromtimestamp(ev["ts"]).strftime("%H:%M:%S")
        ok_cls = "ok" if ev.get("ok") else "fail"
        icon = "OK" if ev.get("ok") else "FAIL"
        dur_str = f' <span class="dur">{ev["dur_s"]:.0f}s</span>' if ev.get("dur_s") else ""
        recent_events += (
            f'<div class="event">'
            f'<span style="color:#555">{ts_str}</span> '
            f'<span class="type">[{ev.get("type", "?")}]</span> '
            f'<span class="{ok_cls}">{icon}</span> {ev.get("label", "")}'
            f'{dur_str}</div>'
        )
    if not recent_events:
        recent_events = '<div style="font-size:11px;color:#555">No events yet</div>'

    # Elapsed
    elapsed = gen.get("elapsed_s", 0)

    def status_class(s):
        s = str(s).split()[0]
        return s if s in ("running", "stopped", "training", "starting", "finished") else "stopped"

    gen_status = gen.get("status", "stopped")

    html = Template(HTML_TEMPLATE).safe_substitute(
        gpu_rows=gpu_rows,
        train_status=train.get("status", "stopped"),
        train_status_class=status_class(train.get("status")),
        train_step=f'{train.get("step", 0):,}',
        train_loss=f'{train.get("loss", 0):.4f}',
        train_lr=train.get("lr", "—"),
        train_n=f'{train.get("n_train", 0):,}',
        loss_chart=loss_chart,
        gen_status=gen_status,
        gen_status_class=status_class(gen_status),
        gen_total=f'{gen_total:,}' if gen_total else "?",
        gen_elapsed=fmt_elapsed(elapsed),
        gen_phase=gen.get("phase", "—"),
        # Precompute
        pc_done=f'{pc_done:,}',
        pc_pct=pc_pct,
        pc_ok=f'{pc_ok:,}',
        pc_fail=f'{pc_fail:,}',
        pc_avg=pc_avg,
        pc_eta=pc_eta,
        pc_active=pc_active,
        # Render
        rn_done=f'{rn_done:,}',
        rn_total=f'{rn_total:,}',
        rn_pct=rn_pct,
        rn_ok=f'{rn_ok:,}',
        rn_fail=f'{rn_fail:,}',
        rn_avg=rn_avg,
        rn_eta=rn_eta,
        rn_queue=rn_queue,
        rn_active=rn_active,
        # Encode
        enc_status=enc.get("status", "stopped"),
        enc_status_class=status_class(enc.get("status")),
        enc_vae=enc.get("vae_progress", "—"),
        enc_jepa=enc.get("jepa_progress", "—"),
        enc_iter=enc.get("iter", "—"),
        enc_new=enc.get("new_animodes", "—"),
        enc_done=enc.get("already_done", "—"),
        # Events
        recent_events=recent_events,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    return html


# ======================================================================
# Background cache — all data collection runs here, never on request
# ======================================================================

from http.server import ThreadingHTTPServer

_cache_html = b"<h1>Loading...</h1>"
_cache_json = b"{}"
_cache_lock = threading.Lock()


def _refresh_loop():
    """Background thread: refresh cache every 5s. All slow I/O happens here."""
    global _cache_html, _cache_json
    while True:
        try:
            html = render_dashboard().encode()
            api = json.dumps({
                "gpu": get_gpu_status(),
                "train": parse_train_from_log(),
                "gen": get_gen_status(),
                "encode": parse_encode_from_log(),
                "timestamp": datetime.now().isoformat(),
            }, indent=2).encode()
            with _cache_lock:
                _cache_html = html
                _cache_json = api
        except Exception as e:
            err_html = f"<h1>Dashboard error: {e}</h1>".encode()
            with _cache_lock:
                _cache_html = err_html
        time.sleep(5)


# ======================================================================
# Server — only serves cached data, never blocks
# ======================================================================

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        with _cache_lock:
            if self.path == "/api":
                body = _cache_json
                ct = "application/json"
            else:
                body = _cache_html
                ct = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    # Start background cache refresh
    t = threading.Thread(target=_refresh_loop, daemon=True)
    t.start()
    time.sleep(2)  # let first cache fill

    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard running at http://0.0.0.0:{args.port}")
    print(f"  Stats dir: {STATS_DIR}")
    print(f"  Background refresh every 5s, requests served from cache (never blocks)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()
