"""
ARIA — Lifelog LLM Launcher

Thin Python wrapper that execs llama-server.exe (prebuilt CUDA binary).
Called by the orchestrator via _build_cmd; re-invokes the binary with the
same argv so the orchestrator's process tracking works normally.

Must be run from the lifelog-llm conda env (python.exe in envs/lifelog-llm).
The binary is expected at: aria_root/tools/llama/llama-server.exe
"""

import subprocess
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    aria_root = Path(__file__).parent.parent.parent

    llama_exe = aria_root / "tools" / "llama" / "llama-server.exe"
    if not llama_exe.exists():
        print(f"[launcher] ERROR: llama-server.exe not found at {llama_exe}", flush=True)
        print(f"[launcher] Run scripts/install_lifelog_llm.ps1 to download it.", flush=True)
        sys.exit(1)

    cmd = [str(llama_exe)] + sys.argv[1:]
    print(f"[launcher] Starting: {' '.join(cmd[:4])} ...", flush=True)

    # cwd = binary directory so Windows finds ggml-cuda.dll and other DLLs
    proc = subprocess.run(cmd, cwd=str(llama_exe.parent))
    sys.exit(proc.returncode)
