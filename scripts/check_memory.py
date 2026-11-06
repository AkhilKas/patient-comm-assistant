"""Check the API's peak memory against a budget (Linux only).

Starts the server, exercises every non-LLM endpoint, and reads the peak
resident memory (VmHWM) from /proc. Exits non-zero if it is over budget.

Run: python scripts/check_memory.py [--budget-mb 400]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
# Running a script puts its own directory on sys.path, not the repo root
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from conftest import SAMPLE_DISCHARGE, make_pdf  # noqa: E402

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"


def memory_mb(pid: int) -> dict[str, float]:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        # Not Linux: current RSS only, so "peak" is a lower bound
        rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])) / 1024
        print("⚠ /proc not available; peak memory is approximated by current RSS")
        return {"VmRSS": rss, "VmHWM": rss}
    fields = {}
    for line in status.read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            fields[key] = int(value.split()[0]) / 1024
    return fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-mb", type=float, default=400)
    args = parser.parse_args()

    env = {
        **os.environ,
        # As on Render before a key is added: LLM on, Gemini only, no key.
        # (LLM_PROVIDER=gemini so a full install doesn't load a local model.)
        "ENABLE_LLM": "true",
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "",
        "VECTOR_STORE_PATH": tempfile.mkdtemp(prefix="vs-"),
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
        cwd=ROOT, env=env,
    )
    try:
        with httpx.Client(base_url=BASE, timeout=60) as client:
            for _ in range(120):
                try:
                    if client.get("/").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(1)
            else:
                print("server did not start", file=sys.stderr)
                return 1
            startup = memory_mb(server.pid)

            pdf = make_pdf([line for line in SAMPLE_DISCHARGE.splitlines() if line.strip()][:45])
            for i in range(3):
                r = client.post("/upload", files={"file": (f"doc{i}.pdf", pdf, "application/pdf")})
                r.raise_for_status()
            for q in ["What medicine should I take?", "When should I go to the ER?", "What can I eat?"]:
                client.post("/ask", json={"question": q}).raise_for_status()
            client.post("/readability", json={"text": SAMPLE_DISCHARGE}).raise_for_status()
            client.get("/stats").raise_for_status()
            assert client.post("/simplify", json={"text": "x"}).status_code == 503

            final = memory_mb(server.pid)
    finally:
        server.terminate()
        server.wait(timeout=30)

    peak = final["VmHWM"]
    print(f"startup RSS: {startup['VmRSS']:.0f} MB")
    print(f"final RSS:   {final['VmRSS']:.0f} MB")
    print(f"peak RSS:    {peak:.0f} MB (budget {args.budget_mb:.0f} MB)")
    if peak > args.budget_mb:
        print("✗ over memory budget", file=sys.stderr)
        return 1
    print("✓ within memory budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
