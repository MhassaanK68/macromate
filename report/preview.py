"""Export main.pdf and open an automatically rebuilding browser preview."""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE / "main.pdf"
FIGURES = HERE.parent / "figures"
state = {"version": 0, "error": ""}
lock = threading.Lock()


def build():
    engine = next((name for name in ("latexmk", "pdflatex", "tectonic") if shutil.which(name)), None)
    if engine is None:
        raise RuntimeError(
            "No LaTeX compiler executable found. pip install pdflatex installs only a "
            "wrapper; install a TeX distribution with pdflatex, latexmk, or tectonic on PATH."
        )
    options = {
        "latexmk": ["-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        "pdflatex": ["-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        "tectonic": ["main.tex"],
    }
    for _ in range(2 if engine == "pdflatex" else 1):
        result = subprocess.run([engine, *options[engine]], cwd=HERE, capture_output=True,
                                text=True, errors="replace", timeout=180, check=False)
        if result.returncode:
            raise RuntimeError("LaTeX compilation failed:\n" +
                               (result.stdout + "\n" + result.stderr)[-3000:])
    if not PDF.is_file() or PDF.stat().st_size == 0:
        raise RuntimeError("Compilation did not produce report/main.pdf")
    with lock:
        state["version"] += 1
        state["error"] = ""
    print(f"PDF ready: {PDF}", flush=True)


def source_state():
    paths = [HERE / "main.tex", *FIGURES.glob("*.png")]
    return tuple((str(p), p.stat().st_mtime_ns) for p in paths if p.exists())


def watch_sources():
    previous = source_state()
    while True:
        time.sleep(1)
        current = source_state()
        if current == previous:
            continue
        previous = current
        try:
            build()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            with lock:
                state["error"] = str(exc)
            print(exc, file=sys.stderr, flush=True)


HTML = b'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Macromate report preview</title>
<style>body{margin:0;font:14px system-ui;background:#f4f4f4}header{padding:10px 16px;background:#fff;border-bottom:1px solid #ccc}#status{margin-left:1em;color:#555}iframe{width:100%;height:calc(100vh - 46px);border:0}</style>
<header>Macromate technical report <span id="status">Loading...</span></header>
<iframe id="pdf" title="Report PDF"></iframe>
<script>
let shown=0;
async function refresh(){
  try{
    const response=await fetch('/status',{cache:'no-store'});
    const value=await response.json();
    document.getElementById('status').textContent=value.error||'Live preview';
    if(value.version!==shown){shown=value.version;document.getElementById('pdf').src='/main.pdf?v='+shown;}
  }catch{document.getElementById('status').textContent='Preview server disconnected';}
}
refresh();setInterval(refresh,1500);
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body, kind = HTML, "text/html; charset=utf-8"
        elif path == "/status":
            with lock:
                body = json.dumps(state).encode()
            kind = "application/json"
        elif path == "/main.pdf" and PDF.is_file():
            body, kind = PDF.read_bytes(), "application/pdf"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="export once without opening a browser")
    args = parser.parse_args()
    try:
        build()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"{exc}\n")
    if args.build:
        return
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Preview: {url} (Ctrl+C to stop)", flush=True)
    threading.Thread(target=watch_sources, daemon=True).start()
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
