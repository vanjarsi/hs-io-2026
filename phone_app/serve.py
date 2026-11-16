#!/usr/bin/env python3
"""Serve the phone app on this computer only (127.0.0.1).

    python3 serve.py [port]            # default port 8000

It is a plain static file server (Python standard library only) with two additions:
  * Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy headers, which
    let the browser use multi-threaded WebAssembly (2-3x faster inference).
    Pass --no-isolation to turn them off.
  * no-cache headers, so a model swap shows up on the next page load.

It listens on 127.0.0.1, so nothing else on the network can reach it. To open
it on a phone, put an HTTPS tunnel in front of it (see README.md).
"""
import functools
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Only these are served. Model weights (*.pt), the venv and scripts stay private.
ALLOWED_FILES = {"index.html", "app.js", "engine.js", "worker.js", "style.css", "favicon.ico"}
ALLOWED_DIRS = {"model": {".onnx", ".json"}, "model_final": {".onnx", ".json"}, "samples": {".jpg", ".jpeg", ".png", ".json"}}
ISOLATE = "--no-isolation" not in sys.argv


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".onnx": "application/octet-stream", ".wasm": "application/wasm", ".js": "text/javascript"}

    def allowed(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        parts = [p for p in path.split("/") if p]
        if not parts:
            return True
        if len(parts) == 1:
            return parts[0] in ALLOWED_FILES
        if len(parts) == 2 and parts[0] in ALLOWED_DIRS:
            return Path(parts[1]).suffix.lower() in ALLOWED_DIRS[parts[0]]
        return False

    def do_GET(self):
        if not self.allowed():
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if not self.allowed():
            self.send_error(404)
            return
        super().do_HEAD()

    def end_headers(self):
        if ISOLATE:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))


def main():
    port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8000)
    handler = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving {ROOT} at http://127.0.0.1:{port}  (cross-origin isolation: {'on' if ISOLATE else 'off'})")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
