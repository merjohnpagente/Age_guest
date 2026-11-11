#!/usr/bin/env python3
"""Local development server for FACELENS.

The production site is fully static (GitHub Pages). This small Python server
exists because some browsers refuse camera access on file:// pages. It adds
the correct MIME types for the local WASM and TFLite assets.

    python3 serve.py            # http://localhost:8000
    python3 serve.py --port 8080
"""

from __future__ import annotations

import argparse
import http.server
import mimetypes
import os
import socketserver
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Python's mimetypes table is incomplete on some systems.
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/octet-stream", ".tflite")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/svg+xml", ".svg")


class FaceLensHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".mjs": "text/javascript",
        ".tflite": "application/octet-stream",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        if args and str(args[1]).startswith(("2", "3")):
            return
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve FACELENS locally over HTTP.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    handler = partial(FaceLensHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
        print(f"FACELENS running at http://{args.host}:{args.port}")
        print("Camera access is allowed on localhost. Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
