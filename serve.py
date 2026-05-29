#!/usr/bin/env python3
"""
Servidor HTTP local para index.html (evita bloqueo CORS de file:// al cargar JSON).
Uso: python serve.py
     Abre http://127.0.0.1:8000/index.html
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8000


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    with ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"[serve] http://127.0.0.1:{PORT}/index.html")
        print(f"[serve] Raíz: {ROOT}")
        print("[serve] Ctrl+C para detener")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
