FROM ghcr.io/mlflow/mlflow:v2.9.2

WORKDIR /app

COPY dist/ /app/dist/

EXPOSE 3000

CMD ["python", "-c", "import http.server, os, socketserver\nos.chdir('/app/dist')\nclass H(http.server.SimpleHTTPRequestHandler):\n    def do_GET(self):\n        if not os.path.exists(self.translate_path(self.path)):\n            self.path = '/index.html'\n        return super().do_GET()\nsocketserver.TCPServer.allow_reuse_address = True\nsocketserver.TCPServer(('0.0.0.0', 3000), H).serve_forever()"]
