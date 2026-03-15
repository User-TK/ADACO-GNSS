import json
import random
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
THRESHOLD = 90.0


def placeholder_model():
    """Return fake percentage outputs from 0 to 100 for several categories."""
    return {
        "Reliability": round(random.uniform(0, 100), 2),
        "Safety": round(random.uniform(0, 100), 2),
        "Confidence": round(random.uniform(0, 100), 2),
        "Stability": round(random.uniform(0, 100), 2),
        "Performance": round(random.uniform(0, 100), 2),
    }


def placeholder_alert(values, threshold=THRESHOLD):
    """Return which categories are above the threshold.
    In a real system this might send a notification, log an incident, or trigger another action.
    For this demo it drives the visible alert banner and alert log in the page.
    """
    return [name for name, value in values.items() if value > threshold]


class DemoHandler(BaseHTTPRequestHandler):
    def _send_text_file(self, filename, content_type):
        path = BASE_DIR / filename
        if not path.exists():
            self.send_error(404, f"File not found: {filename}")
            return

        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self._send_text_file("index.html", "text/html; charset=utf-8")
        elif parsed.path == "/styles.css":
            self._send_text_file("styles.css", "text/css; charset=utf-8")
        elif parsed.path == "/api/feed":
            values = placeholder_model()
            alerts = placeholder_alert(values)
            self._send_json(
                {
                    "timestamp": datetime.now().isoformat(),
                    "threshold": THRESHOLD,
                    "values": values,
                    "alerts": alerts,
                }
            )
        else:
            self.send_error(404, "Not found")

    def log_message(self, format, *args):
        # Keep console output tidy.
        return


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    server = HTTPServer((host, port), DemoHandler)
    print(f"Serving demo on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()
