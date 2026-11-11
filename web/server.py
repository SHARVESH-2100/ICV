"""AI Product Authentication Using Logo & Packaging Recognition — web API.

Standard-library HTTP server (no extra dependencies) that reuses the existing
scoring pipeline. Start it from the project root:

    python web/server.py [--port 8000]

Endpoints
    GET    /                        -> web/index.html
    GET    /css/, /js/              -> static assets
    GET    /api/info                -> engine + config status
    GET    /api/products            -> reference library listing
    GET    /api/reference           -> one reference image (?product=&file=)
    POST   /api/products            -> add reference images (multipart)
    DELETE /api/products/<name>     -> delete a product folder
    POST   /api/verify              -> run the assessment (multipart)
"""

import argparse
import base64
import html
import json
import mimetypes
import re
import shutil
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import config  # noqa: E402
from dataset import IMG_EXTS, list_products, product_images, save_reference  # noqa: E402
from preprocess import imread_bytes  # noqa: E402
from scoring import Assessor, reasons  # noqa: E402

MAX_UPLOAD = 25 * 1024 * 1024
_assessor = None
_assess_lock = threading.Lock()   # serialises assessments (torch is not thread-safe)
_init_lock = threading.Lock()     # guards one-time model load


def get_assessor():
    """Load the Siamese checkpoint once and share it across requests."""
    global _assessor
    with _init_lock:
        if _assessor is None:
            _assessor = Assessor()
        return _assessor


# ── multipart/form-data ──────────────────────────────────────
def parse_multipart(body, content_type):
    """Return {field_name: {"filename": str|None, "data": bytes}}."""
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not m:
        raise ValueError("Missing multipart boundary")
    delim = b"--" + m.group(1).encode()
    out = {}
    for part in body.split(delim):
        if part in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        part = part.lstrip(b"\r\n")
        if part.startswith(b"--"):
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        if not _:
            continue
        data = data[:-2] if data.endswith(b"\r\n") else data
        name = filename = None
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                nm = re.search(r'name="([^"]*)"', line)
                fn = re.search(r'filename="([^"]*)"', line)
                name = nm.group(1) if nm else None
                filename = fn.group(1) if fn else None
        if name:
            out[name] = {"filename": filename, "data": data}
    return out


# ── image helpers ────────────────────────────────────────────
def png_data_url(bgr, max_side=900):
    """Encode a BGR array as a base64 PNG data URL, downscaled if large."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def safe_image_name(name):
    name = Path(unquote(name)).name.strip()
    if not name:
        raise ValueError("Missing file name")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    stem = stem.lstrip(".") or "image"
    if not stem.lower().endswith(IMG_EXTS):
        stem += ".jpg"
    return stem[:80]


def library_summary():
    products = []
    total = 0
    for name in list_products():
        refs = [p.name for p in product_images(name)]
        total += len(refs)
        products.append({"name": name, "count": len(refs), "references": refs})
    return products, total


def named_references(product, max_n=6):
    """Load up to max_n reference images, keeping each file name alongside."""
    loaded = []
    for path in product_images(product)[:max_n]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            loaded.append((path.name, img))
    return loaded


def info_payload():
    from datetime import datetime

    assessor = get_assessor()
    _, total = library_summary()
    exists = config.MODEL_PATH.exists()
    return {
        "model_loaded": assessor.model is not None,
        "device": assessor.device,
        "model_path": str(config.MODEL_PATH),
        "model_mtime": datetime.fromtimestamp(config.MODEL_PATH.stat().st_mtime).strftime("%Y-%m-%d") if exists else None,
        "threshold": config.BINARY_THRESHOLD,
        "verdicts": [{"min": t, "name": n} for t, n in config.VERDICTS],
        "weights": config.WEIGHTS,
        "config": {
            "IMAGE_SIZE": config.IMAGE_SIZE,
            "SIAMESE_SIZE": config.SIAMESE_SIZE,
            "EMBED_DIM": config.EMBED_DIM,
            "ORB_FEATURES": config.ORB_FEATURES,
            "ORB_RATIO_THRESHOLD": config.ORB_RATIO_THRESHOLD,
            "GRID": list(config.GRID),
            "SIAMESE_MARGIN": config.SIAMESE_MARGIN,
            "BINARY_THRESHOLD": config.BINARY_THRESHOLD,
        },
        "products": len(list_products()),
        "images": total,
        "genuine_dir": str(config.GENUINE_DIR),
    }


# ── request handler ──────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "ProductAuthWeb/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))
        sys.stderr.flush()

    # ── response helpers ──
    def _send(self, code, body=b"", ctype="application/octet-stream", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=300")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def json(self, payload, code=200):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def fail(self, message, code=400):
        self.json({"error": html.escape(str(message))}, code)

    def body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_UPLOAD:
            raise ValueError("Upload exceeds the 25 MB limit")
        return self.rfile.read(length)

    def form(self):
        return parse_multipart(self.body(), self.headers.get("Content-Type", ""))

    # ── static ──
    def serve_static(self, rel):
        target = (WEB_DIR / rel).resolve()
        if not str(target).startswith(str(WEB_DIR)) or not target.is_file():
            return self.fail("Not found", 404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith(("text/", "application/javascript")):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    # ── routes ──
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        route, params = url.path, parse_qs(url.query)
        try:
            if route in ("/", "/index.html"):
                return self.serve_static("index.html")
            if route.startswith(("/css/", "/js/")):
                return self.serve_static(route.lstrip("/"))

            if route == "/api/info":
                return self.json(info_payload())
            if route == "/api/products":
                products, _ = library_summary()
                return self.json({"products": products})
            if route == "/api/reference":
                return self.serve_reference(params)

            return self.fail("Not found", 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.fail(exc, 500)

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            if route == "/api/verify":
                return self.verify()
            if route == "/api/products":
                return self.add_references()
            return self.fail("Not found", 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.fail(exc, 500)

    def do_DELETE(self):
        route = unquote(urlparse(self.path).path)
        try:
            if route.startswith("/api/products/"):
                name = route[len("/api/products/"):].strip("/")
                folder = (config.GENUINE_DIR / name).resolve()
                if not name or not str(folder).startswith(str(config.GENUINE_DIR.resolve())) or not folder.is_dir():
                    return self.fail("Product not found", 404)
                shutil.rmtree(folder)
                return self.json({"deleted": name})
            return self.fail("Not found", 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.fail(exc, 500)

    # ── handlers ──
    def serve_reference(self, params):
        product = (params.get("product") or [""])[0]
        filename = (params.get("file") or [""])[0]
        path = (config.GENUINE_DIR / product / filename).resolve()
        root = config.GENUINE_DIR.resolve()
        if not str(path).startswith(str(root)) or not path.is_file():
            return self.fail("Reference not found", 404)
        ctype = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        self._send(200, path.read_bytes(), ctype)

    def verify(self):
        fields = self.form()
        product = (fields.get("product", {}).get("data") or b"").decode("utf-8", "replace").strip()
        upload = fields.get("image")
        if not product:
            return self.fail("No product selected")
        if not upload or not upload["data"]:
            return self.fail("No image uploaded")
        if product not in list_products():
            return self.fail(f"Unknown product: {product}", 404)

        query = imread_bytes(upload["data"])
        if query is None:
            return self.fail("Could not decode the uploaded image")

        refs = named_references(product)
        if not refs:
            return self.fail("No reference images for this product", 409)

        started = time.perf_counter()
        assessor = get_assessor()
        with _assess_lock:
            result = assessor.assess([img for _, img in refs], query)
            notes = reasons(result)
        elapsed = (time.perf_counter() - started) * 1000.0

        metrics = dict(result["metrics"])
        for key in ("heat", "qr_ref", "qr_query"):
            metrics.pop(key, None)

        ref_name = next((name for name, img in refs if img is result["ref"]), "")

        return self.json({
            "result": {
                "score": float(result["score"]),
                "verdict": result["verdict"],
                "binary": result["binary"],
                "color": result["color"],
                "scores": {k: float(v) for k, v in result["scores"].items()},
                "metrics": {k: v for k, v in metrics.items() if not isinstance(v, (bytes, bytearray))},
            },
            "reference": ref_name,
            "reference_url": f"/api/reference?product={product}&file={ref_name}" if ref_name else "",
            "images": {
                "query": png_data_url(result["query"]),
                "overlay": png_data_url(result["overlay"]),
            },
            "reasons": notes,
            "elapsed_ms": round(elapsed, 1),
        })

    def add_references(self):
        fields = self.form()
        product = (fields.get("product", {}).get("data") or b"").decode("utf-8", "replace").strip()
        files = [v for k, v in fields.items() if k == "files" and v["data"]]
        if not product:
            return self.fail("Product name is required")
        if not files:
            return self.fail("Select at least one image")

        saved = []
        for item in files:
            name = safe_image_name(item["filename"] or "reference.jpg")
            path = save_reference(product, name, item["data"])
            saved.append(path.name)

        return self.json({
            "message": f"Added {len(saved)} reference image(s) to “{product}”.",
            "product": product,
            "files": saved,
        }, 201)


def main():
    parser = argparse.ArgumentParser(description="Product authentication web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading engine (this can take a moment)… ", end="", flush=True)
    get_assessor()
    print("ready.")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  AI Product Authentication - Logo & Packaging Recognition  ->  http://{args.host}:{args.port}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()
