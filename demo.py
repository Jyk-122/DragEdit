import argparse
import base64
import io
import json
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from baseline_warp import BaselineWarpSession
from my_warp import MyWarpSession
from sam_provider import SamMaskProvider


ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"
WARP_SESSIONS = {
    "baseline": BaselineWarpSession(),
    "my_warp": MyWarpSession(),
}
LAST_SESSION = WARP_SESSIONS["baseline"]
SAM_PROVIDER = SamMaskProvider()


def refine_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask refinement hook. The current demo keeps the input mask unchanged."""
    return mask


class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/inpaint-mask":
            encoded = cv2.imencode(".png", LAST_SESSION.inpaint_mask * 255)[1]
            self.send_bytes(encoded.tobytes(), "image/png")
            return
        super().do_GET()

    def do_POST(self):
        request = self.read_json()
        if self.path == "/api/image":
            image = decode_data_url(request["image"], "RGB")
            for session in WARP_SESSIONS.values():
                session.set_image(image)
            started = time.perf_counter()
            SAM_PROVIDER.set_image(image)
            self.send_json({
                "width": image.shape[1],
                "height": image.shape[0],
                "sam_ready": SAM_PROVIDER.enabled,
                "sam_preprocess_ms": (time.perf_counter() - started) * 1000,
            })
            return

        if self.path == "/api/mask":
            mask = decode_data_url(request["mask"], "L")
            mask = refine_mask(WARP_SESSIONS["baseline"].image, mask)
            for session in WARP_SESSIONS.values():
                session.set_mask(mask)
            self.send_json({"pixels": int(WARP_SESSIONS["baseline"].mask.sum())})
            return

        if self.path == "/api/warp":
            global LAST_SESSION
            point_pairs = [
                ([pair["source"]["x"], pair["source"]["y"]],
                 [pair["target"]["x"], pair["target"]["y"]])
                for pair in request["point_pairs"]
            ]
            algorithm = request.get("algorithm", "baseline")
            session = WARP_SESSIONS[algorithm]
            started = time.perf_counter()
            if algorithm == "baseline":
                preview, inpaint_mask, target_mask = session.preview(point_pairs)
            else:
                preview, inpaint_mask, _, target_mask = session.preview(
                    point_pairs, request["keep_boundary"]
                )
            LAST_SESSION = session
            preview = compose_ghost_preview(
                session.image, preview, target_mask, inpaint_mask,
                request.get("preview_opacity", 0.72),
            )
            encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92]
            )[1]
            headers = {
                "X-Warp-Ms": f"{(time.perf_counter() - started) * 1000:.1f}",
                "X-Inpaint-Pixels": str(int(inpaint_mask.sum())),
                "X-Target-Mask-Pixels": str(int(target_mask.sum())),
                "X-Warp-Algorithm": algorithm,
            }
            self.send_bytes(encoded.tobytes(), "image/jpeg", headers)
            return

        if self.path == "/api/sam-mask":
            mask = SAM_PROVIDER.select(request["x"], request["y"])
            encoded = cv2.imencode(".png", mask * 255)[1]
            self.send_json({
                "mask": "data:image/png;base64," +
                        base64.b64encode(encoded).decode()
            })
            return

        self.send_error(404)

    def read_json(self):
        length = int(self.headers["Content-Length"])
        return json.loads(self.rfile.read(length))

    def send_json(self, response):
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body, content_type, headers=None):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def decode_data_url(data_url, mode):
    data = base64.b64decode(data_url.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(data)).convert(mode))


def compose_ghost_preview(image, warped, target_mask, inpaint_mask, opacity):
    """Overlay warped pixels and revealed holes while keeping the source visible."""
    result = image.copy()
    region = (target_mask > 0) | (inpaint_mask > 0)
    result[region] = (
        image[region] * (1 - opacity) + warped[region] * opacity
    ).astype(np.uint8)
    return result


def main():
    global SAM_PROVIDER
    parser = argparse.ArgumentParser(description="DragEdit local interaction demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-device")
    args = parser.parse_args()

    SAM_PROVIDER = SamMaskProvider(
        args.sam_checkpoint, args.sam_model_type, args.sam_device
    )

    handler = partial(DemoHandler, directory=WEB_DIR)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"DragEdit demo: {url}")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
