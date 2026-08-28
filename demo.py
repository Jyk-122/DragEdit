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
from flux_inpaint_provider import DEFAULT_PROMPT, FluxInpaintProvider
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
FLUX_PROVIDER = None


def refine_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask refinement hook. The current demo keeps the input mask unchanged."""
    return mask


class DemoHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

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
                "generation_model": FLUX_PROVIDER.model_id,
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
            kernel_size = int(request.get("inpaint_kernel_size", 5))
            session = WARP_SESSIONS[algorithm]
            started = time.perf_counter()
            if algorithm == "baseline":
                preview, inpaint_mask, target_mask = session.preview(
                    point_pairs, kernel_size
                )
            else:
                preview, inpaint_mask, _, target_mask = session.preview(
                    point_pairs, request["keep_boundary"], kernel_size
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
                "X-Inpaint-Kernel-Size": str(kernel_size),
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

        if self.path == "/api/generate":
            try:
                generated, inference_ms, pipeline_inputs = generate_image(request)
                encoded = cv2.imencode(
                    ".png", cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)
                )[1]
                self.send_json({
                    "image": "data:image/png;base64," +
                             base64.b64encode(encoded).decode(),
                    "inference_ms": inference_ms,
                    "model": FLUX_PROVIDER.model_id,
                    "pipeline_inputs": {
                        name: encode_pil_debug_image(image)
                        for name, image in pipeline_inputs.items()
                    },
                })
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
            except Exception as error:
                self.send_json({"error": str(error)}, 500)
            return

        self.send_error(404)

    def read_json(self):
        length = int(self.headers["Content-Length"])
        return json.loads(self.rfile.read(length))

    def send_json(self, response, status=200):
        body = json.dumps(response).encode()
        self.send_response(status)
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


def encode_pil_debug_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "image": "data:image/png;base64," +
                 base64.b64encode(buffer.getvalue()).decode(),
        "width": image.width,
        "height": image.height,
    }


def compose_ghost_preview(image, warped, target_mask, inpaint_mask, opacity):
    """Overlay warped pixels and revealed holes while keeping the source visible."""
    result = image.copy()
    region = (target_mask > 0) | (inpaint_mask > 0)
    result[region] = (
        image[region] * (1 - opacity) + warped[region] * opacity
    ).astype(np.uint8)
    return result


def generate_image(request):
    """Resolve browser/server warp inputs and run FLUX.2 Klein inpainting."""
    if "image_reference" in request:
        image = decode_data_url(request["image"], "RGB")
        warped_image = decode_data_url(request["image_reference"], "RGB")
        inpaint_mask = decode_data_url(request["inpaint_mask"], "L")
        target_mask = decode_data_url(request["target_mask"], "L")
        source_mask = decode_data_url(request["source_mask"], "L")
    else:
        algorithm = request.get("algorithm", "baseline")
        session = WARP_SESSIONS[algorithm]
        if "point_pairs" in request:
            point_pairs = [
                ([pair["source"]["x"], pair["source"]["y"]],
                 [pair["target"]["x"], pair["target"]["y"]])
                for pair in request["point_pairs"]
            ]
            kernel_size = int(request.get("inpaint_kernel_size", 5))
            if algorithm == "baseline":
                session.preview(point_pairs, kernel_size)
            else:
                session.preview(
                    point_pairs, request.get("keep_boundary", True), kernel_size
                )
        image = session.image
        warped_image = session.warped_image
        inpaint_mask = session.inpaint_mask
        target_mask = session.target_mask
        source_mask = session.mask
        if image is None or warped_image is None:
            raise ValueError("请先加载图像并完成一次拖拽编辑。")

    return FLUX_PROVIDER.generate(
        image=image,
        warped_image=warped_image,
        inpaint_mask=inpaint_mask,
        target_mask=target_mask,
        source_mask=source_mask,
        prompt=request.get("prompt") or DEFAULT_PROMPT,
        strength=float(request.get("strength", 1.0)),
        num_inference_steps=int(request.get("num_inference_steps", 4)),
        guidance_scale=float(request.get("guidance_scale", 1.0)),
        seed=int(request.get("seed", 0)),
        padding_mask_crop=int(request.get("padding_mask_crop", 64)),
    )


def main():
    global SAM_PROVIDER, FLUX_PROVIDER
    parser = argparse.ArgumentParser(description="DragEdit local interaction demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument(
        "--flux-model", default="black-forest-labs/FLUX.2-klein-4B"
    )
    parser.add_argument("--flux-device", default="cuda")
    parser.add_argument("--flux-cache-dir")
    parser.add_argument("--flux-cpu-offload", action="store_true")
    args = parser.parse_args()

    SAM_PROVIDER = SamMaskProvider(
        args.sam_checkpoint, args.sam_model_type, args.sam_device
    )
    print(f"Loading generation model: {args.flux_model}")
    FLUX_PROVIDER = FluxInpaintProvider(
        args.flux_model,
        args.flux_device,
        args.flux_cache_dir,
        args.flux_cpu_offload,
    )
    print(f"Generation model ready on {FLUX_PROVIDER.device}")
    print(f"Web assets: {WEB_DIR.resolve()}")

    handler = partial(DemoHandler, directory=WEB_DIR)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"DragEdit demo: {url}")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
