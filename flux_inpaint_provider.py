"""FLUX.2 Klein reference-conditioned inpainting for the final refine step."""

import threading
import time

import cv2
import numpy as np
from PIL import Image


DEFAULT_PROMPT = """Use the image reference as the exact target layout. Re-render the edited region so that the result follows the local deformation, displacement, scale, and rotation shown in the reference image. Preserve the object's identity and intended geometry while repairing warp-induced stretching, blur, tearing, holes, seams, duplicated content, and other implausible regions. Keep the surrounding image unchanged and blend the repaired region naturally with the background."""


def build_repair_mask(inpaint_mask, target_mask):
    """The whole warped object and its repair band are generated together."""
    return ((inpaint_mask > 0) | (target_mask > 0)).astype(np.uint8)


def prepare_image_reference(warped_image, source_mask, target_mask, padding=64):
    """Create a local clean warp crop for image-reference conditioning."""
    reference = warped_image.copy()
    if source_mask is not None:
        revealed = (source_mask > 0) & ~(target_mask > 0)
        if revealed.any():
            filled = cv2.inpaint(
                cv2.cvtColor(reference, cv2.COLOR_RGB2BGR),
                revealed.astype(np.uint8) * 255,
                3,
                cv2.INPAINT_TELEA,
            )
            reference = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
            reference[target_mask > 0] = warped_image[target_mask > 0]

    y, x = np.where(target_mask > 0)
    if len(x) == 0:
        raise ValueError("目标区域为空，请先完成一次拖拽编辑。")
    height, width = target_mask.shape
    left = max(0, int(x.min()) - padding)
    top = max(0, int(y.min()) - padding)
    right = min(width, int(x.max()) + padding + 1)
    bottom = min(height, int(y.max()) + padding + 1)
    return reference[top:bottom, left:right]


class FluxInpaintProvider:
    def __init__(
        self,
        model_id="black-forest-labs/FLUX.2-klein-4B",
        device=None,
        cache_dir=None,
        cpu_offload=False,
    ):
        import torch
        from diffusers import Flux2KleinInpaintPipeline

        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.pipe = Flux2KleinInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.lock = threading.Lock()

    def generate(
        self,
        image,
        warped_image,
        inpaint_mask,
        target_mask,
        source_mask=None,
        prompt=DEFAULT_PROMPT,
        strength=1.0,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=0,
        padding_mask_crop=64,
    ):
        repair_mask = build_repair_mask(inpaint_mask, target_mask)
        if not repair_mask.any():
            raise ValueError("重绘区域为空，请先完成一次拖拽编辑。")
        image_reference = prepare_image_reference(
            warped_image, source_mask, target_mask, padding_mask_crop
        )
        pipeline_inputs = {
            "image": Image.fromarray(image).convert("RGB"),
            "image_reference": Image.fromarray(image_reference).convert("RGB"),
            "mask_image": Image.fromarray(repair_mask * 255).convert("L"),
        }
        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = self.torch.Generator(device=generator_device).manual_seed(seed)

        started = time.perf_counter()
        with self.lock, self.torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                image=pipeline_inputs["image"],
                image_reference=pipeline_inputs["image_reference"],
                mask_image=pipeline_inputs["mask_image"],
                padding_mask_crop=padding_mask_crop,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0].convert("RGB")
        if result.size != (image.shape[1], image.shape[0]):
            result = result.resize((image.shape[1], image.shape[0]), Image.Resampling.LANCZOS)
        return (
            np.array(result),
            (time.perf_counter() - started) * 1000,
            pipeline_inputs,
        )
