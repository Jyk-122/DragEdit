"""Z-Image inpainting for the final hole-repair step."""

import threading
import time

import numpy as np
from PIL import Image


RESAMPLING = getattr(Image, "Resampling", Image)


class ZImageInpaintProvider:
    default_prompt = ""
    default_num_inference_steps = 8
    default_strength = 1.0
    default_guidance_scale = 0.0

    def __init__(
        self,
        model_id="Tongyi-MAI/Z-Image-Turbo",
        device=None,
        cache_dir=None,
        cpu_offload=False,
    ):
        import torch
        from diffusers import ZImageInpaintPipeline

        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.pipe = ZImageInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        self.dimension_multiple = self.pipe.vae_scale_factor * 2
        self.lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.progress = {
            "running": False,
            "percent": 0,
            "step": 0,
            "steps": 0,
            "stage": "等待生成",
        }

    def set_progress(self, **values):
        with self.progress_lock:
            self.progress.update(values)

    def get_progress(self):
        with self.progress_lock:
            return self.progress.copy()

    def generate(
        self,
        image,
        warped_image,
        inpaint_mask,
        target_mask=None,
        source_mask=None,
        prompt="",
        strength=1.0,
        num_inference_steps=8,
        guidance_scale=0.0,
        seed=0,
    ):
        mask = (inpaint_mask > 0).astype(np.uint8)
        if not mask.any():
            self.set_progress(running=False, percent=0, stage="重绘区域为空")
            raise ValueError("重绘区域为空，请先完成一次拖拽编辑。")

        original_height, original_width = mask.shape
        multiple = self.dimension_multiple
        width = max(multiple, round(original_width / multiple) * multiple)
        height = max(multiple, round(original_height / multiple) * multiple)
        image_pil = Image.fromarray(warped_image).convert("RGB")
        mask_pil = Image.fromarray(mask * 255).convert("L")
        if (width, height) != image_pil.size:
            image_pil = image_pil.resize((width, height), RESAMPLING.LANCZOS)
            mask_pil = mask_pil.resize((width, height), RESAMPLING.NEAREST)

        pipeline_inputs = {
            "image": image_pil,
            "mask_image": mask_pil,
        }
        skipped_steps = int(max(num_inference_steps * (1 - strength), 0))
        effective_steps = max(1, num_inference_steps - skipped_steps)
        self.set_progress(
            running=True,
            percent=5,
            step=0,
            steps=effective_steps,
            stage="正在准备 Z-Image Inpainting",
        )

        def on_step_end(pipe, step, timestep, callback_kwargs):
            completed = min(step + 1, effective_steps)
            self.set_progress(
                running=True,
                percent=min(95, 10 + round(85 * completed / effective_steps)),
                step=completed,
                steps=effective_steps,
                stage="正在重绘空洞区域",
            )
            return callback_kwargs

        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = self.torch.Generator(device=generator_device).manual_seed(seed)

        started = time.perf_counter()
        try:
            with self.lock, self.torch.inference_mode():
                generated = self.pipe(
                    prompt=prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                    height=height,
                    width=width,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0].convert("RGB")
            if generated.size != (original_width, original_height):
                generated = generated.resize(
                    (original_width, original_height), RESAMPLING.LANCZOS
                )
            result = warped_image.copy()
            generated = np.array(generated)
            result[mask > 0] = generated[mask > 0]
            self.set_progress(
                running=False,
                percent=100,
                step=effective_steps,
                steps=effective_steps,
                stage="生成完成",
            )
        except Exception:
            self.set_progress(running=False, stage="生成失败")
            raise

        return result, (time.perf_counter() - started) * 1000, pipeline_inputs
