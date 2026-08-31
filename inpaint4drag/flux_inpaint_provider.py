"""FLUX.2 Klein inpainting for the final hole-repair step."""

import threading
import time

import numpy as np
from PIL import Image


DEFAULT_PROMPT = ""


class FluxInpaintProvider:
    default_prompt = DEFAULT_PROMPT
    default_num_inference_steps = 4
    default_strength = 1.0
    default_guidance_scale = 1.0

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
        target_mask,
        source_mask=None,
        prompt=DEFAULT_PROMPT,
        strength=1.0,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=0,
    ):
        effective_steps = max(1, int(num_inference_steps * strength))
        self.set_progress(
            running=True,
            percent=2,
            step=0,
            steps=effective_steps,
            stage="正在准备 Pipeline 输入",
        )
        mask = (inpaint_mask > 0).astype(np.uint8)
        if not mask.any():
            self.set_progress(running=False, percent=0, stage="重绘区域为空")
            raise ValueError("重绘区域为空，请先完成一次拖拽编辑。")
        pipeline_inputs = {
            "image": Image.fromarray(warped_image).convert("RGB"),
            "mask_image": Image.fromarray(mask * 255).convert("L"),
        }
        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = self.torch.Generator(device=generator_device).manual_seed(seed)

        def on_step_end(pipe, step, timestep, callback_kwargs):
            completed = min(step + 1, effective_steps)
            self.set_progress(
                running=True,
                percent=min(95, 10 + round(85 * completed / effective_steps)),
                step=completed,
                steps=effective_steps,
                stage=(
                    "正在解码生成结果"
                    if completed == effective_steps
                    else "正在重绘编辑区域"
                ),
            )
            return callback_kwargs

        started = time.perf_counter()
        try:
            with self.lock, self.torch.inference_mode():
                self.set_progress(
                    running=True,
                    percent=8,
                    stage="正在编码图像与提示词",
                )
                result = self.pipe(
                    prompt=prompt,
                    image=pipeline_inputs["image"],
                    mask_image=pipeline_inputs["mask_image"],
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0].convert("RGB")
            self.set_progress(running=True, percent=98, stage="正在整理生成结果")
            if result.size != (warped_image.shape[1], warped_image.shape[0]):
                result = result.resize(
                    (warped_image.shape[1], warped_image.shape[0]),
                    Image.Resampling.LANCZOS,
                )
            generated = np.array(result)
            result = warped_image.copy()
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
        return (
            result,
            (time.perf_counter() - started) * 1000,
            pipeline_inputs,
        )
