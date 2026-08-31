# DragEdit 生成重绘记录 - Dev_2

更新时间：2026-08-31

本文记录 Warp 交互结果接入 Inpaint Provider 后的数据流、界面和运行参数。
后续开发应先阅读根目录 `AGENTS.md`、`documents/dev_0.md` 和 `documents/dev_1.md`。

## 1. 当前完整链路

Demo 现在包含三个连续阶段：

1. 选择图像和对象 Mask。
2. 通过整体变换或 point pairs 得到 Warp 编辑结果。
3. 使用可切换的 Inpaint Provider 完成空洞和融合带重绘。

默认 Provider 为 SD1.5 Inpainting + LCM LoRA + Tiny VAE；FLUX.2 Klein 与 Z-Image-Turbo
作为可选 Provider。`demo.py` 在创建 HTTP 服务前加载所选模型，使页面可访问时生成管线
已经就绪。

## 2. 生成模型输入

SD1.5、FLUX 和 Z-Image Provider 使用相同的 Inpaint4Drag 输入语义：

```text
image      = 干净 Warp 合成图
mask_image = inpaint_mask
```

Pipeline 结果按全分辨率二值 Mask 与 Warp 图合成，Mask 外像素保持不变。三个 Provider
默认使用空提示词；SD1.5 会缓存空提示词 embedding。

SD1.5 默认推理参数：

| 参数 | 默认值 |
|---|---:|
| `num_inference_steps` | 8 |
| `strength` | 1.0 |
| `guidance_scale` | 1.0 |
| `seed` | 0 |

## 3. Warp 图构造

棋盘格只属于交互预览，不进入生成模型。

- `BaselineWarpSession` 和 `MyWarpSession` 新增 `warped_image`，保存原图与 target object
  合成后的干净图像；公开 `warp_preview()` 返回结构保持不变。
- 浏览器整体变换使用独立 `transformReferenceCanvas` 保存应用位移、旋转和缩放后的
  干净合成图，再复制一份用于叠加棋盘格预览。
- 三个 Provider 都直接使用干净 Warp 合成图作为 `image`，使用原始 `inpaint_mask` 作为
  `mask_image`。

## 4. HTTP 接口

### `POST /api/generate`

整体变换由浏览器提交：

- `image`
- `warped_image`
- `inpaint_mask`
- `target_mask`
- `source_mask`

非刚性变形提交 Warp 算法、point pairs、边界锚点和 kernel size。服务端先用对应 Session
重算当前 Warp，再直接读取 `image`、`warped_image`、`inpaint_mask`、`target_mask` 和
`source_mask`，保证生成输入对应按钮点击时的编辑状态。

接口返回生成 PNG Data URL、模型名、推理耗时，以及实际传给所选 Pipeline 的 PNG 调试图
和各自尺寸。三个 Provider 均返回 `image` 与 `mask_image`。

### `GET /api/generation-progress`

返回当前生成任务的阶段、百分比、已完成 step 和总 step。Provider 通过
`callback_on_step_end` 在每个实际去噪 step 后更新状态；前端生成期间每 400 ms 轮询一次。
图像准备、条件编码、去噪和结果整理分别显示阶段信息，去噪区间按实际 step 分段推进。

## 5. 页面布局

页面使用约 2:4:4 的三列工作区：

- 左侧为纵向编辑操作台，集中图像、Mask、拖拽和生成控件，并使用紧凑字号。
- 中间为 Mask 与拖拽编辑可视区。
- 右侧为原图/生成图对比可视区。

加载图片后会按宽高比自动调整两个可视区：纵向图片使用上下排列，横向或方形图片使用
左右排列。切换图片或预览长边时会重新判断方向并计算 Canvas 显示尺寸。

“生成图片”按钮始终可点击，并在状态栏提示尚缺少的图像、Mask 或拖拽编辑结果；仅在
模型推理期间临时锁定，避免重复提交。

对比台以生成图为底层，原图覆盖左侧区域。竖线向右拖动会显示更多原图，向左拖动会
显示更多生成图。

生成完成后，对比台下方显示所选 Provider 实际传入 Pipeline 的输入缩略图。这些图直接由
Provider 使用的 PIL 对象编码，便于核对 Warp 图和白色重绘区域。

两个缩略图均可点击并在模态查看器中按图像原始像素尺寸显示；大于视口的图像使用滚动区域
查看。查看器支持关闭按钮、Esc 和点击背景关闭。

生成操作台显示实际 Pipeline 进度条、当前阶段、百分比和去噪 step。生成完成后保留 100%
状态，加载下一张图片时重置。

## 6. 依赖与启动参数

`requirements.txt` 使用 Diffusers GitHub 主分支，包含 `AutoPipelineForInpainting`、
`Flux2KleinInpaintPipeline`、`ZImageInpaintPipeline`、Transformers、Accelerate、Safetensors
和 SentencePiece。

相关启动参数：

```text
--inpaint-provider
--sd15-model
--sd15-lora
--sd15-vae
--sd15-device
--sd15-cache-dir
--sd15-cpu-offload
--flux-model
--flux-device
--flux-cache-dir
--flux-cpu-offload
--zimage-model
--zimage-device
--zimage-cache-dir
--zimage-cpu-offload
```

SD1.5 CUDA 使用 float16，FLUX 与 Z-Image CUDA 优先使用 bfloat16；显式选择 CPU 时使用
float32。Z-Image-Turbo 默认使用 8 steps、strength 1.0 和 guidance 0.0。

## 7. 验证

当前静态与算法验证命令：

```powershell
$env:PYTHONPATH = ".;.\inpaint4drag"
D:\anaconda3\envs\pytorch-cpu\python.exe -m unittest discover -s inpaint4drag\tests -v
D:\anaconda3\envs\pytorch-cpu\python.exe -m py_compile inpaint4drag\demo.py inpaint4drag\baseline_warp.py inpaint4drag\my_warp.py inpaint4drag\sam_provider.py inpaint4drag\sd15_inpaint_provider.py inpaint4drag\flux_inpaint_provider.py inpaint4drag\zimage_inpaint_provider.py
node --check inpaint4drag\web\app.js
git diff --check
```

测试集包含既有 Warp 回归、生成请求重算、三个 Provider 的 Warp/Mask 输入语义、最终 Mask
合并和 Pipeline step 进度状态测试。
模型权重加载与 GPU 推理在安装 Diffusers 主分支和对应 Provider 权重的目标环境执行。

## 8. Pipeline 尺寸链路

SD1.5 Provider 将输入宽高调整为最接近的 8 的倍数，并对 Mask 使用最近邻缩放。生成结果
恢复至 Demo 输入尺寸后，使用原始全分辨率 `inpaint_mask` 合成。

FLUX Provider 不设置 `padding_mask_crop`，以完整 Warp `image` 和原始 `mask_image` 坐标系
执行 inpaint。

Z-Image Provider 根据已加载 Pipeline 的 `2 * vae_scale_factor` 动态对齐输入宽高，并对
Mask 使用最近邻缩放。生成结果恢复至 Demo 输入尺寸后，使用原始全分辨率 `inpaint_mask`
合成。

Klein Pipeline 会把超过约 100 万像素的输入先等比例缩小到约 1MP。Provider 在 Pipeline
返回尺寸与 Demo 输入尺寸不一致时，将完整结果恢复到 Demo 输入尺寸。默认 1080 px 长边预览
通常低于该像素阈值。

重绘结构实验可使用相同 Mask、prompt 和 seed，对比 `strength=1.0` 与 `strength=0.8`，
观察完全重生成和保留原图结构之间的差异。
