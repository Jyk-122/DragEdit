# DragEdit FLUX.2 Klein 生成重绘记录 - Dev_2

更新时间：2026-08-28

本文记录 Warp 交互结果接入 FLUX.2 Klein 局部重绘后的数据流、界面和运行参数。
后续开发应先阅读根目录 `AGENTS.md`、`documents/dev_0.md` 和 `documents/dev_1.md`。

## 1. 当前完整链路

Demo 现在包含三个连续阶段：

1. 选择图像和对象 Mask。
2. 通过整体变换或 point pairs 得到 Warp 编辑结果。
3. 使用 `Flux2KleinInpaintPipeline` 对目标对象、空洞和融合带统一重绘。

默认模型为 `black-forest-labs/FLUX.2-klein-4B`。`demo.py` 在创建 HTTP 服务前加载
模型，使页面可访问时生成管线已经就绪。

## 2. 生成模型输入

`flux_inpaint_provider.py` 负责 Diffusers 推理，核心输入语义如下：

```text
image           = 原图
image_reference = 局部干净 Warp 结果
mask_image      = inpaint_mask OR target_mask
```

Mask 的白色区域由模型重新生成。把 `target_mask` 并入最终 Mask，能够让模型重绘完整
变形对象，而不只是填补 source hole 和 target contour band；`image_reference` 则向模型
提供目标形变、位移、缩放和旋转布局。

默认提示词要求遵循参考图的目标布局和对象身份，同时修复 Warp 导致的拉伸、模糊、
撕裂、空洞、接缝、重复内容和不合理结构，并保持外围图像自然衔接。

默认推理参数：

| 参数 | 默认值 |
|---|---:|
| `num_inference_steps` | 4 |
| `strength` | 1.0 |
| `guidance_scale` | 1.0 |
| `padding_mask_crop` | 64 |
| `seed` | 0 |

## 3. Image reference 构造

棋盘格只属于交互预览，不进入生成模型。

- `BaselineWarpSession` 和 `MyWarpSession` 新增 `warped_image`，保存原图与 target object
  合成后的干净图像；公开 `warp_preview()` 返回结构保持不变。
- 浏览器整体变换使用独立 `transformReferenceCanvas` 保存应用位移、旋转和缩放后的
  干净合成图，再复制一份用于叠加棋盘格预览。
- 生成前使用 OpenCV Telea 对 `source_mask & ~target_mask` 做参考图占位修补，并重新覆盖
  target object，避免 source object 残留干扰参考条件。
- 最终 `image_reference` 按 target mask 外接框和 64 px padding 裁成局部参考图。

## 4. HTTP 接口

### `POST /api/generate`

整体变换由浏览器提交：

- `image`
- `image_reference`
- `inpaint_mask`
- `target_mask`
- `source_mask`

非刚性变形提交 Warp 算法、point pairs、边界锚点和 kernel size。服务端先用对应 Session
重算当前 Warp，再直接读取 `image`、`warped_image`、`inpaint_mask`、`target_mask` 和
`source_mask`，保证生成输入对应按钮点击时的编辑状态。

接口返回生成 PNG Data URL、模型名、推理耗时，以及实际传给 Pipeline 的 `image`、
`image_reference`、`mask_image` 三张 PNG 调试图和各自尺寸。

## 5. 页面布局

所有操作控件集中在页面底部工具栏，上方并排显示两个等尺寸展示台：

- 左侧为 Mask 与拖拽编辑台。
- 右侧为原图/生成图对比台。

对比台以生成图为底层，原图覆盖左侧区域。竖线向右拖动会显示更多原图，向左拖动会
显示更多生成图。

生成完成后，对比台下方显示 `image`、`image_reference` 和 `mask_image` 三张缩略图。
这些图直接由 Provider 调用 Pipeline 时使用的 PIL 对象编码，便于核对局部参考图裁剪、
最终白色重绘区域和原图输入。

## 6. 依赖与启动参数

`requirements.txt` 使用 Diffusers GitHub 主分支，以包含
`Flux2KleinInpaintPipeline`，并加入 Transformers、Accelerate、Safetensors 和
SentencePiece。

相关启动参数：

```text
--flux-model
--flux-device
--flux-cache-dir
--flux-cpu-offload
```

Demo 默认使用 CUDA。CUDA 优先采用 bfloat16，不支持时使用 float16；显式选择 CPU 时
使用 float32。

## 7. 验证

当前静态与算法验证命令：

```powershell
D:\anaconda3\envs\pytorch-cpu\python.exe -m unittest discover -s tests -v
D:\anaconda3\envs\pytorch-cpu\python.exe -m py_compile demo.py baseline_warp.py my_warp.py sam_provider.py flux_inpaint_provider.py
node --check web/app.js
git diff --check
```

测试集共 12 项，包含既有 Warp 回归，以及生成请求重算、最终 Mask 合并和局部参考图裁剪测试。
模型权重加载与 GPU 推理在安装 Diffusers 主分支和 FLUX.2 Klein 权重的目标环境执行。
