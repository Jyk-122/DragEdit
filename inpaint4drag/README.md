# DragEdit

Drag-based Image Editing 的交互、Warp 与生成重绘实验 Demo。

## 安装与启动

```powershell
pip install -r requirements.txt
python demo.py
```

服务会先加载默认生成模型 `black-forest-labs/FLUX.2-klein-4B`，随后在
`http://127.0.0.1:7860` 打开页面。首次启动时 Hugging Face 会下载模型权重。

常用启动参数：

```powershell
python demo.py `
  --flux-model black-forest-labs/FLUX.2-klein-4B `
  --flux-device cuda `
  --flux-cache-dir D:\models\huggingface
```

显存需要由 CPU 分担时可以增加 `--flux-cpu-offload`。服务部署在远程机器时可以使用：

```powershell
python demo.py --host 0.0.0.0 --no-browser --flux-device cuda
```

## 工作流

页面上方包含两个展示台：

- 左侧用于画笔或 SAM Mask 选取，以及整体变换、非刚性 point pairs 编辑。
- 右侧在生成完成后显示原图与生成图；拖动竖线可查看两侧差异。

页面下方工具栏依次完成：

1. 加载图像，并通过画笔、橡皮或 SAM 选择对象 Mask。
2. 使用二维整体变换，或使用 point pairs 完成二维非刚性形变。
3. 调整提示词和生成参数，点击“生成图片”运行 FLUX.2 Klein 局部重绘。

最终重绘使用 Diffusers `Flux2KleinInpaintPipeline`。其中：

- `image` 是原图。
- `image_reference` 是不含棋盘格的局部 warp 结果。
- `mask_image` 是 `inpaint_mask | target_mask`，白色区域由模型重新生成。
- 默认参数是 4 steps、strength 1.0、guidance 1.0、padding 64。

`baseline_warp.py` 继续直接调用 Inpaint4Drag 的原始 `bi_warp`；`my_warp.py`
保留独立实验实现。两条非刚性路径都会在 Session 中保存干净 warp 合成图，供生成阶段使用。

## SAM 点选对象

安装 Meta Segment Anything，并在启动时传入 checkpoint：

```powershell
pip install git+https://github.com/facebookresearch/segment-anything.git
python demo.py `
  --sam-checkpoint path\to\sam_vit_b.pth `
  --sam-model-type vit_b `
  --sam-device cuda
```

加载图片后，`sam_provider.py` 会调用一次 `set_image` 提取特征。此后点击对象即可
运行点提示解码，并将最高分候选设为当前 Mask。
