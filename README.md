# DragEdit

Drag-based Image Editing 的交互与算法实验。

## 本地交互 Demo

```powershell
pip install -r requirements.txt
python demo.py
```

浏览器会打开 `http://127.0.0.1:7860`。Demo 支持：

- 通过画笔绘制 Mask，或使用 SAM 点击选择对象。
- 按长边设置预览分辨率，可直接输入或选择 640、1080、2160。
- 对 mask object 进行平移、旋转和等比例缩放。
- 创建、拖动和删除 point pairs，实时预览非刚性形变。
- 保持原图与原始 Mask 固定，以透明图层显示变形区域和空缺区域。

二维整体变换由浏览器 Canvas 实时绘制；非刚性变形在 Python 中计算。Demo
默认使用 `baseline_warp.py`，它直接调用参考项目的原始 `bi_warp`。界面也可以
显式切换到 `my_warp.py`，其中每项实验改动均以 `OPTIMIZATION` 注释标出。

## SAM 点选对象

安装 Meta Segment Anything，并在启动时传入 checkpoint：

```powershell
pip install git+https://github.com/facebookresearch/segment-anything.git
python demo.py --sam-checkpoint path\to\sam_vit_b.pth --sam-model-type vit_b --sam-device cuda
```

加载图片后，`sam_provider.py` 会调用一次 `set_image` 完成图像特征提取。此后每次
在图像中点击对象只运行点提示解码，并将最高分的分割结果设为当前 Mask。未传入
checkpoint 时，画笔功能保持可用，SAM 点选按钮会保持未启用状态。
