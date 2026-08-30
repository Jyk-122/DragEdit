# DragEdit 交互预览与空洞参数记录 - Dev_1

更新时间：2026-08-27

生成重绘阶段的当前实现见 `documents/dev_2.md`。

本文记录 `Dev_0` 之后确认的前端交互、Warp 预览合成和 inpaint 空洞参数设计。后续开发应先阅读根目录 `AGENTS.md`、`documents/dev_0.md`，再阅读本文。

## 1. 本阶段结果

本阶段统一了二维非刚性形变与二维整体变换的预览语义：

- 两种编辑方式使用相同的 Warp 图层透明度。
- 两种编辑方式使用相同的空洞扩张核尺寸。
- inpaint 预览区域都包含 source revealed hole 和 target contour band。
- 空洞区域使用 Photoshop 风格的灰白棋盘格。
- Mask 来源工具与编辑方式切换时保留当前预览图像。

## 2. 界面参数与显示

### 2.1 预览长边

预览长边使用普通数字输入框，不提供预设下拉列表：

- 默认值：1080 px。
- 可填写范围：320–4096 px。
- 输入值表示图像长边，图像始终保持原始宽高比。
- 修改后重新载入当前图像，并重置 Mask 与编辑状态。

### 2.2 Point pair 标注

- source/target 标注点普通显示半径为 4 px。
- 选中态显示半径为 5 px。
- 鼠标命中半径保持为 12 px，显示尺寸与交互热区相互独立。

### 2.3 空洞占位

空洞和目标轮廓融合带使用灰白棋盘格：

- 单格尺寸：10 个预览图像像素。
- 亮色值：RGB `(240, 240, 240)`。
- 灰色值：RGB `(200, 200, 200)`。
- Baseline、My Warp 和浏览器整体变换使用同一组样式参数。

### 2.4 Warp 图层透明度

- 范围：0%–100%。
- 默认值：100%。
- 0% 显示原图，100% 显示完整 Warp 预览。
- 非刚性形变通过 `/api/warp` 的 `preview_opacity` 在 Python 中合成。
- 整体变换先在独立 Canvas 中生成完整结果，再与原图混合一次。
- Mask overlay 使用独立的 36% 显示透明度，不随 Warp 图层透明度变化。

## 3. 预览状态管理

Mask 工具状态和编辑预览状态相互独立：

- 切换到画笔、橡皮或 SAM 时，`previewCanvas` 保留当前 Warp 结果。
- 返回非刚性形变时继续显示已有 Python 预览，新的 point 编辑再触发请求。
- 返回整体变换时使用最新 Mask assets 重绘当前平移、旋转和缩放状态。
- 异步 `/api/warp` 响应使用 `warpEpoch` 过滤失效结果；切换到 Mask 工具不会丢弃当前有效的非刚性响应。
- 调整整体变换透明度时，即使当前处于 Mask 工具，也会更新底层整体变换预览。

## 4. 空洞扩张参数

### 4.1 UI 与 API

界面提供“空洞扩张大小”滑块：

- 范围：1–31 px。
- 步长：2，只产生奇数核。
- 默认值：5 px。

非刚性请求通过以下字段传递：

```json
{
  "inpaint_kernel_size": 5
}
```

`POST /api/warp` 将该值传入 `BaselineWarpSession.preview()` 或 `MyWarpSession.preview()`，并在响应头返回：

```text
X-Inpaint-Kernel-Size: 5
```

Baseline 继续直接调用参考实现 `bi_warp(mask, control_points, kernel_size)`；My Warp 将同一值传入 `build_inpaint_mask()`。默认值保持为 5。

### 4.2 Inpaint 区域语义

两种编辑方式均采用以下预览区域语义：

```text
revealed_hole = source_mask AND NOT target_mask
inpaint_mask = dilate(revealed_hole, kernel_size) OR target_contour_band
```

- `revealed_hole` 表示 object 离开后暴露出的 source 区域。
- `target_contour_band` 表示移动或形变后 object 轮廓内外两侧的融合带。
- 棋盘格最后写入完整变换图，因此会同时覆盖 source hole 和移动后局部图像的目标轮廓边缘。

### 4.3 整体变换的浏览器实现

二维整体变换继续由浏览器实时计算：

1. 使用与 object 相同的平移、旋转、缩放矩阵生成 target mask。
2. 计算 `source_mask AND NOT target_mask`。
3. 对 revealed hole 使用当前方形核膨胀。
4. 从 target mask 提取一像素边界，再使用同一方形核生成 target contour band。
5. 合并两部分 inpaint preview mask。
6. 将棋盘格写入完整变换图，再按照 Warp 图层透明度与原图混合。

前端膨胀采用可复用的 TypedArray 缓冲区和横向、纵向滑动窗口。单次膨胀复杂度为 `O(width × height)`，不直接执行 `kernel_size²` 次邻域扫描。

未产生平移、旋转或缩放时，整体变换保持原图显示，不提前绘制 inpaint 区域。

## 5. 主要代码位置

| 文件 | 当前职责变化 |
|---|---|
| `web/index.html` | 数字长边输入、Warp 透明度和空洞扩张控件 |
| `web/app.js` | 预览状态保持、整体变换离屏合成、target mask 与 inpaint preview mask 计算 |
| `demo.py` | 接收并转发 `inpaint_kernel_size`，返回对应响应头 |
| `baseline_warp.py` | Session 接收可调 kernel size，使用统一棋盘格占位 |
| `my_warp.py` | Session 接收可调 kernel size，使用统一棋盘格占位 |
| `tests/test_baseline_warp.py` | Baseline kernel size 对 inpaint 面积的回归测试 |
| `tests/test_warp.py` | My Warp kernel size 对 inpaint 面积的回归测试 |

`reference/` 中的 Inpaint4Drag 实现保持不变。

## 6. 验证状态

已执行：

```powershell
D:\anaconda3\envs\pytorch-cpu\python.exe -m unittest discover -s tests -v
D:\anaconda3\envs\pytorch-cpu\python.exe -m py_compile demo.py baseline_warp.py my_warp.py sam_provider.py
node --check web/app.js
git diff --check
```

当前共有 9 个算法测试，全部通过。新增覆盖包括：

- Baseline 使用更大的 `kernel_size` 时产生更大的 inpaint 区域。
- My Warp 使用更大的 `kernel_size` 时产生更大的 inpaint 区域。

## 7. 当前接口边界

- 非刚性 inpaint mask 由 Python Session 保存，可通过 `GET /api/inpaint-mask` 获取最近一次结果。
- 整体变换的 inpaint preview mask 当前在浏览器中实时生成，用于交互预览。
- 整体变换结果和浏览器 inpaint preview mask 尚未提交给最终生成模型。
- 最终 diffusion inpainting 接入后，应继续沿用 source revealed hole 与 target contour band 的统一语义。
- 前端文件更新后使用 `Ctrl+F5` 刷新本地 Demo。
