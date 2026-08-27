# DragEdit 项目设计与开发记录 - Dev_0

更新时间：2026-08-27

本文面向后续参与本项目的模型与开发者，记录产品目标、已经确认的交互设计、当前代码架构、算法基线、实验结论、已修复问题和后续研究方向。开始新的开发任务前，应先阅读本文和项目根目录的 `AGENTS.md`。

## 1. 项目目标

这是一个 Drag-based Image Editing 算法实验项目，最终目标是形成适合端侧设备的低延迟产品 Demo。

当前选择 Inpaint4Drag 作为主要参考，原因是它把编辑过程拆成了两个阶段：

1. 使用二维 warp 立即生成交互预览。
2. 用户结束操作后，再用 diffusion inpainting 修复空缺、边缘和局部失真。

这种设计比每次鼠标移动都运行 diffusion 或特征空间优化更适合实时交互。项目现阶段的首要任务是验证 warp preview 的可控性、连续性和视觉合理性，而不是先追求最终生成质量。

参考项目位于 `reference/Inpaint4Drag/`。`reference/` 中的开源项目只用于阅读和对照，不要修改。

## 2. 已确认的产品交互分类

### 2.1 二维整体变换

面向 object 的平移、旋转和等比例缩放，使用类似 PPT 编辑形状的独立交互：

- 平移：点击 object 后拖动。
- 旋转：拖动 object 上方的旋转手柄。
- 缩放：拖动边框角点或边中手柄，当前实现为等比例缩放。

整体变换的意图是明确的 rigid motion，不应再由 point pairs 猜测用户到底想平移还是形变。

### 2.2 二维非刚性形变

使用任意数量 point pairs 表达局部 deformation：

- 第一次点击确定 source。
- 鼠标移动时显示虚线预览箭头。
- 第二次点击确定 target。
- 已建立的 source 和 target 都可以继续拖动。
- 右键或 Delete/Backspace 删除点对。

产品设计上，point pairs 模式表示非刚性编辑；整体移动有独立的 rigid transform 模式。严格 Baseline 仍保留 Inpaint4Drag 的原始数学行为，其中单 point pair 会导致整个连通区域平移。这一行为需要保留用于对照实验，不能在 Baseline 内改变。

### 2.3 三维刚性变化

这是二维整体变换的后续升级方向。可通过深度图、单视图三维重建或 SAM 3D 获取 object 的几何表示，在三维中进行旋转和平移，再投影为二维图像、可见性变化和二维 flow。

该方向尤其适合“转脸”等包含遮挡关系变化的编辑。纯二维插值同时存在两个根本限制：

- 位移场不理解图像内容和语义结构。
- 三维旋转产生的透视、遮挡和新显露表面无法由普通二维 IDW 准确描述。

三维刚性变化暂未实现。

### 2.4 三维非刚性形变

不作为产品功能方向。让普通用户在三维空间设置大量点对接近专业建模操作，交互成本过高。

### 2.5 更自然的轮廓形变输入

“瘦脸”这类操作如果要求固定眼睛、鼻子并设置多个脸颊点，交互过于复杂。已经提出但尚未实现的方向是：

- 用户沿目标局部轮廓画一段线。
- 再画箭头表示收缩或扩张方向。
- 算法将轮廓采样为一组有结构的约束，形成连续形变场。

后续可把“线 + 箭头”转换为沿曲线分布的 source points，并根据箭头构造 target points；还需要研究端点衰减、曲线法向约束、局部影响范围和与未编辑边界的平滑连接。

## 3. 开发与实验原则

### 3.1 Baseline 必须严格可追溯

`baseline_warp.py` 是严格实验基线：

- 直接调用 `reference/Inpaint4Drag/utils/drag.py::bi_warp`。
- 不改变控制点、插值方法、采样规则或 inpaint mask 规则。
- Demo 额外构造 `target_mask`，只用于显示合成和统计，不改变 `bi_warp`。
- Demo 默认选择 Baseline Warp。

任何算法改进都写入 `my_warp.py` 或新的独立实验文件，不能混入 Baseline。一次实验尽量只改变一个因素，保证结果可解释、可回退、可做 A/B 对照。

### 3.2 Python 与浏览器的职责

- Python：非刚性 warp、位移场、target mask、inpaint mask、SAM 推理和 Ghost 预览合成。
- JavaScript：鼠标/触控事件、点与手柄显示、Canvas 展示、请求调度。
- 当前二维整体变换使用浏览器 Canvas 实时绘制，以获得连续交互；非刚性算法全部由 Python 执行。

用户不会维护 JavaScript，因此交互代码应保持直接；算法实验接口必须集中在简洁的 Python 函数中。

### 3.3 项目代码风格

- 这是算法实验项目，不是生产服务。
- 代码优先简洁、易懂、便于修改。
- 不增加与实验无关的防御性框架和安全层。
- 不修改 `reference/`。
- 工作区可能已有用户改动；不要重置或覆盖不相关内容。

## 4. 当前代码架构

### 4.1 主要文件

| 文件 | 职责 |
|---|---|
| `demo.py` | 本地 HTTP 服务、API、Warp Session 管理、Ghost 预览合成 |
| `baseline_warp.py` | 严格调用原始 `bi_warp` 的基线封装 |
| `my_warp.py` | 独立的实验 warp，实现处用 `OPTIMIZATION` 标注 |
| `sam_provider.py` | 可选的 Meta Segment Anything 点提示分割 |
| `web/index.html` | Demo 控件和页面结构 |
| `web/app.js` | Canvas 交互、Mask 绘制、point pairs、请求队列、页面布局 |
| `web/style.css` | 页面视觉与画布布局 |
| `tests/test_baseline_warp.py` | Baseline 等价性和点击坐标回归测试 |
| `tests/test_warp.py` | My Warp 位移场和预览测试 |
| `README.md` | 启动方法与 SAM 使用说明 |

没有采用 Gradio。原因是拖动端点、虚线箭头、实时手柄和连续 Canvas 刷新更适合轻量网页前端；Gradio 的事件模型容易让每次操作表现为一次组件刷新。

### 4.2 HTTP API

#### `POST /api/image`

输入当前预览分辨率的 PNG Data URL。它会：

- 为 Baseline 和 My Warp Session 设置图像。
- 在配置 SAM 时调用一次 `SamPredictor.set_image()` 提取图像特征。
- 返回宽高、`sam_ready` 和 SAM 预处理耗时。

#### `POST /api/mask`

输入浏览器绘制或 SAM 生成的二值 Mask。当前 `refine_mask()` 是原样返回的接口占位。

#### `POST /api/warp`

主要输入字段：

- `algorithm`：`baseline` 或 `my_warp`。
- `point_pairs`：source/target 点对。
- `keep_boundary`：只对 My Warp 有意义。
- `preview_opacity`：Ghost Warp 图层透明度。

返回 JPEG 预览，并通过响应头返回：

- `X-Warp-Ms`
- `X-Inpaint-Pixels`
- `X-Target-Mask-Pixels`
- `X-Warp-Algorithm`

#### `POST /api/sam-mask`

输入一个图像坐标 `(x, y)`，使用一个正点提示运行 SAM decoder，返回最高分候选 Mask。

#### `GET /api/inpaint-mask`

返回最近一次非刚性 Warp 的 inpaint mask PNG，用于检查和后续生成接口接入。

当前服务使用全局 Session，定位是单用户本地实验 Demo。

## 5. Baseline Warp 的准确含义

### 5.1 原始流程

`bi_warp(region_mask, control_points, kernel_size=5)` 的主要过程是：

1. 查找 Mask 的外轮廓和连通区域。
2. 根据落入各区域的控制点计算方向。
3. 对轮廓和区域像素插值位移。
4. 构造 target contour 和 target region。
5. 通过反向插值找到每个 target pixel 对应的 source pixel。
6. 返回 source pixels、target pixels 和最终 inpaint mask。

Baseline wrapper 使用返回的 source/target 索引把原图像素复制到目标位置。

### 5.2 Inpaint Mask 与网格

`bi_warp` 计算的生成区域由两部分组成：

```text
revealed_hole = source_mask AND NOT target_mask
final_inpaint_mask = dilate(revealed_hole) OR target_contour_band
```

- `revealed_hole` 是物体移动后暴露出的原位置。
- `target_contour_band` 是目标物体轮廓周围主动留给生成模型重绘的融合带。
- `kernel_size` 控制空缺膨胀和目标轮廓宽度，Baseline 默认是 5。

黑白网格不是 `bi_warp` 生成的纹理。它是原始 UI 和当前 Baseline wrapper 在 `inpaint_mask` 区域绘制的占位提示，表示这些像素还需要 diffusion inpainting。原始 Inpaint4Drag 会把同一个 Mask 作为 `mask_image` 传给 Stable Diffusion Inpainting。

后续可以把 `kernel_size` 暴露为实验参数，研究融合带宽度对预览与最终生成质量的影响；Baseline 的默认值和默认路径仍应保持不变。

### 5.3 浏览器浮点点击失效问题

曾经出现“只有起点落在非常特殊的位置才有 Warp”的问题。原因是：

- Gradio `SelectData` 提供整数像素坐标。
- 浏览器经过 CSS 缩放后得到浮点坐标。
- 原始 `find_control_points` 使用接近精确匹配的 `< 1e-6` 条件，在整数像素区域中找不到大多数浮点 source。

当前修复位于 `baseline_warp.py`：在调用原始 `bi_warp` 前使用 `np.rint` 恢复 Gradio 的整数输入语义。`bi_warp` 本身没有修改。已有回归测试覆盖该问题。

## 6. My Warp 当前实验内容

`my_warp.py` 与 Baseline 独立，当前包含六项明确标注的实验变化：

1. 使用 Numba 构造全图稠密位移场。
2. 可选地加入零位移 Mask 边界锚点。
3. 使用定点迭代构造 backward map。
4. 通过反向 Warp source mask 获得 target mask。
5. 使用浮点映射和双线性图像采样。
6. 缓存网格图案与 source-hole preview base。

当前实验结论：关闭“自动固定远端边界”后，Baseline 和 My Warp 的表现接近；自动加入远端零位移锚点会明显限制形变传播，容易产生不自然的拉扯，不适合作为默认优化。

当前 UI 中该选项只在选择 My Warp 时显示，HTML 里的复选框仍处于默认勾选状态，这是保留的对照实验状态，不代表产品默认方案已经认可边界锚点。后续优化应继续逐项拆分，尤其不要把边界锚点与采样、插值、反向映射等变化合并评价。

## 7. 当前前端交互状态

### 7.1 图像分辨率

- 默认预览长边为 1080。
- 输入框提供 640、1080、2160 建议值，也可以手动输入其他数值。
- 选定值表示长边，图像严格保持原始宽高比。
- 示例：1920×1080 设置长边 1080 后得到 1080×608；1080×1920 得到 608×1080。
- 画布像素尺寸和页面 stage 显示尺寸分别计算，但使用同一个宽高比。
- 修改长边会重新载入当前图片并重置 Mask 与编辑状态。

空页面提示只在尚未加载图片时显示，加载后隐藏。

### 7.2 Mask 来源

产品只保留两种 Mask 来源：

1. 用户自己用画笔绘制，可切换橡皮并清空。
2. 图片完成 SAM 预处理后，用户点击对象选择 SAM Mask。

不提供 Mask 文件导入或标签颜色选择流程。

Mask 在所有模式下持续可见：

- 画笔绘制过程中和绘制完成后使用同一种半透明红色。
- 当前颜色为 RGB `(255, 62, 92)`，alpha 为 36%（Canvas 像素约 92/255）。
- 不额外绘制高不透明度边缘。
- 整体变换模式中，Mask 随 object 一起变换。
- 非刚性模式中，原始 Mask 固定在原位置，便于比较 source 与 warped result。

为了支持较高分辨率，画笔移动时直接增量绘制 overlay，而不是每个 pointer move 都重建整幅 RGBA Mask；pointer up 后再统一重建 Mask assets 并同步给 Python。

### 7.3 Ghost 非刚性预览

用户建立第一个完整 point pair 后：

- 原图保持不动。
- 原始 Mask 保持在原位置。
- Warp 后的 target region 与网格空缺区域通过半透明 Ghost 图层显示。
- Ghost 透明度范围为 20%–100%，默认 72%。

当前 Ghost 合成由 Python 的 `compose_ghost_preview()` 完成：只在 `target_mask OR inpaint_mask` 区域混合原图与 warp preview，随后编码为 JPEG 返回。透明度变化会重新请求一次 Python Warp。若后续专门优化透明度滑动性能，可以让 Python 返回带 alpha 的 Warp layer，并在浏览器只调整显示 alpha；Warp、位移场和 Mask 计算仍然保留在 Python。

### 7.4 二维整体变换的当前状态

整体变换由浏览器 Canvas 连续绘制：

- object 使用当前 Mask 从原图中提取。
- source 区域显示网格占位。
- object 通过 translate/rotate/uniform scale 绘制到目标位置。
- Mask overlay 使用相同矩阵变换。

当前 Demo 尚未为整体变换生成独立的 Python inpaint mask，也没有把整体变换结果接入最终生成模型。

## 8. SAM 接入

SAM 接口已实现于 `sam_provider.py`，使用 Meta `segment-anything`：

```powershell
pip install git+https://github.com/facebookresearch/segment-anything.git
python demo.py --sam-checkpoint path\to\sam_vit_b.pth --sam-model-type vit_b --sam-device cuda
```

行为如下：

- checkpoint 在服务启动时加载。
- 图片上传后运行一次 image encoder，即 `predictor.set_image(image)`。
- 预处理完成后前端启用“SAM 点选对象”。
- 用户每次点击只发送坐标，运行一个正点提示的 mask decoder。
- `multimask_output=True`，选择 score 最高的结果作为当前 Mask。
- 未配置 checkpoint 时画笔功能可直接使用，界面会显示 SAM 的启动参数提示。

当前 SAM 交互只支持单个正点，没有正负点组合、候选 Mask 切换和 Mask refine。这些可以后续扩展，但不应重新引入 Mask 文件导入流程。

## 9. 运行环境与验证

### 9.1 Python 环境

项目当前验证环境：

```text
D:\anaconda3\envs\pytorch-cpu\python.exe
```

如果 `python demo.py` 报 `ModuleNotFoundError: No module named 'torch'`，说明当前命令使用的解释器没有安装 PyTorch。先确认：

```powershell
conda activate pytorch-cpu
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

也可以直接使用上述环境的绝对 Python 路径。

### 9.2 启动

```powershell
python demo.py
```

默认地址为 `http://127.0.0.1:7860`。服务器测试时可以使用：

```powershell
python demo.py --host 0.0.0.0 --no-browser
```

前端代码更新后建议使用 `Ctrl+F5` 强制刷新浏览器缓存。

### 9.3 已执行验证

以下检查均已通过：

```powershell
D:\anaconda3\envs\pytorch-cpu\python.exe -m unittest discover -s tests -v
D:\anaconda3\envs\pytorch-cpu\python.exe -m py_compile demo.py baseline_warp.py my_warp.py sam_provider.py
node --check web/app.js
git diff --check
```

当前共有 7 个算法测试：

- Baseline 单点对区域平移。
- 控制点只影响所属连通区域。
- 浏览器浮点坐标恢复 Gradio 整数语义。
- Baseline wrapper 与原始 `bi_warp` 等价。
- My Warp 的 IDW 控制点匹配。
- My Warp 边界锚点行为。
- My Warp preview、mask 和 displacement field 输出。

实际 HTTP 联调也已验证 `/api/image`、`/api/mask` 和 `/api/warp`。使用参考 512×512 样例和浮点 point pair 时，Baseline 能正常返回 target mask 与 inpaint mask。

## 10. 当前功能边界

以下内容是后续工作的明确上下文：

- Demo 当前重点是实时 Warp preview，没有接入最终 diffusion refine 流程。
- `refine_mask()` 是保留接口，当前不修改用户 Mask。
- SAM 需要单独安装包并提供 checkpoint。
- 2160 长边可用于性能实验，但 Baseline 和大图编码往返可能无法达到每帧实时。
- Ghost 当前是 Python 合成后的 JPEG，不是浏览器中的独立 RGBA 图层。
- 二维整体变换和非刚性 Warp 目前走不同的预览路径。
- 三维刚性编辑、遮挡处理和新显露表面生成尚未实现。
- 人脸转动不应仅依靠增加更多二维固定点解决，应优先研究深度或三维表示。

## 11. 推荐的后续实验顺序

1. 固定 Baseline，建立一组典型图片、Mask 和 point pairs 的可视化回归样例。
2. 暴露 `inpaint kernel_size`，分别评价真实空缺和目标轮廓融合带。
3. 将 My Warp 的六项变化拆成独立开关或独立实现，每次只比较一项。
4. 把自动边界锚点从默认形变假设转为显式实验变量，研究局部影响范围的更自然表达。
5. 研究“轮廓线 + 箭头”到结构化 point constraints 的转换，用瘦脸、衣物轮廓和物体弯曲作为案例。
6. 为 motion 与 deformation 建立不同算法路径，避免由同一组 point pairs 猜测意图。
7. 评估 Ghost RGBA 图层和 Warp 结果缓存，减少透明度调整与重复请求的开销。
8. 在有 GPU 的服务器接入 SAM，测试单点误选、多个相邻 object 和细小结构。
9. 在 Warp preview 稳定后，再接入 diffusion inpainting，保持预览阶段与最终生成阶段解耦。
10. 对转脸等任务单独验证深度/三维重建到二维 flow、可见性 Mask 和 inpainting Mask 的完整链路。

## 12. 新会话开始时的检查清单

1. 阅读根目录 `AGENTS.md` 和本文。
2. 运行 `git status --short`，保留用户已有改动。
3. 不修改 `reference/`。
4. 确认当前任务属于 Baseline 验证、My Warp 实验、前端交互、SAM 还是最终生成。
5. 算法变化只进入实验实现，并明确标注相对 Baseline 的唯一差异。
6. 保持图像宽高比、统一的 36% Mask 透明度和原位置 Ghost Mask 语义。
7. 完成后运行对应测试、Python 编译检查和 JavaScript 语法检查。
