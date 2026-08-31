const previewCanvas = document.querySelector("#previewCanvas");
const overlayCanvas = document.querySelector("#overlayCanvas");
const preview = previewCanvas.getContext("2d");
const overlay = overlayCanvas.getContext("2d");
const stage = document.querySelector("#stage");
const workspace = document.querySelector("main");
const footer = document.querySelector("footer");
const editorViewport = document.querySelector("#editorViewport");
const comparisonViewport = document.querySelector("#comparisonViewport");
const stagesArea = document.querySelector("#stagesArea");
const emptyState = document.querySelector("#emptyState");
const comparisonEmpty = document.querySelector("#comparisonEmpty");
const comparisonStage = document.querySelector("#comparisonStage");
const generatedCanvas = document.querySelector("#generatedCanvas");
const originalCanvas = document.querySelector("#originalCanvas");
const pipelineInputs = document.querySelector("#pipelineInputs");
const statusText = document.querySelector("#status");
const performanceText = document.querySelector("#performance");
const instructions = document.querySelector("#instructions");
const imageInput = document.querySelector("#imageInput");
const previewLongEdge = document.querySelector("#previewLongEdge");
const brushSize = document.querySelector("#brushSize");
const keepBoundary = document.querySelector("#boundaryAnchors");
const warpAlgorithm = document.querySelector("#warpAlgorithm");
const warpOpacity = document.querySelector("#warpOpacity");
const warpOpacityValue = document.querySelector("#warpOpacityValue");
const inpaintKernelSize = document.querySelector("#inpaintKernelSize");
const inpaintKernelSizeValue = document.querySelector("#inpaintKernelSizeValue");
const boundaryAnchorRow = document.querySelector("#boundaryAnchorRow");
const samButton = document.querySelector('[data-mode="sam"]');
const samHint = document.querySelector("#samHint");
const generateButton = document.querySelector("#generateImage");
const generationPrompt = document.querySelector("#generationPrompt");
const generationSteps = document.querySelector("#generationSteps");
const generationStrength = document.querySelector("#generationStrength");
const generationGuidance = document.querySelector("#generationGuidance");
const generationSeed = document.querySelector("#generationSeed");
const generationHint = document.querySelector("#generationHint");
const generationProgress = document.querySelector("#generationProgress");
const generationProgressBar = document.querySelector("#generationProgressBar");
const generationProgressStage = document.querySelector("#generationProgressStage");
const generationProgressValue = document.querySelector("#generationProgressValue");
const pipelineInputDialog = document.querySelector("#pipelineInputDialog");
const pipelineInputDialogTitle = document.querySelector("#pipelineInputDialogTitle");
const pipelineInputDialogSize = document.querySelector("#pipelineInputDialogSize");
const pipelineInputFullImage = document.querySelector("#pipelineInputFullImage");
const debugInputViews = {
  image: {
    image: document.querySelector("#debugImageInput"),
    size: document.querySelector("#debugImageInputSize"),
  },
  mask_image: {
    image: document.querySelector("#debugMaskImage"),
    size: document.querySelector("#debugMaskImageSize"),
  },
};
const MASK_OPACITY = 0.36;
const MASK_MODES = new Set(["paint", "erase", "sam"]);

let width = 0;
let height = 0;
let original = null;
let mask = null;
let maskBounds = null;
let objectCanvas = null;
let transformCanvas = null;
let transformReferenceCanvas = null;
let transformMaskCanvas = null;
let transformMaskBuffers = null;
let maskOverlayCanvas = null;
let maskCanvas = null;
let sourceCanvas = null;
let mode = "paint";
let editMode = null;
let renderRequested = false;
let pythonReady = Promise.resolve();
let pendingWarp = null;
let warpInFlight = false;
let warpEpoch = 0;
let currentImageFile = null;
let comparisonPosition = 50;
let generationProgressEpoch = 0;
let generationDefaultsApplied = false;

let painting = false;
let lastPaintPoint = null;
let hoverPoint = null;
let transformAction = null;
let endpointDrag = null;
let pendingSource = null;
let selectedPair = -1;
let pointPairs = [];
let transform = {x: 0, y: 0, angle: 0, scale: 1};


const instructionText = {
  paint: "按住鼠标涂抹 object 区域。",
  erase: "按住鼠标擦除 mask。",
  sam: "点击要编辑的对象，使用 SAM 生成该对象的 Mask。",
  transform: "拖动 object 平移；拖动边框手柄缩放；拖动顶部圆点旋转。",
  deform: "点击两次创建 point pair；拖动任一端点实时修改；右键或 Delete 删除。",
};


function setMode(nextMode) {
  const editModeChanged = !MASK_MODES.has(nextMode) && editMode !== nextMode;
  if (editModeChanged) {
    editMode = nextMode;
    warpEpoch++;
  }
  mode = nextMode;
  document.querySelectorAll(".mode-button").forEach(button => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  instructions.textContent = original ? instructionText[mode] : "请先加载图像。";
  pendingSource = null;
  hoverPoint = null;
  if (editModeChanged || nextMode === "transform") schedulePreview();
  else drawOverlay();
}


function canvasPoint(event) {
  const rect = overlayCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * width / rect.width,
    y: (event.clientY - rect.top) * height / rect.height,
  };
}


function displayScale() {
  return width / overlayCanvas.getBoundingClientRect().width;
}


function insideMask(point) {
  const x = Math.round(point.x);
  const y = Math.round(point.y);
  return x >= 0 && y >= 0 && x < width && y < height && mask[y * width + x] > 0;
}


function scaledImageSize(imageWidth, imageHeight) {
  const scale = Number(previewLongEdge.value) / Math.max(imageWidth, imageHeight);
  return {
    width: Math.round(imageWidth * scale),
    height: Math.round(imageHeight * scale),
  };
}


function layoutStage() {
  const availableSpace = viewport => {
    const style = getComputedStyle(viewport);
    return {
      width: viewport.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight),
      height: viewport.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom),
    };
  };
  const editorSpace = availableSpace(editorViewport);
  const comparisonSpace = availableSpace(comparisonViewport);
  const availableWidth = Math.min(editorSpace.width, comparisonSpace.width);
  const availableHeight = Math.min(editorSpace.height, comparisonSpace.height);
  const scale = Math.min(1, availableWidth / width, availableHeight / height);
  const displayWidth = Math.round(width * scale);
  const displayHeight = Math.round(height * scale);
  for (const imageStage of [stage, comparisonStage]) {
    imageStage.style.width = `${displayWidth}px`;
    imageStage.style.height = `${displayHeight}px`;
  }
}


async function loadImage(file) {
  currentImageFile = file;
  statusText.textContent = "正在加载图像并进行 Python / SAM 预处理…";
  const bitmap = await createImageBitmap(file);
  ({width, height} = scaledImageSize(bitmap.width, bitmap.height));
  stagesArea.classList.toggle("portrait-layout", height < width);

  previewCanvas.width = overlayCanvas.width = width;
  previewCanvas.height = overlayCanvas.height = height;
  generatedCanvas.width = originalCanvas.width = width;
  generatedCanvas.height = originalCanvas.height = height;
  stage.style.aspectRatio = `${width} / ${height}`;
  comparisonStage.style.aspectRatio = `${width} / ${height}`;
  preview.drawImage(bitmap, 0, 0, width, height);
  original = preview.getImageData(0, 0, width, height);
  sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  sourceCanvas.getContext("2d").putImageData(original, 0, 0);
  transformCanvas = document.createElement("canvas");
  transformCanvas.width = width;
  transformCanvas.height = height;
  transformReferenceCanvas = document.createElement("canvas");
  transformReferenceCanvas.width = width;
  transformReferenceCanvas.height = height;
  transformMaskCanvas = document.createElement("canvas");
  transformMaskCanvas.width = width;
  transformMaskCanvas.height = height;
  transformMaskBuffers = {};
  for (const name of ["target", "revealed", "boundary", "hole", "band", "inpaint", "scratch"]) {
    transformMaskBuffers[name] = new Uint8Array(width * height);
  }
  mask = new Uint8Array(width * height);
  pointPairs = [];
  editMode = null;
  transform = {x: 0, y: 0, angle: 0, scale: 1};
  updateMaskAssets(false);
  emptyState.hidden = true;
  stage.hidden = false;
  comparisonEmpty.hidden = false;
  comparisonStage.hidden = true;
  pipelineInputs.hidden = true;
  generationProgress.hidden = true;
  if (pipelineInputDialog.open) pipelineInputDialog.close();
  layoutStage();
  setMode("paint");
  samButton.disabled = true;
  samHint.textContent = "正在检查 SAM 并提取图像特征…";
  pythonReady = syncImage().then(async metadata => {
    samButton.disabled = !metadata.sam_ready;
    samHint.textContent = metadata.sam_ready
      ? `SAM 预处理完成（${metadata.sam_preprocess_ms.toFixed(0)} ms），点击对象即可生成 Mask。`
      : "SAM 未配置；当前可使用画笔绘制 Mask。启动时传入 --sam-checkpoint 可启用点选对象。";
    generationHint.textContent = `${metadata.generation_model} 已加载，可使用当前编辑结果生成重绘。`;
    if (!generationDefaultsApplied) {
      const defaults = metadata.generation_defaults;
      generationPrompt.value = defaults.prompt;
      generationSteps.value = defaults.num_inference_steps;
      generationStrength.value = defaults.strength;
      generationGuidance.value = defaults.guidance_scale;
      generationDefaultsApplied = true;
    }
    await syncMask();
    return metadata;
  });
  await pythonReady;
  const samState = samButton.disabled ? "画笔 Mask 可用" : "SAM 与画笔 Mask 可用";
  statusText.textContent = `预览分辨率 ${width} × ${height}，${samState}`;
}


async function syncImage() {
  const response = await fetch("/api/image", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({image: sourceCanvas.toDataURL("image/png")}),
  });
  return response.json();
}


async function syncMask() {
  await fetch("/api/mask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({mask: maskCanvas.toDataURL("image/png")}),
  });
}


async function requestSamMask(point) {
  statusText.textContent = "SAM 正在生成对象 Mask…";
  const response = await fetch("/api/sam-mask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      x: Math.round(point.x),
      y: Math.round(point.y),
    }),
  });
  const result = await response.json();
  const image = new Image();
  image.src = result.mask;
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0, width, height);
  const data = context.getImageData(0, 0, width, height).data;
  for (let i = 0; i < mask.length; i++) {
    mask[i] = data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2] > 384 ? 1 : 0;
  }
  updateMaskAssets();
  statusText.textContent = "SAM 对象 Mask 已生成";
}


function paintCircle(point, value) {
  const radius = Number(brushSize.value) * width / overlayCanvas.getBoundingClientRect().width / 2;
  const left = Math.max(0, Math.floor(point.x - radius));
  const right = Math.min(width - 1, Math.ceil(point.x + radius));
  const top = Math.max(0, Math.floor(point.y - radius));
  const bottom = Math.min(height - 1, Math.ceil(point.y + radius));
  const radiusSquared = radius * radius;
  for (let y = top; y <= bottom; y++) {
    for (let x = left; x <= right; x++) {
      const dx = x - point.x;
      const dy = y - point.y;
      if (dx * dx + dy * dy <= radiusSquared) mask[y * width + x] = value;
    }
  }
}


function paintLine(from, to, value) {
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const radius = Number(brushSize.value) * displayScale() / 2;
  const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius * 0.45)));
  for (let step = 1; step <= steps; step++) {
    const amount = step / steps;
    paintCircle({
      x: from.x + (to.x - from.x) * amount,
      y: from.y + (to.y - from.y) * amount,
    }, value);
  }
}


function dilateMask(source, kernelSize, result = null, horizontal = null) {
  const radius = Math.floor(kernelSize / 2);
  result ??= new Uint8Array(source.length);
  if (radius === 0) {
    result.set(source);
    return result;
  }
  horizontal ??= new Uint8Array(source.length);

  for (let y = 0; y < height; y++) {
    const row = y * width;
    let count = 0;
    for (let x = 0; x <= Math.min(radius, width - 1); x++) count += source[row + x];
    for (let x = 0; x < width; x++) {
      horizontal[row + x] = count > 0 ? 1 : 0;
      const remove = x - radius;
      const add = x + radius + 1;
      if (remove >= 0) count -= source[row + remove];
      if (add < width) count += source[row + add];
    }
  }

  for (let x = 0; x < width; x++) {
    let count = 0;
    for (let y = 0; y <= Math.min(radius, height - 1); y++) count += horizontal[y * width + x];
    for (let y = 0; y < height; y++) {
      result[y * width + x] = count > 0 ? 1 : 0;
      const remove = y - radius;
      const add = y + radius + 1;
      if (remove >= 0) count -= horizontal[remove * width + x];
      if (add < height) count += horizontal[add * width + x];
    }
  }
  return result;
}


function drawTransformedAsset(context, asset) {
  const pivot = originalPivot();
  context.save();
  context.imageSmoothingEnabled = true;
  context.translate(pivot.x + transform.x, pivot.y + transform.y);
  context.rotate(transform.angle);
  context.scale(transform.scale, transform.scale);
  context.translate(-pivot.x, -pivot.y);
  context.drawImage(asset, 0, 0);
  context.restore();
}


function buildTransformInpaintMask() {
  const targetContext = transformMaskCanvas.getContext("2d");
  targetContext.setTransform(1, 0, 0, 1, 0, 0);
  targetContext.clearRect(0, 0, width, height);
  drawTransformedAsset(targetContext, maskOverlayCanvas);
  const targetPixels = targetContext.getImageData(0, 0, width, height).data;
  const buffers = transformMaskBuffers;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = y * width + x;
      const target = targetPixels[pixel * 4 + 3] >= 128 ? 1 : 0;
      buffers.target[pixel] = target;
      buffers.revealed[pixel] = mask[pixel] && !target ? 1 : 0;
    }
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = y * width + x;
      buffers.boundary[pixel] = buffers.target[pixel] && (
        x === 0 || y === 0 || x === width - 1 || y === height - 1 ||
        !buffers.target[pixel - 1] || !buffers.target[pixel + 1] ||
        !buffers.target[pixel - width] || !buffers.target[pixel + width]
      ) ? 1 : 0;
    }
  }

  const kernelSize = Number(inpaintKernelSize.value);
  dilateMask(buffers.revealed, kernelSize, buffers.hole, buffers.scratch);
  dilateMask(buffers.boundary, kernelSize, buffers.band, buffers.scratch);
  for (let pixel = 0; pixel < buffers.inpaint.length; pixel++) {
    buffers.inpaint[pixel] = buffers.hole[pixel] || buffers.band[pixel] ? 1 : 0;
  }
  return buffers.inpaint;
}


function updateMaskAssets(sync = true) {
  const objectPixels = new Uint8ClampedArray(original.data);
  const overlayPixels = new Uint8ClampedArray(width * height * 4);
  const maskPixels = new Uint8ClampedArray(width * height * 4);
  let left = width, right = -1, top = height, bottom = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = y * width + x;
      const index = pixel * 4;
      maskPixels[index + 3] = 255;
      if (mask[pixel]) {
        left = Math.min(left, x);
        right = Math.max(right, x);
        top = Math.min(top, y);
        bottom = Math.max(bottom, y);
        overlayPixels[index] = 255;
        overlayPixels[index + 1] = 62;
        overlayPixels[index + 2] = 92;
        overlayPixels[index + 3] = 255;
        maskPixels[index] = 255;
        maskPixels[index + 1] = 255;
        maskPixels[index + 2] = 255;
      } else {
        objectPixels[index + 3] = 0;
      }
    }
  }

  maskBounds = right >= left ? {left, right, top, bottom} : null;
  objectCanvas = document.createElement("canvas");
  objectCanvas.width = width;
  objectCanvas.height = height;
  objectCanvas.getContext("2d").putImageData(new ImageData(objectPixels, width, height), 0, 0);
  maskOverlayCanvas = document.createElement("canvas");
  maskOverlayCanvas.width = width;
  maskOverlayCanvas.height = height;
  maskOverlayCanvas.getContext("2d").putImageData(new ImageData(overlayPixels, width, height), 0, 0);
  maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  maskCanvas.getContext("2d").putImageData(new ImageData(maskPixels, width, height), 0, 0);
  warpEpoch++;
  if (sync) pythonReady = pythonReady.then(syncMask);
  schedulePreview();
}


function drawMaskStroke(from, to, value) {
  const context = maskOverlayCanvas.getContext("2d");
  const diameter = Number(brushSize.value) * displayScale();
  context.save();
  context.globalCompositeOperation = value ? "source-over" : "destination-out";
  context.strokeStyle = "rgb(255, 62, 92)";
  context.fillStyle = "rgb(255, 62, 92)";
  context.lineWidth = diameter;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(from.x, from.y);
  context.lineTo(to.x, to.y);
  context.stroke();
  context.beginPath();
  context.arc(to.x, to.y, diameter / 2, 0, Math.PI * 2);
  context.fill();
  context.restore();
  drawOverlay();
}


function schedulePreview() {
  if (!original || renderRequested) return;
  renderRequested = true;
  requestAnimationFrame(renderPreview);
}


function renderPreview() {
  renderRequested = false;
  if (MASK_MODES.has(mode)) {
    drawOverlay();
    return;
  }
  if (maskBounds && mode === "deform" && pointPairs.length > 0) {
    requestPythonPreview();
    drawOverlay();
    return;
  }
  if (mode === "transform") {
    renderTransformPreview();
    return;
  }

  const started = performance.now();
  preview.setTransform(1, 0, 0, 1, 0, 0);
  preview.clearRect(0, 0, width, height);

  preview.putImageData(original, 0, 0);
  drawOverlay();
  performanceText.textContent = `预览 ${(performance.now() - started).toFixed(1)} ms`;
}


function renderTransformPreview() {
  if (!original) return;
  const started = performance.now();
  const reference = transformReferenceCanvas.getContext("2d");
  reference.setTransform(1, 0, 0, 1, 0, 0);
  reference.clearRect(0, 0, width, height);
  reference.putImageData(original, 0, 0);
  if (maskBounds) drawTransformedAsset(reference, objectCanvas);

  const transformed = transformCanvas.getContext("2d");
  transformed.setTransform(1, 0, 0, 1, 0, 0);
  transformed.clearRect(0, 0, width, height);
  const transformedState = Math.abs(transform.x) > 1e-4 || Math.abs(transform.y) > 1e-4 ||
    Math.abs(transform.angle) > 1e-4 || Math.abs(transform.scale - 1) > 1e-4;
  transformed.drawImage(transformReferenceCanvas, 0, 0);

  if (maskBounds && transformedState) {
    const inpaintMask = buildTransformInpaintMask();
    const transformedImage = transformed.getImageData(0, 0, width, height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixel = y * width + x;
        if (!inpaintMask[pixel]) continue;
        const index = pixel * 4;
        const grid = ((Math.floor(x / 10) + Math.floor(y / 10)) & 1) ? 200 : 240;
        transformedImage.data[index] = grid;
        transformedImage.data[index + 1] = grid;
        transformedImage.data[index + 2] = grid;
      }
    }
    transformed.putImageData(transformedImage, 0, 0);
  }

  preview.setTransform(1, 0, 0, 1, 0, 0);
  preview.putImageData(original, 0, 0);
  preview.save();
  preview.globalAlpha = Number(warpOpacity.value) / 100;
  preview.drawImage(transformCanvas, 0, 0);
  preview.restore();
  drawOverlay();
  performanceText.textContent = `预览 ${(performance.now() - started).toFixed(1)} ms`;
}


function requestPythonPreview() {
  pendingWarp = {
    epoch: warpEpoch,
    point_pairs: pointPairs.map(pair => ({
      source: {...pair.source},
      target: {...pair.target},
    })),
    keep_boundary: keepBoundary.checked,
    algorithm: warpAlgorithm.value,
    preview_opacity: Number(warpOpacity.value) / 100,
    inpaint_kernel_size: Number(inpaintKernelSize.value),
  };
  runWarpQueue();
}


async function runWarpQueue() {
  if (warpInFlight || !pendingWarp) return;
  warpInFlight = true;
  const request = pendingWarp;
  pendingWarp = null;
  await pythonReady;
  const started = performance.now();
  const response = await fetch("/api/warp", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(request),
  });
  const bitmap = await createImageBitmap(await response.blob());
  if (request.epoch === warpEpoch && editMode === "deform" && pointPairs.length > 0) {
    preview.setTransform(1, 0, 0, 1, 0, 0);
    preview.drawImage(bitmap, 0, 0, width, height);
    drawOverlay();
    const algorithm = response.headers.get("X-Warp-Algorithm") === "baseline"
      ? "Baseline"
      : "My Warp";
    performanceText.textContent =
      `${algorithm} ${response.headers.get("X-Warp-Ms")} ms · 往返 ${(performance.now() - started).toFixed(1)} ms`;
  }
  bitmap.close();
  warpInFlight = false;
  runWarpQueue();
}


function maskDataUrl(values) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let pixel = 0; pixel < values.length; pixel++) {
    const value = values[pixel] ? 255 : 0;
    const index = pixel * 4;
    pixels[index] = value;
    pixels[index + 1] = value;
    pixels[index + 2] = value;
    pixels[index + 3] = 255;
  }
  canvas.getContext("2d").putImageData(new ImageData(pixels, width, height), 0, 0);
  return canvas.toDataURL("image/png");
}


function hasTransformEdit() {
  return Math.abs(transform.x) > 1e-4 || Math.abs(transform.y) > 1e-4 ||
    Math.abs(transform.angle) > 1e-4 || Math.abs(transform.scale - 1) > 1e-4;
}


function generationParameters() {
  return {
    prompt: generationPrompt.value.trim(),
    num_inference_steps: Number(generationSteps.value),
    strength: Number(generationStrength.value),
    guidance_scale: Number(generationGuidance.value),
    seed: Number(generationSeed.value),
  };
}


function updateGenerationProgress(percent, stage, step=0, steps=0) {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  generationProgress.hidden = false;
  generationProgressBar.value = value;
  generationProgressValue.textContent = `${Math.round(value)}%`;
  generationProgressStage.textContent = step > 0 && steps > 0
    ? `${stage} · ${step}/${steps}`
    : stage;
}


async function pollGenerationProgress(epoch) {
  while (epoch === generationProgressEpoch) {
    try {
      const response = await fetch("/api/generation-progress", {cache: "no-store"});
      const progress = await response.json();
      if (epoch !== generationProgressEpoch) return;
      if (progress.running) {
        updateGenerationProgress(
          progress.percent,
          progress.stage,
          progress.step,
          progress.steps,
        );
      }
    } catch (_) {
      // The generation request still reports the final server error.
    }
    await new Promise(resolve => setTimeout(resolve, 400));
  }
}


async function generateImage() {
  if (!original || !maskBounds) {
    statusText.textContent = "请先加载图像并选取 Mask。";
    return;
  }
  if (editMode === "transform" && !hasTransformEdit()) {
    statusText.textContent = "请先平移、旋转或缩放对象。";
    return;
  }
  if (editMode === "deform" && pointPairs.length === 0) {
    statusText.textContent = "请先建立至少一个 point pair。";
    return;
  }
  if (!editMode) {
    statusText.textContent = "请先选择一种拖拽编辑方式并完成编辑。";
    return;
  }

  await pythonReady;
  generateButton.disabled = true;
  generateButton.textContent = "正在生成…";
  statusText.textContent = "生成模型正在重绘编辑区域…";
  updateGenerationProgress(1, "正在提交生成任务");
  const progressEpoch = ++generationProgressEpoch;
  void pollGenerationProgress(progressEpoch);

  const request = generationParameters();
  if (editMode === "transform") {
    renderTransformPreview();
    const inpaint = buildTransformInpaintMask();
    request.image = sourceCanvas.toDataURL("image/png");
    request.warped_image = transformReferenceCanvas.toDataURL("image/png");
    request.inpaint_mask = maskDataUrl(inpaint);
    request.target_mask = maskDataUrl(transformMaskBuffers.target);
    request.source_mask = maskCanvas.toDataURL("image/png");
  } else {
    request.algorithm = warpAlgorithm.value;
    request.point_pairs = pointPairs.map(pair => ({
      source: {...pair.source},
      target: {...pair.target},
    }));
    request.keep_boundary = keepBoundary.checked;
    request.inpaint_kernel_size = Number(inpaintKernelSize.value);
  }

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(request),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "生成失败");
    await showComparison(result.image, result.pipeline_inputs);
    const seconds = (result.inference_ms / 1000).toFixed(1);
    updateGenerationProgress(100, "生成完成");
    statusText.textContent = `生成完成 · ${result.model}`;
    performanceText.textContent = `生成 ${seconds} s`;
  } catch (error) {
    updateGenerationProgress(generationProgressBar.value, "生成失败");
    statusText.textContent = `生成失败：${error.message}`;
  } finally {
    generationProgressEpoch++;
    generateButton.disabled = false;
    generateButton.textContent = "生成图片";
  }
}


function openPipelineInput(name) {
  const view = debugInputViews[name];
  if (!view?.image.src) return;
  pipelineInputDialogTitle.textContent = name;
  pipelineInputDialogSize.textContent = view.size.textContent;
  pipelineInputFullImage.src = view.image.src;
  pipelineInputFullImage.alt = `${name} 原始分辨率输入`;
  pipelineInputDialog.showModal();
}


async function showComparison(dataUrl, inputs) {
  const image = new Image();
  image.src = dataUrl;
  await image.decode();
  generatedCanvas.getContext("2d").drawImage(image, 0, 0, width, height);
  originalCanvas.getContext("2d").putImageData(original, 0, 0);
  comparisonEmpty.hidden = true;
  comparisonStage.hidden = false;
  for (const [name, view] of Object.entries(debugInputViews)) {
    const input = inputs[name];
    const figure = view.image.closest("figure");
    figure.hidden = !input;
    if (!input) {
      view.image.removeAttribute("src");
      view.size.textContent = "";
      continue;
    }
    view.image.src = input.image;
    view.size.textContent = `${input.width}×${input.height}`;
  }
  pipelineInputs.hidden = false;
  setComparisonPosition(50);
  layoutStage();
}


function setComparisonPosition(percent) {
  comparisonPosition = Math.max(0, Math.min(100, percent));
  comparisonStage.style.setProperty("--comparison-position", `${comparisonPosition}%`);
}


function updateComparison(event) {
  const rect = comparisonStage.getBoundingClientRect();
  setComparisonPosition((event.clientX - rect.left) * 100 / rect.width);
}


comparisonStage.addEventListener("pointerdown", event => {
  comparisonStage.setPointerCapture(event.pointerId);
  updateComparison(event);
});
comparisonStage.addEventListener("pointermove", event => {
  if (comparisonStage.hasPointerCapture(event.pointerId)) updateComparison(event);
});


function originalPivot() {
  return {
    x: (maskBounds.left + maskBounds.right) / 2,
    y: (maskBounds.top + maskBounds.bottom) / 2,
  };
}


function transformPoint(point) {
  const pivot = originalPivot();
  const dx = (point.x - pivot.x) * transform.scale;
  const dy = (point.y - pivot.y) * transform.scale;
  const cosine = Math.cos(transform.angle);
  const sine = Math.sin(transform.angle);
  return {
    x: pivot.x + transform.x + dx * cosine - dy * sine,
    y: pivot.y + transform.y + dx * sine + dy * cosine,
  };
}


function inverseTransformPoint(point) {
  const pivot = originalPivot();
  const dx = point.x - pivot.x - transform.x;
  const dy = point.y - pivot.y - transform.y;
  const cosine = Math.cos(-transform.angle);
  const sine = Math.sin(-transform.angle);
  return {
    x: pivot.x + (dx * cosine - dy * sine) / transform.scale,
    y: pivot.y + (dx * sine + dy * cosine) / transform.scale,
  };
}


function gizmoGeometry() {
  const corners = [
    {x: maskBounds.left, y: maskBounds.top},
    {x: maskBounds.right, y: maskBounds.top},
    {x: maskBounds.right, y: maskBounds.bottom},
    {x: maskBounds.left, y: maskBounds.bottom},
  ].map(transformPoint);
  const edgeHandles = [
    midpoint(corners[0], corners[1]), midpoint(corners[1], corners[2]),
    midpoint(corners[2], corners[3]), midpoint(corners[3], corners[0]),
  ];
  const top = edgeHandles[0];
  const offset = 30 * displayScale();
  const rotation = {
    x: top.x + Math.sin(transform.angle) * offset,
    y: top.y - Math.cos(transform.angle) * offset,
  };
  return {corners, scaleHandles: corners.concat(edgeHandles), top, rotation};
}


function midpoint(a, b) {
  return {x: (a.x + b.x) / 2, y: (a.y + b.y) / 2};
}


function drawOverlay() {
  overlay.clearRect(0, 0, width, height);
  if (!original) return;
  const scale = displayScale();

  if (MASK_MODES.has(mode)) {
    drawMaskGuide();
    if ((mode === "paint" || mode === "erase") && hoverPoint) {
      overlay.beginPath();
      overlay.arc(hoverPoint.x, hoverPoint.y, Number(brushSize.value) * scale / 2, 0, Math.PI * 2);
      overlay.strokeStyle = "rgba(255,255,255,.9)";
      overlay.lineWidth = scale;
      overlay.stroke();
    }
    return;
  }

  if (mode === "transform" && maskBounds) {
    drawTransformedMask();
    drawGizmo(scale);
  }
  if (mode === "deform") {
    drawMaskGuide();
    drawPointPairs(scale);
  }
}


function drawMaskGuide() {
  overlay.save();
  overlay.globalAlpha = MASK_OPACITY;
  overlay.drawImage(maskOverlayCanvas, 0, 0);
  overlay.restore();
}


function drawTransformedMask() {
  overlay.save();
  overlay.globalAlpha = MASK_OPACITY;
  drawTransformedAsset(overlay, maskOverlayCanvas);
  overlay.restore();
}


function drawGizmo(scale) {
  const gizmo = gizmoGeometry();
  overlay.strokeStyle = "#f4f7ff";
  overlay.lineWidth = 1.5 * scale;
  overlay.setLineDash([5 * scale, 4 * scale]);
  overlay.beginPath();
  overlay.moveTo(gizmo.corners[0].x, gizmo.corners[0].y);
  for (let i = 1; i < gizmo.corners.length; i++) overlay.lineTo(gizmo.corners[i].x, gizmo.corners[i].y);
  overlay.closePath();
  overlay.stroke();
  overlay.setLineDash([]);
  overlay.beginPath();
  overlay.moveTo(gizmo.top.x, gizmo.top.y);
  overlay.lineTo(gizmo.rotation.x, gizmo.rotation.y);
  overlay.stroke();

  for (const handle of gizmo.scaleHandles) {
    overlay.fillStyle = "#ffffff";
    overlay.fillRect(handle.x - 4 * scale, handle.y - 4 * scale, 8 * scale, 8 * scale);
    overlay.strokeStyle = "#456df2";
    overlay.strokeRect(handle.x - 4 * scale, handle.y - 4 * scale, 8 * scale, 8 * scale);
  }
  overlay.beginPath();
  overlay.arc(gizmo.rotation.x, gizmo.rotation.y, 6 * scale, 0, Math.PI * 2);
  overlay.fillStyle = "#7b9cff";
  overlay.fill();
  overlay.strokeStyle = "white";
  overlay.stroke();
}


function drawPointPairs(scale) {
  pointPairs.forEach((pair, index) => {
    drawArrow(pair.source, pair.target, false, index === selectedPair, scale);
    drawPoint(pair.source, "#4e8cff", index === selectedPair, scale);
    drawPoint(pair.target, "#ff5e7a", index === selectedPair, scale);
  });
  if (pendingSource && hoverPoint) {
    drawArrow(pendingSource, hoverPoint, true, true, scale);
    drawPoint(pendingSource, "#4e8cff", true, scale);
  }
}


function drawPoint(point, color, selected, scale) {
  overlay.beginPath();
  overlay.arc(point.x, point.y, (selected ? 5 : 4) * scale, 0, Math.PI * 2);
  overlay.fillStyle = color;
  overlay.fill();
  overlay.lineWidth = 1.5 * scale;
  overlay.strokeStyle = "white";
  overlay.stroke();
}


function drawArrow(from, to, dashed, selected, scale) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  const head = 11 * scale;
  overlay.save();
  overlay.strokeStyle = selected ? "#ffffff" : "rgba(255,255,255,.78)";
  overlay.lineWidth = (selected ? 2.5 : 2) * scale;
  overlay.setLineDash(dashed ? [7 * scale, 6 * scale] : []);
  overlay.beginPath();
  overlay.moveTo(from.x, from.y);
  overlay.lineTo(to.x, to.y);
  overlay.stroke();
  overlay.setLineDash([]);
  overlay.beginPath();
  overlay.moveTo(to.x, to.y);
  overlay.lineTo(to.x - head * Math.cos(angle - Math.PI / 6), to.y - head * Math.sin(angle - Math.PI / 6));
  overlay.moveTo(to.x, to.y);
  overlay.lineTo(to.x - head * Math.cos(angle + Math.PI / 6), to.y - head * Math.sin(angle + Math.PI / 6));
  overlay.stroke();
  overlay.restore();
}


function hitPointPair(point) {
  const radiusSquared = Math.pow(12 * displayScale(), 2);
  let best = null;
  let bestDistance = radiusSquared;
  pointPairs.forEach((pair, index) => {
    for (const endpoint of ["source", "target"]) {
      const dx = point.x - pair[endpoint].x;
      const dy = point.y - pair[endpoint].y;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        best = {index, endpoint};
        bestDistance = distance;
      }
    }
  });
  return best;
}


function near(point, target, radius) {
  return Math.hypot(point.x - target.x, point.y - target.y) <= radius;
}


function startTransform(point) {
  const gizmo = gizmoGeometry();
  const radius = 11 * displayScale();
  if (near(point, gizmo.rotation, radius)) {
    const pivot = transformPoint(originalPivot());
    transformAction = {
      type: "rotate",
      pointerAngle: Math.atan2(point.y - pivot.y, point.x - pivot.x),
      angle: transform.angle,
    };
    return;
  }
  if (gizmo.scaleHandles.some(handle => near(point, handle, radius))) {
    const pivot = transformPoint(originalPivot());
    transformAction = {
      type: "scale",
      distance: Math.hypot(point.x - pivot.x, point.y - pivot.y),
      scale: transform.scale,
    };
    return;
  }
  if (insideMask(inverseTransformPoint(point))) {
    transformAction = {type: "move", point, x: transform.x, y: transform.y};
  }
}


function moveTransform(point) {
  if (transformAction.type === "move") {
    transform.x = transformAction.x + point.x - transformAction.point.x;
    transform.y = transformAction.y + point.y - transformAction.point.y;
  } else {
    const pivot = transformPoint(originalPivot());
    if (transformAction.type === "rotate") {
      const angle = Math.atan2(point.y - pivot.y, point.x - pivot.x);
      transform.angle = transformAction.angle + angle - transformAction.pointerAngle;
    } else {
      const distance = Math.hypot(point.x - pivot.x, point.y - pivot.y);
      transform.scale = Math.max(0.05, transformAction.scale * distance / transformAction.distance);
    }
  }
  schedulePreview();
}


overlayCanvas.addEventListener("pointerdown", event => {
  if (!original || event.button !== 0) return;
  const point = canvasPoint(event);
  overlayCanvas.setPointerCapture(event.pointerId);

  if (mode === "paint" || mode === "erase") {
    painting = true;
    lastPaintPoint = point;
    const value = mode === "paint" ? 1 : 0;
    paintCircle(point, value);
    drawMaskStroke(point, point, value);
  } else if (mode === "sam") {
    requestSamMask(point);
  } else if (mode === "transform" && maskBounds) {
    startTransform(point);
  } else if (mode === "deform" && maskBounds) {
    const hit = hitPointPair(point);
    if (hit) {
      selectedPair = hit.index;
      endpointDrag = hit;
      drawOverlay();
    } else if (pendingSource) {
      pointPairs.push({source: pendingSource, target: point});
      pendingSource = null;
      selectedPair = pointPairs.length - 1;
      schedulePreview();
    } else if (insideMask(point)) {
      pendingSource = point;
      hoverPoint = point;
      drawOverlay();
    }
  }
});


overlayCanvas.addEventListener("pointermove", event => {
  if (!original) return;
  const point = canvasPoint(event);
  hoverPoint = point;
  if (painting) {
    const value = mode === "paint" ? 1 : 0;
    paintLine(lastPaintPoint, point, value);
    drawMaskStroke(lastPaintPoint, point, value);
    lastPaintPoint = point;
  } else if (transformAction) {
    moveTransform(point);
  } else if (endpointDrag) {
    pointPairs[endpointDrag.index][endpointDrag.endpoint] = point;
    selectedPair = endpointDrag.index;
    schedulePreview();
  } else {
    drawOverlay();
  }
});


function finishPointer() {
  if (painting) updateMaskAssets();
  painting = false;
  transformAction = null;
  endpointDrag = null;
}


overlayCanvas.addEventListener("pointerup", finishPointer);
overlayCanvas.addEventListener("pointercancel", finishPointer);
overlayCanvas.addEventListener("pointerleave", () => {
  if (!painting && !transformAction && !endpointDrag) {
    hoverPoint = null;
    drawOverlay();
  }
});


overlayCanvas.addEventListener("contextmenu", event => {
  event.preventDefault();
  if (mode !== "deform" || !original) return;
  const hit = hitPointPair(canvasPoint(event));
  if (!hit) return;
  pointPairs.splice(hit.index, 1);
  selectedPair = -1;
  schedulePreview();
});


function deleteSelectedPair() {
  if (selectedPair < 0) return;
  pointPairs.splice(selectedPair, 1);
  selectedPair = -1;
  schedulePreview();
}


document.addEventListener("keydown", event => {
  if (event.key === "Delete" || event.key === "Backspace") deleteSelectedPair();
  if (event.key === "Escape") {
    pendingSource = null;
    selectedPair = -1;
    drawOverlay();
  }
});


imageInput.addEventListener("change", () => loadImage(imageInput.files[0]));
previewLongEdge.addEventListener("change", () => {
  if (currentImageFile) loadImage(currentImageFile);
});
window.addEventListener("resize", () => {
  if (original) layoutStage();
});
brushSize.addEventListener("input", drawOverlay);
keepBoundary.addEventListener("change", schedulePreview);
warpOpacity.addEventListener("input", () => {
  warpOpacityValue.textContent = `${warpOpacity.value}%`;
  if (editMode === "transform") renderTransformPreview();
  else schedulePreview();
});
inpaintKernelSize.addEventListener("input", () => {
  inpaintKernelSizeValue.textContent = `${inpaintKernelSize.value} px`;
  if (editMode === "transform") renderTransformPreview();
  else if (mode === "deform") schedulePreview();
});
warpAlgorithm.addEventListener("change", () => {
  boundaryAnchorRow.hidden = warpAlgorithm.value === "baseline";
  warpEpoch++;
  schedulePreview();
});
document.querySelectorAll(".mode-button").forEach(button => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
document.querySelector("#clearMask").addEventListener("click", () => {
  if (!mask) return;
  mask.fill(0);
  pointPairs = [];
  updateMaskAssets();
});
document.querySelector("#deletePair").addEventListener("click", deleteSelectedPair);
document.querySelector("#resetEdit").addEventListener("click", () => {
  pointPairs = [];
  pendingSource = null;
  selectedPair = -1;
  transform = {x: 0, y: 0, angle: 0, scale: 1};
  schedulePreview();
});
generateButton.addEventListener("click", generateImage);
document.querySelectorAll("[data-pipeline-input]").forEach(button => {
  button.addEventListener("click", () => openPipelineInput(button.dataset.pipelineInput));
});
document.querySelector("#closePipelineInputDialog").addEventListener("click", () => {
  pipelineInputDialog.close();
});
pipelineInputDialog.addEventListener("click", event => {
  if (event.target === pipelineInputDialog) pipelineInputDialog.close();
});


warpOpacityValue.textContent = `${warpOpacity.value}%`;
inpaintKernelSizeValue.textContent = `${inpaintKernelSize.value} px`;
setMode("paint");
