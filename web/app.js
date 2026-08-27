const previewCanvas = document.querySelector("#previewCanvas");
const overlayCanvas = document.querySelector("#overlayCanvas");
const preview = previewCanvas.getContext("2d");
const overlay = overlayCanvas.getContext("2d");
const stage = document.querySelector("#stage");
const workspace = document.querySelector("main");
const footer = document.querySelector("footer");
const emptyState = document.querySelector("#emptyState");
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
const boundaryAnchorRow = document.querySelector("#boundaryAnchorRow");
const samButton = document.querySelector('[data-mode="sam"]');
const samHint = document.querySelector("#samHint");

let width = 0;
let height = 0;
let original = null;
let mask = null;
let maskBounds = null;
let baseImage = null;
let objectCanvas = null;
let maskOverlayCanvas = null;
let maskCanvas = null;
let sourceCanvas = null;
let mode = "paint";
let renderRequested = false;
let pythonReady = Promise.resolve();
let pendingWarp = null;
let warpInFlight = false;
let warpEpoch = 0;
let currentImageFile = null;

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
  warpEpoch++;
  mode = nextMode;
  document.querySelectorAll(".mode-button").forEach(button => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  instructions.textContent = original ? instructionText[mode] : "请先加载图像。";
  pendingSource = null;
  hoverPoint = null;
  schedulePreview();
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
  const style = getComputedStyle(workspace);
  const availableWidth = workspace.clientWidth -
    parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const availableHeight = workspace.clientHeight - footer.offsetHeight -
    parseFloat(style.paddingTop) - parseFloat(style.paddingBottom) - 16;
  const scale = Math.min(1, availableWidth / width, availableHeight / height);
  stage.style.width = `${Math.round(width * scale)}px`;
  stage.style.height = `${Math.round(height * scale)}px`;
}


async function loadImage(file) {
  currentImageFile = file;
  statusText.textContent = "正在加载图像并进行 Python / SAM 预处理…";
  const bitmap = await createImageBitmap(file);
  ({width, height} = scaledImageSize(bitmap.width, bitmap.height));

  previewCanvas.width = overlayCanvas.width = width;
  previewCanvas.height = overlayCanvas.height = height;
  stage.style.aspectRatio = `${width} / ${height}`;
  preview.drawImage(bitmap, 0, 0, width, height);
  original = preview.getImageData(0, 0, width, height);
  sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  sourceCanvas.getContext("2d").putImageData(original, 0, 0);
  mask = new Uint8Array(width * height);
  pointPairs = [];
  transform = {x: 0, y: 0, angle: 0, scale: 1};
  updateMaskAssets(false);
  emptyState.hidden = true;
  stage.hidden = false;
  layoutStage();
  setMode("paint");
  samButton.disabled = true;
  samHint.textContent = "正在检查 SAM 并提取图像特征…";
  pythonReady = syncImage().then(async metadata => {
    samButton.disabled = !metadata.sam_ready;
    samHint.textContent = metadata.sam_ready
      ? `SAM 预处理完成（${metadata.sam_preprocess_ms.toFixed(0)} ms），点击对象即可生成 Mask。`
      : "SAM 未配置；当前可使用画笔绘制 Mask。启动时传入 --sam-checkpoint 可启用点选对象。";
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


function updateMaskAssets(sync = true) {
  const basePixels = new Uint8ClampedArray(original.data);
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
        const checker = ((Math.floor(x / 10) + Math.floor(y / 10)) & 1) ? 202 : 235;
        basePixels[index] = checker;
        basePixels[index + 1] = checker;
        basePixels[index + 2] = checker;
        overlayPixels[index] = 255;
        overlayPixels[index + 1] = 62;
        overlayPixels[index + 2] = 92;
        overlayPixels[index + 3] = 92;
        maskPixels[index] = 255;
        maskPixels[index + 1] = 255;
        maskPixels[index + 2] = 255;
      } else {
        objectPixels[index + 3] = 0;
      }
    }
  }

  maskBounds = right >= left ? {left, right, top, bottom} : null;
  baseImage = new ImageData(basePixels, width, height);
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
  context.strokeStyle = "rgba(255, 62, 92, .36)";
  context.fillStyle = "rgba(255, 62, 92, .36)";
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
  if (maskBounds && mode === "deform" && pointPairs.length > 0) {
    requestPythonPreview();
    drawOverlay();
    return;
  }

  const started = performance.now();
  preview.setTransform(1, 0, 0, 1, 0, 0);
  preview.clearRect(0, 0, width, height);

  if (!maskBounds || ["paint", "erase", "sam"].includes(mode)) {
    preview.putImageData(original, 0, 0);
  } else if (mode === "transform") {
    preview.putImageData(baseImage, 0, 0);
    const pivot = originalPivot();
    preview.save();
    preview.imageSmoothingEnabled = true;
    preview.translate(pivot.x + transform.x, pivot.y + transform.y);
    preview.rotate(transform.angle);
    preview.scale(transform.scale, transform.scale);
    preview.translate(-pivot.x, -pivot.y);
    preview.drawImage(objectCanvas, 0, 0);
    preview.restore();
  } else preview.putImageData(original, 0, 0);

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
  if (request.epoch === warpEpoch && mode === "deform" && pointPairs.length > 0) {
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

  if (["paint", "erase", "sam"].includes(mode)) {
    overlay.drawImage(maskOverlayCanvas, 0, 0);
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
    overlay.drawImage(maskOverlayCanvas, 0, 0);
    drawPointPairs(scale);
  }
}


function drawTransformedMask() {
  const pivot = originalPivot();
  overlay.save();
  overlay.translate(pivot.x + transform.x, pivot.y + transform.y);
  overlay.rotate(transform.angle);
  overlay.scale(transform.scale, transform.scale);
  overlay.translate(-pivot.x, -pivot.y);
  overlay.drawImage(maskOverlayCanvas, 0, 0);
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
  overlay.arc(point.x, point.y, (selected ? 7 : 6) * scale, 0, Math.PI * 2);
  overlay.fillStyle = color;
  overlay.fill();
  overlay.lineWidth = 2 * scale;
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
  schedulePreview();
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


setMode("paint");
