// 取得页面上需要操作的元素。
const canvas = document.getElementById("drawingCanvas");
const context = canvas.getContext("2d");
const clearButton = document.getElementById("clearButton");
const predictButton = document.getElementById("predictButton");
const buttonLabel = predictButton.querySelector(".button-label");
const message = document.getElementById("message");
const predictedDigit = document.getElementById("predictedDigit");
const confidence = document.getElementById("confidence");
const predictionCard = document.getElementById("predictionCard");
const probabilityRows = [...document.querySelectorAll(".probability-row")];

let isDrawing = false;
let hasDrawing = false;

// 设置接近 MNIST 的画布：黑色背景、白色圆头粗笔迹。
function resetCanvas() {
  context.fillStyle = "#000000";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "#ffffff";
  context.lineWidth = 28;
  context.lineCap = "round";
  context.lineJoin = "round";
  hasDrawing = false;
}

// 浏览器显示尺寸可能与 Canvas 内部分辨率不同，因此需要换算坐标。
function getCanvasPoint(event) {
  const rectangle = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rectangle.left) * (canvas.width / rectangle.width),
    y: (event.clientY - rectangle.top) * (canvas.height / rectangle.height),
  };
}

function startDrawing(event) {
  event.preventDefault();
  isDrawing = true;
  hasDrawing = true;
  canvas.setPointerCapture(event.pointerId);

  const point = getCanvasPoint(event);
  context.beginPath();
  context.moveTo(point.x, point.y);
  // 单击也画出一个小圆点，否则没有移动时 Canvas 不会产生笔迹。
  context.lineTo(point.x + 0.01, point.y + 0.01);
  context.stroke();
}

function draw(event) {
  if (!isDrawing) return;
  event.preventDefault();
  const point = getCanvasPoint(event);
  context.lineTo(point.x, point.y);
  context.stroke();
}

function stopDrawing(event) {
  if (!isDrawing) return;
  isDrawing = false;
  context.closePath();
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

function resetResults() {
  predictedDigit.textContent = "?";
  predictedDigit.setAttribute("aria-label", "等待预测");
  confidence.textContent = "—";
  predictionCard.classList.remove("has-result");
  probabilityRows.forEach((row) => {
    row.classList.remove("winner");
    row.querySelector(".probability-fill").style.width = "0%";
    row.querySelector(".probability-value").textContent = "0.0%";
  });
}

function clearAll() {
  resetCanvas();
  resetResults();
  message.textContent = "尽量写在画布中央，笔画清晰一些。";
  message.classList.remove("error");
}

function showPrediction(result) {
  predictedDigit.textContent = result.digit;
  predictedDigit.setAttribute("aria-label", `预测数字 ${result.digit}`);
  confidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;

  // 移除后重新加入动画类，让每次新预测都有一次“盖章”效果。
  predictionCard.classList.remove("has-result");
  void predictionCard.offsetWidth;
  predictionCard.classList.add("has-result");

  probabilityRows.forEach((row, digit) => {
    const percentage = result.probabilities[digit] * 100;
    row.classList.toggle("winner", digit === result.digit);
    row.querySelector(".probability-fill").style.width = `${Math.max(percentage, 0.3)}%`;
    row.querySelector(".probability-value").textContent = `${percentage.toFixed(1)}%`;
  });
}

async function predictDrawing() {
  if (!hasDrawing) {
    message.textContent = "画布还是空的，请先写一个数字。";
    message.classList.add("error");
    return;
  }

  predictButton.disabled = true;
  buttonLabel.textContent = "正在识别…";
  message.textContent = "CNN 正在读取你的笔迹。";
  message.classList.remove("error");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") }),
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "识别失败，请重新尝试");
    }

    showPrediction(result);
    message.textContent = `识别完成：模型认为这是数字 ${result.digit}。`;
  } catch (error) {
    message.textContent = error.message;
    message.classList.add("error");
  } finally {
    predictButton.disabled = false;
    buttonLabel.textContent = "开始识别";
  }
}

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
clearButton.addEventListener("click", clearAll);
predictButton.addEventListener("click", predictDrawing);

// 页面第一次打开时初始化黑色画布。
clearAll();
