"""MNIST 手写数字识别网页。

运行方式：python web_app.py
然后浏览器访问：http://127.0.0.1:5000
"""

import base64
import io
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from torchvision import transforms

from model import SimpleCNN


# Flask 会自动从 templates 读取 HTML，从 static 读取 CSS 和 JavaScript。
app = Flask(__name__)

# 模型只在网页启动时加载一次，而不是每次点击识别都重新加载。
# Path(__file__) 保证即使从其他目录启动，也能找到项目中的模型文件。
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model() -> SimpleCNN:
    """读取训练好的参数，并把模型切换到预测模式。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型 {MODEL_PATH}，请先运行 python train.py")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = SimpleCNN(dropout=checkpoint.get("dropout", 0.25)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # 关闭 Dropout，使同一张图每次预测结果一致。
    return model


# 启动服务时就准备好模型，后续所有请求共同使用它。
MODEL = load_trained_model()


def data_url_to_image(data_url: str) -> Image.Image:
    """把浏览器 Canvas 发送的 data URL 解码成 Pillow 灰度图片。"""
    if not data_url.startswith("data:image/png;base64,"):
        raise ValueError("图片格式不正确，请重新绘制")

    encoded_data = data_url.split(",", maxsplit=1)[1]
    try:
        image_bytes = base64.b64decode(encoded_data, validate=True)
        return Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as error:
        raise ValueError("无法读取手写图片，请清空后重试") from error


def prepare_canvas_image(image: Image.Image) -> torch.Tensor:
    """把画布笔迹处理成接近 MNIST 的 28×28 张量。

    网页画布本身使用黑底白字，与 MNIST 一致。这里先找出笔迹范围，
    再把数字等比例缩放并居中，避免数字写在画布角落时严重影响预测。
    """
    # 找出所有明显笔迹的包围盒。getbbox() 在全黑空画布上会返回 None。
    binary_image = image.point(lambda pixel: 255 if pixel > 20 else 0)
    bounding_box = binary_image.getbbox()
    if bounding_box is None:
        raise ValueError("画布还是空的，请先写一个数字")

    # 裁出笔迹，并保持宽高比例缩放到最大 20×20。
    # MNIST 数字周围会留出一些黑边，所以不直接拉伸到完整 28×28。
    digit = image.crop(bounding_box)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # 创建 28×28 黑底，把缩放后的数字粘到中央。
    centered_image = Image.new("L", (28, 28), color=0)
    left = (28 - digit.width) // 2
    top = (28 - digit.height) // 2
    centered_image.paste(digit, (left, top))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # 添加批量维度：[1, 28, 28] → [1, 1, 28, 28]。
    return transform(centered_image).unsqueeze(0)


@app.get("/")
def index() -> str:
    """返回手写识别页面。"""
    return render_template("index.html", device=DEVICE.type.upper())


@app.post("/api/predict")
def predict():
    """接收画布图片，返回预测数字、置信度和十个类别的概率。"""
    request_data = request.get_json(silent=True) or {}
    image_data = request_data.get("image", "")

    try:
        image = data_url_to_image(image_data)
        image_tensor = prepare_canvas_image(image).to(DEVICE)

        # 预测阶段不计算梯度，可以减少内存占用并提高速度。
        with torch.no_grad():
            logits = MODEL(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        predicted_digit = int(probabilities.argmax().item())
        return jsonify(
            {
                "digit": predicted_digit,
                "confidence": round(float(probabilities[predicted_digit]), 6),
                "probabilities": [round(float(value), 6) for value in probabilities],
            }
        )
    except ValueError as error:
        # 用户可修正的输入问题返回 400，并把明确提示展示到页面。
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    print(f"模型运行设备: {DEVICE}")
    print("请在浏览器打开: http://127.0.0.1:5000")
    # debug=False 避免开发服务器重复加载模型两次。
    app.run(host="127.0.0.1", port=5000, debug=False)
