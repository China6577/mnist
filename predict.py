"""使用训练好的模型识别一张自己的数字图片。"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms

from model import SimpleCNN


def parse_args() -> argparse.Namespace:
    """读取模型路径和待识别图片路径。"""
    parser = argparse.ArgumentParser(description="识别一张手写数字图片")
    parser.add_argument("image", type=Path, help="待识别图片，例如 my_digit.png")
    parser.add_argument("--model", type=Path, default=Path("outputs/best_model.pth"), help="模型文件")
    return parser.parse_args()


def prepare_image(image_path: Path) -> torch.Tensor:
    """把普通图片处理成与 MNIST 相同的黑底白字 28×28 灰度图。"""
    image = Image.open(image_path).convert("L")

    # MNIST 是黑色背景、白色数字。如果自己的图片背景偏白，就自动反色。
    # 用整张图的平均亮度做简单判断，便于手机拍照或画图软件生成的图片使用。
    if ImageStat.Stat(image).mean[0] > 127:
        image = ImageOps.invert(image)

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # unsqueeze(0) 添加批量维度：[1, 28, 28] → [1, 1, 28, 28]。
    return transform(image).unsqueeze(0)


def main() -> None:
    """加载模型、完成预测并显示每个数字的概率。"""
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"找不到图片: {args.image}")
    if not args.model.exists():
        raise FileNotFoundError(f"找不到模型: {args.model}，请先运行 python train.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    model = SimpleCNN(dropout=checkpoint.get("dropout", 0.25)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor = prepare_image(args.image).to(device)
    # 推理不需要梯度，可以节省内存和计算时间。
    with torch.no_grad():
        logits = model(image_tensor)
        # 训练时损失函数内部处理 Softmax；展示概率时才显式使用它。
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_digit = probabilities.argmax().item()

    print(f"预测数字: {predicted_digit}")
    print(f"置信度: {probabilities[predicted_digit].item():.2%}")
    print("各数字概率:")
    print("  ".join(f"{digit}: {probability.item():.2%}" for digit, probability in enumerate(probabilities)))


if __name__ == "__main__":
    main()
