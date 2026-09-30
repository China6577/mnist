"""训练、验证并测试 SimpleCNN。

初学者可以先直接运行：python train.py
训练过程会自动下载 MNIST，并把最佳模型保存在 outputs/best_model.pth。
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import SimpleCNN


# MNIST 训练集像素的平均值和标准差，是该数据集常用的标准化参数。
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def parse_args() -> argparse.Namespace:
    """读取命令行参数；默认值已经是一套容易得到好结果的配置。"""
    parser = argparse.ArgumentParser(description="使用简单 CNN 识别 MNIST 数字")
    parser.add_argument("--epochs", type=int, default=5, help="完整学习训练集的次数")
    parser.add_argument("--batch-size", type=int, default=128, help="每次送入网络的图片数")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam 学习率")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout 丢弃比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，便于复现实验")
    parser.add_argument("--num-workers", type=int, default=0, help="读取数据的子进程数；Windows 默认 0 最稳妥")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="数据保存目录")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="结果保存目录")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="调参时不查看测试集；确定最终参数后再测试",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """固定各处随机数，让相同参数的多次实验尽量得到相同结果。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    """优先使用 NVIDIA GPU，没有 GPU 时自动使用 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_data_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """下载 MNIST，并创建训练、验证、测试数据加载器。"""
    # ToTensor 将图片变成 [0, 1] 的张量；Normalize 再进行标准化。
    # 标准化让输入数值分布更稳定，通常能使梯度下降更容易训练。
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )

    # 官方训练集有 60,000 张图片，测试集有 10,000 张图片。
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # 留出 6,000 张验证图片用于选参数；测试集只用于最终评价。
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        lengths=[54_000, 6_000],
        generator=generator,
    )

    # pin_memory 在使用 GPU 时能加快 CPU 到 GPU 的数据传输。
    pin_memory = torch.cuda.is_available()

    # 训练集需要打乱，防止网络按照固定的数据顺序学习。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # 验证集和测试集只做评价，不需要打乱。
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """让模型完整学习一遍训练集，返回平均损失和准确率。"""
    # 切换到训练模式，使 Dropout 生效。
    model.train()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        # 把当前批次的图片和正确答案放到 CPU 或 GPU 上。
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一批数据留下的梯度，否则 PyTorch 会自动累加梯度。
        optimizer.zero_grad()

        # 1. 前向传播：让网络根据图片算出 10 个类别分数。
        logits = model(images)
        # 2. 计算损失：衡量预测与正确答案之间有多大差距。
        loss = loss_function(logits, labels)
        # 3. 反向传播：根据损失自动计算每个参数的梯度。
        loss.backward()
        # 4. 更新参数：Adam 按梯度调整卷积核和全连接层权重。
        optimizer.step()

        # loss.item() 是当前批次的平均损失，乘批量大小后再统一求平均。
        current_batch_size = labels.size(0)
        total_loss += loss.item() * current_batch_size
        predictions = logits.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        sample_count += current_batch_size

    return total_loss / sample_count, correct_count / sample_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """评价模型但不修改参数，返回平均损失和准确率。"""
    # 评价模式会关闭 Dropout，使每次预测稳定一致。
    model.eval()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_function(logits, labels)

        current_batch_size = labels.size(0)
        total_loss += loss.item() * current_batch_size
        predictions = logits.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        sample_count += current_batch_size

    return total_loss / sample_count, correct_count / sample_count


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    """把每轮指标保存成 CSV，并绘制容易观察的学习曲线。"""
    csv_path = output_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_accuracy"] * 100 for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["validation_accuracy"] * 100 for row in history], marker="o", label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """按照准备数据、创建模型、训练、测试的顺序运行完整实验。"""
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs、batch-size 必须大于等于 1，learning-rate 必须大于 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    device = choose_device()
    print(f"运行设备: {device}")
    print("正在准备 MNIST 数据……")
    train_loader, validation_loader, test_loader = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.seed
    )

    # 创建模型并把其参数放到选择的设备上。
    model = SimpleCNN(dropout=args.dropout).to(device)
    print(model)

    # 多分类问题使用交叉熵损失。它会同时完成 LogSoftmax 和负对数似然计算。
    loss_function = nn.CrossEntropyLoss()
    # Adam 会为每个参数自动调节更新幅度，对初学者而言稳定且少调参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_validation_accuracy = 0.0
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_function, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"训练损失 {train_loss:.4f}, 训练准确率 {train_accuracy:.2%} | "
            f"验证损失 {validation_loss:.4f}, 验证准确率 {validation_accuracy:.2%}"
        )

        # 只保存验证准确率最高的一轮，而不是盲目保存最后一轮。
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dropout": args.dropout,
                    "validation_accuracy": validation_accuracy,
                    "epoch": epoch,
                },
                args.output_dir / "best_model.pth",
            )

    # 重新加载验证集表现最佳的轮次。
    checkpoint = torch.load(args.output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    elapsed_seconds = time.perf_counter() - start_time

    save_history(history, args.output_dir)
    results = {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "best_epoch": checkpoint["epoch"],
        "best_validation_accuracy": checkpoint["validation_accuracy"],
        "training_seconds": elapsed_seconds,
    }

    # 调参实验设置 --skip-test，避免反复查看测试集造成“答案泄露”。
    # 最终参数确定以后不设置该选项，只对测试集评价一次。
    if not args.skip_test:
        test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
        results["test_loss"] = test_loss
        results["test_accuracy"] = test_accuracy
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"\n最佳轮次: {checkpoint['epoch']}")
    if args.skip_test:
        print("调参模式：未查看测试集。")
    else:
        print(f"测试集损失: {results['test_loss']:.4f}")
        print(f"测试集准确率: {results['test_accuracy']:.2%}")
    print(f"总训练时间: {elapsed_seconds:.1f} 秒")
    print(f"模型和曲线已保存到: {args.output_dir.resolve()}")


if __name__ == "__main__":
    # Windows 多进程环境需要把程序入口放在这个判断中。
    main()
