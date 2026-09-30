"""定义一个适合初学者学习的 MNIST 卷积神经网络（CNN）。"""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """使用两层卷积识别 0～9 手写数字的简单 CNN。"""

    def __init__(self, dropout: float = 0.25) -> None:
        """创建网络中的所有层。

        参数:
            dropout: 随机丢弃神经元的比例，用来减轻过拟合。
        """
        # 调用父类构造方法。所有自定义 PyTorch 网络都必须这样做。
        super().__init__()

        # 第一组特征提取层：
        # 输入形状是 [批量大小, 1, 28, 28]，1 表示灰度图只有一个通道。
        # 16 个 3×3 卷积核会学习边缘、短线等简单特征。
        # padding=1 在图片四周补一圈 0，使卷积后的宽高仍然是 28×28。
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        # 第二组特征提取层：
        # 输入是第一层得到的 16 张特征图，输出 32 张更丰富的特征图。
        # 它会把边缘组合成圆弧、拐角和数字局部形状。
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        # ReLU 把负数变成 0，保留正数，让网络能够学习非线性关系。
        self.relu = nn.ReLU()

        # 2×2 最大池化从每个小区域保留最大值，使宽和高都缩小一半。
        # 池化可以减少计算量，并让轻微移动的数字仍容易被识别。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 输入 28×28，经过两次池化后：28→14→7。
        # 此时共有 32 张 7×7 的特征图，所以全连接层输入数是 32×7×7。
        # 128 个神经元负责综合前面提取出的局部特征。
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)

        # Dropout 只在训练时随机把一部分特征变成 0，避免网络死记训练图片。
        self.dropout = nn.Dropout(p=dropout)

        # 最后输出 10 个分数，分别对应数字 0、1、...、9。
        # 这里不使用 Softmax，因为 CrossEntropyLoss 内部已经包含相应计算。
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """规定一批图片从输入到输出依次经过哪些层。"""
        # [N, 1, 28, 28] → [N, 16, 28, 28] → [N, 16, 14, 14]
        x = self.pool(self.relu(self.conv1(x)))

        # [N, 16, 14, 14] → [N, 32, 14, 14] → [N, 32, 7, 7]
        x = self.pool(self.relu(self.conv2(x)))

        # 从三维特征图摊平成一维向量：[N, 32, 7, 7] → [N, 1568]。
        # 从第 1 维开始展开，可以保留第 0 维的批量大小 N。
        x = torch.flatten(x, start_dim=1)

        # 综合特征，然后在训练时执行 Dropout。
        x = self.dropout(self.relu(self.fc1(x)))

        # 返回 10 个未经 Softmax 的原始分数（logits）。
        return self.fc2(x)


if __name__ == "__main__":
    # 直接运行 model.py 时，用一批模拟图片检查每一层形状是否正确。
    model = SimpleCNN()
    example_images = torch.randn(4, 1, 28, 28)
    example_outputs = model(example_images)
    print(model)
    print(f"输入形状: {tuple(example_images.shape)}")
    print(f"输出形状: {tuple(example_outputs.shape)}")
