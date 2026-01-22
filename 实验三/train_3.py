import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class CNNModel(nn.Module):
    """CNN模型定义"""

    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Dropout防止过拟合
            nn.Dropout2d(0.25)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x


class DigitRecognizer:
    """手写数字识别器"""

    def __init__(self, model_path='models/digit_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def load_data(self):
        """加载MNIST数据集"""
        print("正在加载MNIST数据集...")

        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=self.transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1000,
            shuffle=False,
            num_workers=2
        )

        return train_loader, test_loader

    def train(self, epochs=10, learning_rate=0.001):
        """训练模型"""
        train_loader, test_loader = self.load_data()

        # 初始化模型
        self.model = CNNModel().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 学习率调度
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        train_losses = []
        train_accs = []
        test_accs = []

        print("开始训练模型...")
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"测试集大小: {len(test_loader.dataset)}")

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch + 1}/{epochs} | '
                          f'Batch: {batch_idx}/{len(train_loader)} | '
                          f'Loss: {loss.item():.4f}')

            # 计算训练准确率
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # 测试阶段
            test_acc = self.evaluate(test_loader)
            test_accs.append(test_acc)

            scheduler.step()

            print(f'\nEpoch {epoch + 1} 结果:')
            print(f'  训练损失: {train_loss:.4f}')
            print(f'  训练准确率: {train_acc:.2f}%')
            print(f'  测试准确率: {test_acc:.2f}%')
            print('-' * 50)

        # 保存模型
        self.save_model()

        # 绘制训练曲线
        self.plot_training_curves(train_losses, train_accs, test_accs)

        return self.model

    def evaluate(self, test_loader):
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100. * correct / total
        return accuracy

    def save_model(self):
        """保存模型"""
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': str(self.model),
        }, self.model_path)
        print(f"模型已保存到 {self.model_path}")

    def load_model(self):
        """加载预训练模型"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = CNNModel().to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"模型已从 {self.model_path} 加载")
            return True
        else:
            print("未找到预训练模型")
            return False

    def plot_training_curves(self, losses, train_accs, test_accs):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        axes[0].plot(losses, label='Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].grid(True)
        axes[0].legend()

        # 准确率曲线
        axes[1].plot(train_accs, label='Training Accuracy')
        axes[1].plot(test_accs, label='Test Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Curves')
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('models/training_curves.png', dpi=150)
        plt.show()


def main():
    """主函数：训练模型"""
    print("=" * 60)
    print("手写数字识别模型训练")
    print("=" * 60)

    # 创建识别器
    recognizer = DigitRecognizer()

    # 训练或加载模型
    if not recognizer.load_model():
        print("开始训练新模型...")
        recognizer.train(epochs=15)

    # 评估模型
    train_loader, test_loader = recognizer.load_data()
    test_acc = recognizer.evaluate(test_loader)
    print(f"\n最终模型在测试集上的准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    main()