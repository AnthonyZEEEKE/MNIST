import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from digit_recognizer import MNISTCNN

# 全局配置
RANDOM_SEED = np.random.randint(1, 10000)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seeds(seed):
    """设置所有随机种子以确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_data_transforms():
    """创建训练数据转换管道"""
    return transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def prepare_dataloaders(batch_size=128):
    """准备训练和验证数据加载器"""
    transform = create_data_transforms()
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    # 随机划分训练集和验证集
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    return {
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    }

def initialize_model():
    """初始化模型并加载到设备"""
    model = MNISTCNN().to(DEVICE).float()
    return model


def train_network(model, dataloaders, epochs=15, lr=0.0012):
    """训练神经网络模型"""
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 初始化训练统计
    train_metrics = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']}
    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 50)

        # 训练阶段
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播和反向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)

        # 计算训练集指标
        epoch_train_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_train_acc = running_corrects.float() / len(dataloaders['train'].dataset)

        # 验证阶段
        epoch_val_loss, epoch_val_acc = evaluate_model(model, dataloaders['val'], criterion)

        # 更新学习率调度器
        scheduler.step()

        # 记录指标
        train_metrics['epoch'].append(epoch+1)
        train_metrics['train_loss'].append(epoch_train_loss)
        train_metrics['val_loss'].append(epoch_val_loss)
        train_metrics['train_acc'].append(epoch_train_acc.item())
        train_metrics['val_acc'].append(epoch_val_acc.item())

        # 打印 epoch 统计
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        print(f'Val   Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'mnist_cnn_best.pth')
            print(f'Best model saved with accuracy: {best_val_acc:.4f}')

    return pd.DataFrame(train_metrics)


def evaluate_model(model, dataloader, criterion):
    """在验证集上评估模型性能"""
    model.eval()
    running_loss, running_corrects = 0.0, 0

    # 禁用梯度计算
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 统计
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)

    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.float() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def visualize_training_results(metrics_df):
    """可视化训练结果"""
    plt.figure(figsize=(14, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    sns.lineplot(data=metrics_df, x='epoch', y='train_loss', marker='o', label='训练损失')
    sns.lineplot(data=metrics_df, x='epoch', y='val_loss', marker='s', label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    sns.lineplot(data=metrics_df, x='epoch', y='train_acc', marker='o', label='训练准确率')
    sns.lineplot(data=metrics_df, x='epoch', y='val_acc', marker='s', label='验证准确率')
    plt.title('训练与验证准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('C:\\Users\\Lenovo\\Downloads\\MNIST\\training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # 设置随机种子
    set_random_seeds(RANDOM_SEED)
    print(f'使用随机种子: {RANDOM_SEED}')
    print(f'使用设备: {DEVICE}')

    # 准备数据
    dataloaders = prepare_dataloaders(batch_size=128)
    print('数据加载完成')

    # 创建模型
    model = initialize_model()
    print('模型初始化完成')

    # 训练模型
    metrics = train_network(model, dataloaders, epochs=15)

    # 可视化结果
    visualize_training_results(metrics)
    print('训练完成，结果已保存')


if __name__ == '__main__':
    main()