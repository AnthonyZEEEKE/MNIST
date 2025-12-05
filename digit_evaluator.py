import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from digit_recognizer import MNISTCNN
import random

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    # 类常量定义
    RANDOM_SEED = np.random.randint(1, 10000)
    PLOT_STYLE = 'classic'
    FIGURE_DPI = 300

    def __init__(self, model_path='mnist_cnn_best.pth', batch_size=128):
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载模型
        self.model = self._load_model(model_path)
        # 准备数据
        self.test_loader, self.test_dataset = self._prepare_test_data(batch_size)
        # 初始化评估指标
        self.true_labels = None
        self.pred_labels = None
        self.accuracy = 0.0

    def _load_model(self, model_path):
        # 加载训练好的模型
        model = MNISTCNN().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()  # 设置为评估模式
        return model

    def _prepare_test_data(self, batch_size):
        # 定义测试数据转换
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载测试数据集
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=test_transform
        )

        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        return test_loader, test_dataset

    def run_evaluation(self):
        # 执行完整评估流程
        print(f'\n开始模型评估 (使用设备: {self.device})')
        print('=' * 60)

        # 执行预测并计算指标
        self.true_labels, self.pred_labels = self._predict_and_evaluate()

        # 生成评估报告
        self._generate_classification_report()

        # 创建可视化结果
        self._visualize_results()

        print('\n评估完成! 结果已保存到当前目录')

    def _predict_and_evaluate(self):
        # 存储所有预测和真实标签
        all_preds = []
        all_labels = []
        total_correct = 0
        total_samples = 0

        # 禁用梯度计算以加速推理
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                # 将数据移动到设备
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                # 统计结果
                batch_size = images.size(0)
                total_samples += batch_size
                total_correct += (preds == labels).sum().item()

                # 收集预测和标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    print(f'处理批次: {batch_idx + 1}/{len(self.test_loader)}, ' \
                          f'当前准确率: {total_correct / total_samples:.4f}')

        # 计算总体准确率
        self.accuracy = total_correct / total_samples
        print(f'\n测试集准确率: {self.accuracy:.4f}')

        return np.array(all_labels), np.array(all_preds)

    def _generate_classification_report(self):
        # 生成详细分类报告
        print('\n===== 分类性能报告 =====')
        report = classification_report(
            self.true_labels,
            self.pred_labels,
            target_names=[str(i) for i in range(10)]
        )
        print(report)

        # 计算每类准确率
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for label, pred in zip(self.true_labels, self.pred_labels):
            class_correct[label] += (label == pred)
            class_total[label] += 1

        # 打印每类准确率
        print('\n===== 数字类别准确率 =====')
        for i in range(10):
            if class_total[i] > 0:
                print(f'数字 {i}: {100 * class_correct[i] / class_total[i]:.2f}% ' \
                      f'({int(class_correct[i])}/{int(class_total[i])})')

    def _visualize_results(self):
        # 设置绘图风格
        plt.style.use(self.PLOT_STYLE)

        # 绘制混淆矩阵
        self._plot_confusion_matrix()

        # 可视化错误分类样本
        self._display_misclassified_examples()

    def _plot_confusion_matrix(self):
        # 计算混淆矩阵
        cm = confusion_matrix(self.true_labels, self.pred_labels)

        # 绘制归一化和原始混淆矩阵
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # 原始计数混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('混淆矩阵 (样本计数)')
        ax1.set_xlabel('预测标签')
        ax1.set_ylabel('真实标签')

        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', ax=ax2)
        ax2.set_title('混淆矩阵 (归一化准确率)')
        ax2.set_xlabel('预测标签')
        ax2.set_ylabel('真实标签')

        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=self.FIGURE_DPI, bbox_inches='tight')
        plt.close()

    def _display_misclassified_examples(self, num_samples=12):
        # 设置随机种子以确保结果可复现
        random.seed(self.RANDOM_SEED)

        # 查找错误分类的样本
        misclassified = []
        max_attempts = min(2000, len(self.test_dataset))  # 限制最大尝试次数

        # 随机选择样本进行检查
        indices = random.sample(range(len(self.test_dataset)), max_attempts)

        # 检查哪些样本被错误分类
        with torch.no_grad():
            for idx in indices:
                image, true_label = self.test_dataset[idx]
                image_tensor = image.unsqueeze(0).to(self.device)
                output = self.model(image_tensor)
                _, pred_label = torch.max(output, 1)
                pred_label = pred_label.item()

                if pred_label != true_label:
                    # 存储错误分类样本的信息
                    misclassified.append({
                        'image': image.squeeze().numpy(),
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': torch.softmax(output, dim=1)[0][pred_label].item()
                    })

                # 达到所需样本数量则停止
                if len(misclassified) >= num_samples:
                    break

        # 如果找到足够的错误样本则绘制
        if misclassified:
            # 创建图像网格
            rows = (num_samples + 3) // 4  # 每行显示4个样本
            fig, axes = plt.subplots(rows, 4, figsize=(16, 3 * rows))
            axes = axes.flatten()  # 将axes转换为1D数组

            for i, sample in enumerate(misclassified):
                axes[i].imshow(sample['image'], cmap='gray_r')
                axes[i].set_title(f'真实: {sample["true_label"]}\n' \
                                  f'预测: {sample["pred_label"]}\n' \
                                  f'置信度: {sample["confidence"]:.2f}')
                axes[i].axis('off')

            # 隐藏未使用的子图
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.savefig('misclassified_examples.png', dpi=self.FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f'已保存 {len(misclassified)} 个错误分类样本的可视化结果')
        else:
            print('未找到足够的错误分类样本进行可视化')


if __name__ == '__main__':
    # 创建评估器实例
    evaluator = ModelEvaluator()

    # 运行完整评估流程
    evaluator.run_evaluation()

    # 打印最终准确率
    print(f'\n模型最终准确率: {evaluator.accuracy:.4f}')
    print(f'评估使用的随机种子: {ModelEvaluator.RANDOM_SEED}')

