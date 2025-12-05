import torch
import torch.nn as nn
from torch.nn import init
from torchsummary import summary


# MNIST数字识别卷积神经网络实现
class MNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTCNN, self).__init__()

        # 输入层到卷积层的转换
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # 深度卷积特征提取
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接分类器
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

        # 参数初始化
        self._initialize_weights()

    def _initialize_weights(self):
        # 卷积层参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # 批归一化层参数初始化
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            # 全连接层参数初始化
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    def forward(self, x):
        # 前向传播过程
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    # 模型结构打印
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTCNN().to(device)
    print('MNIST CNN Model Summary:')
    summary(model, (1, 28, 28))

