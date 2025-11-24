import torch
import torch.nn as nn


# 创建一个简单的 PyTorch 模型（Sequential，适合融合）
class SimpleTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, 10),
        )
        self.features2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, 10),
        )
        
    def forward(self, x):
        return self.features(x) + self.features2(x)


# ResNet 基础块
class ResidualBlock(nn.Module):
    """ResNet 基础残差块"""
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.relu2 = nn.ReLU()
        
        # shortcut: 如果输入输出维度不同，需要投影
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        out = out + identity  # 残差连接
        out = self.relu2(out)
        
        return out


# 完整的 ResNet 模型
class LinearResNetModel(nn.Module):
    """简化版 ResNet，用于测试"""
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        
        # 初始层
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 残差块
        self.layer1 = ResidualBlock(128, 128, 128)
        self.layer2 = ResidualBlock(128, 256, 256)
        
        # 分类头
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x

# ...existing code...

class ResNetBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetConv18(nn.Module):
    """标准 ResNet18 结构（卷积版），输入格式 (N, C, H, W)"""
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResNetBasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResNetBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNetBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNetBasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResNetBasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, C, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SimpleRNNModel(nn.Module):
    """基于 nn.RNN 的简单序列模型。输入 (batch, seq_len, input_size)。"""
    def __init__(self, input_size=32, hidden_size=64, num_layers=1, num_classes=10):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.rnn(x)          # out: (B, T, H)
        last = out[:, -1, :]          # 取最后时间步
        return self.fc(last)


class SimpleLSTMModel(nn.Module):
    """基于 nn.LSTM 的简单序列模型。输入 (batch, seq_len, input_size)。"""
    def __init__(self, input_size=32, hidden_size=64, num_layers=1, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)  # out: (B, T, H)
        last = out[:, -1, :]
        return self.fc(last)


class SimpleTransformerModel(nn.Module):
    """基于 nn.TransformerEncoder 的简单 Transformer 分类器。
       期望输入 (batch, seq_len, d_model)。"""
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, num_classes=10):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 如果输入维度不是 d_model，用户可在外部先用线性投影
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        # TransformerEncoderLayer with batch_first=True accepts (B, T, D)
        out = self.encoder(x)        # (B, T, D)
        pooled = out.mean(dim=1)     # 平均池化
        return self.fc(pooled)

# ...existing code...