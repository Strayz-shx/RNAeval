
from torchvision.models import resnet18
import torch.nn as nn
import torch
import torch.nn.functional as F


class ResNet_18_grayscale_mat(nn.Module):
    
    def __init__(self, input_num_channels=4):
        super(ResNet_18_grayscale_mat, self).__init__()
        self.resnet_18 = resnet18(pretrained=False)
        self.resnet_18.conv1 = nn.Conv2d(in_channels=input_num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.input_num_features = self.resnet_18.fc.in_features
        self.resnet_18.fc = nn.Linear(in_features=self.input_num_features, out_features=2)
        
    
    def forward(self, x):
        x = self.resnet_18(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 动态计算各分支的输出通道数
        self.branch1_out = out_channels // 4
        self.branch2_out = out_channels // 2
        self.branch3_out = out_channels // 8
        self.branch4_out = out_channels - (self.branch1_out + self.branch2_out + self.branch3_out)

        # 分支1: 1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, self.branch1_out, kernel_size=1, stride=stride),  # 应用stride进行下采样
            nn.BatchNorm2d(self.branch1_out),
            nn.ReLU(inplace=True)
        )

        # 分支2: 1x1卷积 -> 3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, self.branch2_out // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.branch2_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.branch2_out // 2, self.branch2_out, kernel_size=3,
                      stride=stride, padding=1),  # 应用stride进行下采样
            nn.BatchNorm2d(self.branch2_out),
            nn.ReLU(inplace=True)
        )

        # 分支3: 1x1卷积 -> 5x5卷积（拆分为两个3x3卷积）
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, self.branch3_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.branch3_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.branch3_out, self.branch3_out, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(self.branch3_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.branch3_out, self.branch3_out, kernel_size=3,
                      stride=stride, padding=1),  # 应用stride进行下采样
            nn.BatchNorm2d(self.branch3_out),
            nn.ReLU(inplace=True)
        )

        # 分支4: 3x3最大池化 -> 1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),  # 应用stride进行下采样
            nn.Conv2d(in_channels, self.branch4_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.branch4_out),
            nn.ReLU(inplace=True)
        )

        # 快捷连接（处理通道和尺寸变化）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # 拼接所有分支的输出
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out += identity
        out = F.relu(out)

        return out


class RNAInception(nn.Module):
    def __init__(self, out_channels):
        super(RNAInception, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=644, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 64, stride=1, num_blocks=2)
        self.layer1_seq = self._make_layer(32, 64, stride=1, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, stride=2, num_blocks=2)
        self.layer3 = self._make_layer(128, 256, stride=2, num_blocks=2)
        self.layer4 = self._make_layer(256, 512, stride=2, num_blocks=2)

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 2)

    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        # 第一个块处理下采样和通道数变化
        layers.append(InceptionBlock(in_channels, out_channels, stride))
        # 后续块保持通道数和尺寸
        for _ in range(1, num_blocks):
            layers.append(InceptionBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, y, x):
        # struc = x.permute(0, 3, 1, 2)  # 结构模态的信息
        struc = x
        seq = y.permute(0, 2, 1)

        seq = torch.relu(self.conv1(seq))
        seq = seq.permute(0, 2, 1)
        seq = torch.cat(
            (seq.unsqueeze(2).expand(-1, -1, seq.size(1), -1), seq.unsqueeze(1).expand(-1, seq.size(1), -1, -1)),
            dim=-1).contiguous()
        seq = seq.permute(0, 3, 1, 2)

        struc = F.relu(self.bn2(self.conv2(struc)))
        struc = nn.MaxPool2d(kernel_size=2, stride=2)(struc)
        seq = nn.MaxPool2d(kernel_size=2, stride=2)(seq)

        seq = self.layer1_seq(seq)
        struc = self.layer1(struc)

        struc = struc + seq
        struc = self.layer2(struc)
        struc = self.layer3(struc)
        struc = self.layer4(struc)

        out = self.aap(struc)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

# The following example is for testing the ResNet_18_grayscale_mat architecture
'''
NN_model = ResNet_18_grayscale_mat()
input = torch.randn((20,4,410,410))
result = NN_model(input)
print(result.shape)
'''