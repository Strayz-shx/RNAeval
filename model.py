import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34
from torch.nn import Conv1d, Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d


class Cnn_trans(nn.Module):
    def __init__(self, block, layers, layers_struc, out_channels):
        super(Cnn_trans, self).__init__()
        self.in_channels = 32
        self.in_channels_struc = 32
        self.conv1 = nn.Conv1d(in_channels=645, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.layer1_struc = self._make_layer_struc(block, 64, layers_struc[0])
        # self.layer2_struc = self._make_layer_struc(block, 128, layers_struc[1], stride=2)
        # self.layer3_struc = self._make_layer_struc(block, 256, layers_struc[2], stride=2)
        # self.layer4_struc = self._make_layer_struc(block, 512, layers_struc[3], stride=2)

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        self.fc1 = nn.Linear(out_channels, 64)
        self.fc2 = nn.Linear(64, 3)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_layer_struc(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels_struc != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels_struc, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels_struc, out_channels, stride, downsample))
        self.in_channels_struc = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels_struc, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, y, free_energy=None, pair_prob=None):
        x = x.unsqueeze(3)
        struc = x.permute(0, 3, 1, 2)

        seq = y.permute(0, 2, 1)

        seq = torch.relu(self.conv1(seq))
        seq = seq.permute(0, 2, 1)
        seq = self.linear(seq)
        seq = torch.cat(
            (seq.unsqueeze(2).expand(-1, -1, seq.size(1), -1), seq.unsqueeze(1).expand(-1, seq.size(1), -1, -1)),
            dim=-1).contiguous()
        seq = seq.permute(0, 3, 1, 2)

        struc = torch.relu(self.bn2(self.conv2(struc)))
        struc = nn.MaxPool2d(kernel_size=2, stride=2)(struc)
        seq = nn.MaxPool2d(kernel_size=2, stride=2)(seq)

        seq = self.layer1(seq)
        struc = self.layer1_struc(struc)

        struc = struc + seq

        # seq = self.layer2(seq)
        struc = self.layer2(struc)

        # struc = struc + seq

        # seq = self.layer3(seq)
        struc = self.layer3(struc)

        # struc = struc + seq

        # seq = self.layer4(seq)
        struc = self.layer4(struc)

        # struc = struc + seq

        out = self.aap(struc)
        out = self.flatten(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        # out = self.sigmoid(out)

        return out


class LinearForD(nn.Module):
    def __init__(self, input_dim=32, output_dim=32):
        super(LinearForD, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (B, L, D)
        x = self.linear(x)  # Apply linear transformation to D dimension
        return x


class InceptionBlock_modify(nn.Module):
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

        # 分支3: 1x1卷积 -> 两个3x3卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, self.branch3_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.branch3_out),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.branch3_out, self.branch3_out, kernel_size=3,
            #           stride=1, padding=1),
            # nn.BatchNorm2d(self.branch3_out),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.branch3_out, self.branch3_out, kernel_size=5,
                      stride=stride, padding=2),  # 应用stride进行下采样
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
        # identity = self.shortcut(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # 拼接所有分支的输出
        out = torch.cat([b1, b2, b3, b4], dim=1)
        # out += identity
        out = F.relu(out)

        return out


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
                      stride=stride, padding=1),
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
        self.conv1 = nn.Conv1d(in_channels=645, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.linear = LinearForD(input_dim=16, output_dim=16)

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

    def forward(self, x, y):
        x = x.unsqueeze(-1)
        struc = x.permute(0, 3, 1, 2)  # 结构模态的信息
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



class RNAInception_modify(nn.Module):
    def __init__(self, out_channels):
        super(RNAInception_modify, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=645, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.linear = LinearForD(input_dim=16, output_dim=16)

        self.layer1 = self._make_layer(32, 64, stride=1, num_blocks=2)
        self.layer1_seq = self._make_layer(32, 64, stride=1, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, stride=2, num_blocks=2)
        # self.layer2_seq = self._make_layer(64, 128, stride=2, num_blocks=2)
        self.layer3 = self._make_layer(128, 256, stride=2, num_blocks=2)
        # self.layer3_seq = self._make_layer(128, 256, stride=2, num_blocks=2)
        self.layer4 = self._make_layer(256, 512, stride=2, num_blocks=2)
        # self.layer4_seq = self._make_layer(256, 512, stride=2, num_blocks=2)
        # self.layer5 = self._make_layer(512,1024,stride=2,num_blocks=2)


        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 2)

    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        # 第一个块处理下采样和通道数变化
        layers.append(InceptionBlock_modify(in_channels, out_channels, stride))
        # 后续块保持通道数和尺寸
        for _ in range(1, num_blocks):
            layers.append(InceptionBlock_modify(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = x.unsqueeze(-1)
        struc = x.permute(0, 3, 1, 2)  # 结构模态的信息
        seq = y.permute(0, 2, 1)
        # seq = seq[:, 5:, :]

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
        # seq = self.layer2_seq(seq)

        # struc = struc+seq
        struc = self.layer3(struc)
        # seq = self.layer3_seq(seq)

        # struc = struc+seq
        struc = self.layer4(struc)
        # seq = self.layer4_seq(struc)
        # struc = struc+seq
        # struc = self.layer5(struc)

        out = self.aap(struc)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = F.relu(out)

        return out


class RNACnn_trans(nn.Module):
    def __init__(self, block, layers, layers_struc, out_channels):
        super(RNACnn_trans, self).__init__()
        self.in_channels = 32
        self.in_channels_struc = 32
        self.conv1 = nn.Conv1d(in_channels=645, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.layer1_struc = self._make_layer_struc(block, 64, layers_struc[0])
        self.layer2_struc = self._make_layer_struc(block, 128, layers_struc[1], stride=2)
        self.layer3_struc = self._make_layer_struc(block, 256, layers_struc[2], stride=2)
        self.layer4_struc = self._make_layer_struc(block, 512, layers_struc[3], stride=2)

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        self.fc1 = nn.Linear(out_channels, 64)
        self.fc2 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


    def _make_layer_struc(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels_struc != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels_struc, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels_struc, out_channels, stride, downsample))
        self.in_channels_struc = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels_struc, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, y, free_energy=None, pair_prob=None):
        x = x.unsqueeze(-1)
        struc = x.permute(0, 3, 1, 2)  # 结构模态的信息
        seq = y.permute(0, 2, 1)

        seq = torch.relu(self.conv1(seq))
        seq = seq.permute(0, 2, 1)
        seq_trans = torch.cat((seq.unsqueeze(2).expand(-1, -1, seq.size(1), -1), seq.unsqueeze(1).expand(-1, seq.size(1), -1, -1)), dim=-1).contiguous()
        seq_trans = seq_trans.permute(0, 3, 1, 2)

        struc = torch.relu(self.bn2(self.conv2(struc)))

        struc = nn.MaxPool2d(kernel_size=2, stride=2)(struc)
        seq = nn.MaxPool2d(kernel_size=2, stride=2)(seq_trans)

        seq = self.layer1(seq)
        struc = self.layer1_struc(struc)

        struc = struc + seq
        struc = self.layer2_struc(struc)
        struc = self.layer3_struc(struc)
        struc = self.layer4_struc(struc)

        out = self.aap(struc)
        out = self.flatten(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        # out = self.sigmoid(out)

        return out


class ResNet_18_grayscale_mat(nn.Module):

    def __init__(self, input_num_channels=4):
        super(ResNet_18_grayscale_mat, self).__init__()
        self.resnet_18 = resnet18(pretrained=False)
        self.resnet_18.conv1 = nn.Conv2d(in_channels=input_num_channels, out_channels=64, kernel_size=7, stride=2,
                                         padding=3, bias=False)
        self.input_num_features = self.resnet_18.fc.in_features
        self.resnet_18.fc = nn.Linear(in_features=self.input_num_features, out_features=2)

    def forward(self, x):
        x = self.resnet_18(x)
        return x


class ResNet_18_pair_grayscale_mat_nt_localized_info_mat(nn.Module):

    def __init__(self, nt_info_mat_input_num_channels=1, grayscale_mat_input_num_channels=4):
        super(ResNet_18_pair_grayscale_mat_nt_localized_info_mat, self).__init__()
        self.resnet_18_color_mat = resnet18(pretrained=False)
        self.resnet_18_color_mat.conv1 = nn.Conv2d(in_channels=grayscale_mat_input_num_channels, out_channels=64,
                                                   kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_18_color_mat_modified = nn.Sequential(*list(self.resnet_18_color_mat.children())[:-1], nn.Flatten())
        self.resnet_18_nt_localized_mat = resnet18(pretrained=False)
        self.resnet_18_nt_localized_mat.conv1 = nn.Conv2d(in_channels=nt_info_mat_input_num_channels, out_channels=64,
                                                          kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_18_nt_localized_mat_modified = nn.Sequential(*list(self.resnet_18_nt_localized_mat.children())[:-1],
                                                                 nn.Flatten())
        self.linear_layer_1 = nn.Linear(in_features=1024, out_features=100)
        self.linear_layer_2 = nn.Linear(in_features=100, out_features=100)
        self.linear_layer_3 = nn.Linear(in_features=100, out_features=2)

    def forward(self, color_mat, nt_localized_info_mat):
        x_1 = self.resnet_18_color_mat_modified(color_mat)
        x_2 = self.resnet_18_nt_localized_mat_modified(nt_localized_info_mat)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.linear_layer_1(x)
        x = self.linear_layer_2(x)
        x = self.linear_layer_3(x)
        return x




