import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1), padding=(1, 0))
        conv3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(1, 3), padding=(0, 1))
        conv4 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        for conv in [conv1, conv2, conv3, conv4]:
            self.convs.append(conv)

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        for i in range(4):
            x = self.convs[i](x)
            if i == 3:
                x = self.norm(x)
            x = F.relu(x)
        return x


class LPRNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)), BasicBlock(64, 128)
        )
        self.block3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            BasicBlock(64, 256),
            BasicBlock(256, 256),
        )
        self.block4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(64, 256, kernel_size=(1, 4)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, kernel_size=(13, 1)),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )
        self.last_conv = nn.Conv2d(448 + num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x1 = F.avg_pool2d(x1, kernel_size=5, stride=5)
        x2 = F.avg_pool2d(x2, kernel_size=5, stride=5)
        x3 = F.avg_pool2d(x3, kernel_size=(4, 10), stride=(4, 2))

        contexts = []
        for context in [x1, x2, x3, x4]:
            mean = torch.mean(context.pow(2))
            context = torch.div(context, mean)
            contexts.append(context)

        x = torch.cat(contexts, dim=1)
        x = self.last_conv(x)
        logits = torch.mean(x, dim=2)

        return logits
