import torch
import torch.nn as nn
import torch.nn.functional as F


# Faster implementation of layer normalization from Meta Research
# ConvNeXt: https://github.com/facebookresearch/ConvNeXt


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


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

        self.norm = LayerNorm(out_channels, eps=1e-6, data_format='channels_first')
        self.gamma = nn.Parameter(1e-4 * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        for i in range(4):
            x = self.convs[i](x)
            if i == 3:
                x = self.norm(x)
            x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return x


class LPRNetChina(nn.Module):
    """Net following original implementation of LPRNet (https://arxiv.org/abs/1806.10447)
    for Chinese license plates recognition.
    """

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


class LPRNetEU(nn.Module):
    """Implementation of LPRNet (https://arxiv.org/abs/1806.10447)
    modified for the EU license plates aspect ratio
    and improved with good practices of modern neural networks.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            LayerNorm(64, eps=1e-6, data_format='channels_first'),
            nn.GELU(),
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
            LayerNorm(256, eps=1e-6, data_format='channels_first'),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, kernel_size=(13, 1), padding=(0, 2)),
            LayerNorm(num_classes, eps=1e-6, data_format='channels_first'),
            nn.GELU(),
        )
        self.last_conv = nn.Conv2d(448 + num_classes, num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x1 = F.avg_pool2d(x1, kernel_size=5, stride=(3, 4), padding=(1, 1))
        x2 = F.avg_pool2d(x2, kernel_size=5, stride=(3, 4), padding=(1, 1))
        x3 = F.avg_pool2d(x3, kernel_size=(3, 5), stride=(3, 2), padding=(1, 2))

        contexts = []
        for context in [x1, x2, x3, x4]:
            mean = torch.mean(context.pow(2))
            context = torch.div(context, mean)
            contexts.append(context)

        x = torch.cat(contexts, dim=1)
        x = self.last_conv(x)
        logits = torch.mean(x, dim=2)

        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
            module.bias.data.fill_(0.01)
