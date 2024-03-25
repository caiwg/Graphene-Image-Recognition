import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDenseBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels + 3 * out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = F.relu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = F.relu(self.conv4(torch.cat([x, x1, x2, x3], 1)))

        out = torch.cat([x, x1, x2, x3, x4], 1)
        return out


class ResidualDenseNet(nn.Module):
    def __init__(self, num_blocks, out_channels, in_channels, final_out_channels):
        super(ResidualDenseNet, self).__init__()

        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualDenseBlock(in_channels, out_channels))
            in_channels += out_channels * 4  # 修正这里的系数

        self.residual_dense_blocks = nn.Sequential(*layers)
        self.conv_final = nn.Conv2d(in_channels, final_out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.residual_dense_blocks(x)
        x = self.conv_final(x)
        return x
