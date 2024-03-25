import torch
import torch.nn as nn

from nets.MultiAttention import MultiHeadAttention
from nets.ResidualDenseNet import ResidualDenseNet
from nets.resnet import resnet50
from nets.selfAttention import ResidualDenseBlockAttention
from nets.vgg import VGG16


# 一次向上采样，目前经历了四次向上采样
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # 双线性上采样
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs




# 一次向上采样，目前经历了四次向上采样
class MulUnetUp(nn.Module):
    def __init__(self, in_size, out_size, in_heads, out_heads):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # 双线性上采样
        self.relu = nn.ReLU(inplace=True)

        # 添加多头注意力模块
        self.mha = MultiHeadAttention(in_size, out_size, in_heads, out_heads)

    def forward(self, inputs1, inputs2):
        # 将input1通过多头注意力模块
        inputs1 = self.mha(inputs1)

        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs



class Unet(nn.Module):
    #  初始化网络
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]  # 每个下采样层输出通道数的列表。
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]  # 每个上采样层输出通道数的列表
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))

        # self.rdn = ResidualDenseNet(num_blocks=5, out_channels=32, in_channels=1024, final_out_channels=256)
        self.rdn = ResidualDenseBlockAttention(in_channels=1024, out_channels=32, num_blocks=5, num_heads=8)

        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])  # 1024，512
        # self.up_concat4 = MulUnetUp(in_filters[3], out_filters[3], in_heads, out_heads)
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    # 根据主干网络提取图像特征，得到feat
    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        #  如果使用ResNet50作为主干，则对 up1 进行额外的上采样卷积。
        if self.up_conv is not None:
            up1 = self.up_conv(up1)
        #  最终通过卷积层 self.final 得到最终的分割结果 final。
        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
