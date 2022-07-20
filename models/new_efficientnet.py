import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from efficientnet_pytorch import EfficientNet
from .SAIA import SAIA_conv, SAIA_conv_simple
from timm.models import tf_efficientnet_b4_ns
import pdb


class Efficientnet_Attv2(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes=1,
        pretrained=True,
        isattention=False,
        istest=False,
    ):
        super(Efficientnet_Attv2, self).__init__()

        self.model = tf_efficientnet_b4_ns(pretrained=True)

        num_ftrs = 1792
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
        self.layerstem = nn.Sequential(*list(self.model.children())[:3])

        self.layer0 = self.model.blocks[0]
        self.layer1 = self.model.blocks[1]
        self.layer2 = self.model.blocks[2]
        self.layer3 = self.model.blocks[3]
        self.layer4 = self.model.blocks[4]
        self.layer5 = self.model.blocks[5]
        self.layer6 = self.model.blocks[6]
        self.layertail = nn.Sequential(*list(self.model.children())[-5:-2])
        self.att0conv = SAIA_conv_simple(
            24, kernel_size=7, padding=3, isspace=True, ischannel=True
        )
        self.att1conv = SAIA_conv_simple(
            32, kernel_size=7, padding=5, isspace=True, ischannel=True
        )
        self.att2conv = SAIA_conv_simple(
            56, kernel_size=5, padding=3, isspace=True, ischannel=True
        )
        self.att3conv = SAIA_conv_simple(
            112, kernel_size=3, padding=1, isspace=True, ischannel=True
        )
        self.att4conv = SAIA_conv_simple(
            160, kernel_size=3, isspace=True, ischannel=True
        )
        self.att5conv = SAIA_conv_simple(
            272, kernel_size=3, isspace=True, ischannel=True
        )
        self.att6conv = SAIA_conv_simple(
            448, kernel_size=3, isspace=True, ischannel=True
        )
        self.avgpool1 = nn.AdaptiveMaxPool2d((38, 38))
        self.avgpool2 = nn.AdaptiveMaxPool2d((19, 19))
        self.avgpool0 = nn.AdaptiveMaxPool2d((75, 75))
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 56, 1, 1, 0),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 160, 1, 1, 0),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 112, 1, 1, 0),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(56, 160, 1, 1, 0),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(num_ftrs, num_classes)
        self.drop = nn.Dropout(p=0.2)

    def forward_features(self, x):

        bs = x.shape[0]

        x = self.layerstem(x)
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x1, att1 = self.att1conv(x1)
        res12 = self.avgpool1(self.conv1(att1))
        res14 = self.avgpool2(self.conv2(att1))
        res13 = self.avgpool2(self.conv4(att1))
        x2 = self.layer2(x1)
        x2, att2 = self.att2conv(x2 + res12)
        res23 = self.avgpool2(self.conv3(att2))
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x4, att4 = self.att4conv(x4 + res23 + res14)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layertail(x6)
        output = x7
        return output
