import torch.nn as nn
import torch
import pdb
import numpy as np
import cv2
from torch.nn import functional as F


def save_feature(x, name):
    image = np.abs(x.detach().cpu().numpy())
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image_norm = image_norm.astype(int)
    image_norm[image_norm < 0] = 0
    image_norm = cv2.applyColorMap(np.uint8(image_norm), cv2.COLORMAP_TURBO)
    cv2.imwrite(f"{name}", image_norm)


class SAIA_conv(nn.Module):
    def __init__(
        self,
        outdim,
        kernel_size=0,
        padding=0,
        isspace=True,
        ischannel=True,
        crop_size=20,
    ):
        super(SAIA_conv, self).__init__()

        self.outdim = outdim
        self.drop_rate = 0.3
        self.temperature = 0.03
        self.band_width = 1.0
        self.radius = 3
        self.isspace = isspace
        self.ischannel = ischannel

        self.patch_sampling_num = 9
        self.down_ratio = 16

        if self.isspace:
            num_channel_s = self.outdim
        if self.ischannel:
            num_channel_c = self.outdim
            self.W_channel = nn.Sequential(
                nn.Linear(num_channel_c, num_channel_c // self.down_ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(num_channel_c // self.down_ratio, num_channel_c, bias=False),
                nn.Sigmoid(),
            )

        self.padder = nn.ConstantPad2d(
            (
                padding + self.radius,
                padding + self.radius + 1,
                padding + self.radius,
                padding + self.radius + 1,
            ),
            0,
        )

    def forward(self, x_old):

        with torch.no_grad():
            distances = []
            padded_x_old = self.padder(x_old)
            sampled_i = torch.randint(
                -self.radius, self.radius + 1, size=(self.patch_sampling_num,)
            ).tolist()
            sampled_j = torch.randint(
                -self.radius, self.radius + 1, size=(self.patch_sampling_num,)
            ).tolist()
            for i, j in zip(sampled_i, sampled_j):
                tmp = (
                    padded_x_old[
                        :,
                        :,
                        self.radius : -self.radius - 1,
                        self.radius : -self.radius - 1,
                    ]
                    - padded_x_old[
                        :,
                        :,
                        self.radius + i : -self.radius - 1 + i,
                        self.radius + j : -self.radius - 1 + j,
                    ]
                )
                tmp = torch.exp(-(tmp**2) / 2 / self.band_width**2)
                distances.append(tmp.clone())

            distance = torch.cat(distances, dim=1)
            batch_size, _, h_dis, w_dis = distance.shape
            distance2 = -torch.log(distance)
            sum_dis2 = (
                distance2.view(batch_size, -1, self.patch_sampling_num, h_dis, w_dis)
                .sum(dim=2)
                .view(batch_size, -1, h_dis, w_dis)
            )
        if self.ischannel:
            distance_channel = sum_dis2[:]
            channel_attention = torch.mean(
                distance_channel.view(batch_size, self.outdim, -1), dim=2
            )
            channel_attention = self.W_channel(channel_attention).view(
                batch_size, -1, 1, 1
            )
        if self.isspace:

            distance_space = sum_dis2
            space_attention = distance_space
            batch_size, channels, h, w = x_old.shape
            attention_image = (nn.Sigmoid()(space_attention) * x_old) + x_old
        if self.isspace and self.ischannel:
            return (
                attention_image * (channel_attention.expand_as(x_old)),
                space_attention,
            )
        elif self.isspace:
            return attention_image
        elif self.ischannel:
            return x_old * (channel_attention.expand_as(x_old))


class SAIA_conv_simple(nn.Module):
    def __init__(self, outdim, kernel_size=3, padding=1, isspace=True, ischannel=True):
        super(SAIA_conv_simple, self).__init__()

        self.drop_rate = 0.3
        self.temperature = 0.03
        self.band_width = 1.0

        self.isspace = isspace
        self.ischannel = ischannel
        self.outdim = outdim

        kernel = torch.ones((outdim, 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        kernel2 = torch.ones((outdim, 1, 1, 1)) * (kernel_size * kernel_size)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.inorm = nn.InstanceNorm2d(outdim, affine=False)
        self.pad = padding
        self.channel_range = 5

    def forward(self, x):
        with torch.no_grad():
            batch_size = x.shape[0]
            num_channel = x.shape[1]

            x1 = F.conv2d(x, self.weight, padding=self.pad, groups=self.outdim)
            x2 = F.conv2d(x, self.weight2, padding=0, groups=self.outdim)
            intra_distance = torch.abs(x2 - x1)

            pad_x = torch.cat([x, x[:, : self.channel_range + 1, :, :]], dim=1)
            distances = []
            for i in range(1, self.channel_range + 1):
                tmp = x[:, :, :, :] - pad_x[:, i : num_channel + i, :, :]
                distances.append(tmp.clone())

            distance = torch.cat(distances, dim=1)
            batch_size, _, h_dis, w_dis = distance.shape
            distance = distance.view(
                batch_size, -1, self.channel_range, h_dis, w_dis
            ).sum(dim=2)
            inter_distance = torch.abs(distance.view(batch_size, -1, h_dis, w_dis))
            att = intra_distance + 0.1 * inter_distance
            bs, c, h, w = att.shape
        if self.ischannel:
            distance_channel = att[:]
            distance_channel = (
                distance_channel / distance_channel.mean() / 2 / self.band_width**2
            )
            channel_attention = torch.mean(
                distance_channel.view(batch_size, self.outdim, -1), dim=2
            )
            channel_attention = channel_attention.view(batch_size, -1, 1, 1) + 1

        if self.isspace:
            distance_space = att
            distance_space = (
                distance_space / distance_space.mean() / 2 / self.band_width**2
            )
            space_attention = distance_space
            batch_size, channels, h, w = x.shape
            attention_image = (nn.Sigmoid()(space_attention) + 1) * x

        if self.isspace and self.ischannel:
            return attention_image * (channel_attention.expand_as(x)), space_attention
        elif self.isspace:
            return attention_image, x
        elif self.ischannel:
            return x * (channel_attention.expand_as(x)), x


class SAIA_metric(nn.Module):
    def __init__(
        self,
        outdim,
        kernel_size=3,
        padding=1,
        isspace=True,
        ischannel=True,
        crop_size=0,
    ):
        super(SAIA_metric, self).__init__()

        self.drop_rate = 0.3
        self.temperature = 0.03
        self.band_width = 1.0

        self.isspace = isspace
        self.ischannel = ischannel
        self.outdim = outdim
        kernel = torch.ones((outdim, 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        kernel2 = torch.ones((outdim, 1, 1, 1)) * (kernel_size * kernel_size)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.inorm = nn.InstanceNorm2d(outdim, affine=False)
        self.pad = padding
        self.channel_range = 5
        self.crop_size = crop_size

    def forward(self, x):
        with torch.no_grad():
            batch_size = x.shape[0]
            num_channel = x.shape[1]
            # intra-feature
            x1 = F.conv2d(x, self.weight, padding=self.pad, groups=self.outdim)
            x2 = F.conv2d(x, self.weight2, padding=0, groups=self.outdim)
            intra_distance = torch.abs(x2 - x1)
            att = intra_distance
            bs, c, h, w = att.shape
            if self.crop_size != 0:
                att[:, :, : self.crop_size, :] = 0
                att[:, :, h - self.crop_size :, :] = 0
                att[:, :, :, : self.crop_size] = 0
                att[:, :, :, h - self.crop_size :] = 0
        return att
