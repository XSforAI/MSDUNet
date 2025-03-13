import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import torchvision
from mmcv.cnn import build_activation_layer
from thop import profile
from timm.layers import trunc_normal_

from models.backbone.transxnet import transxnet_b, transxnet_s, transxnet_t, transxnet_xs, transxnet_xxs
from ptflops import get_model_complexity_info


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * residual

class FU(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(FU, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x, skip):
        offsets = self.offset_net(x)
        out = self.deform_conv(skip, offsets)
        return out

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

class MultiScaleDeformConv_3x3(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleDeformConv_3x3, self).__init__()
        self.sub_channel = in_channels // 4
        groups = self.sub_channel
        self.deform_conv1 = nn.Conv2d(self.sub_channel, groups, kernel_size=(1, 1))
        self.deform_conv3 = DeformConv(self.sub_channel, groups, kernel_size=(3, 3), padding=1, dilation=1)
        self.deform_conv5 = DeformConv(self.sub_channel, groups, kernel_size=(3, 3), padding=2, dilation=2)
        self.deform_conv7 = DeformConv(self.sub_channel, groups, kernel_size=(3, 3), padding=3, dilation=3)

    def forward(self, x):
        # Split the input tensor along the channel dimension into 4 equal parts
        channel_split = torch.chunk(x, 4, dim=1)

        # Apply deformable convolution with different kernel sizes
        out1 = self.deform_conv1(channel_split[0])
        out3 = self.deform_conv3(channel_split[1])
        out5 = self.deform_conv5(channel_split[2])
        out7 = self.deform_conv7(channel_split[3])

        # Concatenate the results along the channel dimension
        out = torch.cat([out1, out3, out5, out7], dim=1)

        return out

class MSDCDecoder_3x3_LS_up(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0,
                 shift_size=5,
                 layer_scale_init_value=1e-5,
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDeformConv_3x3(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.drop = nn.Dropout(drop)

        self.norm1 = nn.BatchNorm2d(in_features)
        self.up_conv = up_layer(out_features, out_features)
        # self.shift_size = shift_size
        # self.padding = shift_size // 2

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale(out_features, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_up, x_skip):

        x = torch.cat([x_up, x_skip], dim=1)
        x = self.norm1(x)
        B, C, H, W = x.shape

        # x = self.shift(x, 2, H, W)

        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.drop(x)

        # x = self.shift(x, 3, H, W)

        x = self.fc2(x)
        x = self.drop(x)

        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # x = self.norm1(x)
        x = self.up_conv(x)
        x = self.layer_scale(x) + x

        return x

class up_layer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_layer, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x

class MSDUFormer_Xnet_3463_FUESMS(nn.Module):   #MSDUnet
    def __init__(self, num_classes=4, img_size=512, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], linear=False):
        super().__init__()
        dpr = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

        self.backbone = transxnet_xxs()
        path = './pretrained_pth/transxnet/transx-s.pth.tar'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict, strict=False)

        self.up_last = up_layer(embed_dims[3], embed_dims[2])

        self.msecoder_3 = MSDCDecoder_3x3_LS_up(in_features=embed_dims[2]*2, hidden_features=embed_dims[2] * 2,
                                    out_features= embed_dims[1], drop=dpr[4])
        self.msecoder_2 = MSDCDecoder_3x3_LS_up(in_features=embed_dims[1]*2, hidden_features=embed_dims[1] * 2,
                                    out_features=embed_dims[0], drop=dpr[5])
        self.msecoder_1 = MSDCDecoder_3x3_LS_up(in_features=embed_dims[0]*2, hidden_features=embed_dims[0] * 2,
                                    out_features=embed_dims[0], drop=dpr[6])

        self.FU_3 = FU(embed_dims[2], embed_dims[2], kernel_size=(7, 7), padding=6, dilation=2)
        self.FU_2 = FU(embed_dims[1], embed_dims[1], kernel_size=(5, 5), padding=4, dilation=2)
        self.FU_1 = FU(embed_dims[0], embed_dims[0], kernel_size=(3, 3), padding=2, dilation=2)
        self.FU_E = SpatialAttention()


        self.up_top = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder0 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x0, x1, x2, x3 = self.backbone(x)
        x_3_up = self.up_last(x3)
        x2 = self.FU_E(self.FU_3(x_3_up, x2))
        d3 = self.msecoder_3(x_3_up, x2)
        x1 = self.FU_E(self.FU_2(d3, x1))
        d2 = self.msecoder_2(d3, x1)
        x0 = self.FU_E(self.FU_1(d2, x0))
        d1 = self.msecoder_1(d2, x0)
        d0 = self.up_top(d1)
        out = self.decoder0(d0)



        return out



if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    # models = edgevit_xxs()
    models = MSDUFormer_Xnet_3463_FUESMS(num_classes=10)

    print(models)

    # macs, paarms = get_model_complexity_info(models, (3, 224, 224))
    # print('{:<30}  {:<8}'.format('CC:', macs))
    # print('{:<30}  {:<8}'.format('param:', paarms))
    flops, params = profile(models, inputs=(x, ))
    print("参数量: {:.2f}M".format(params / 1e6))
    print("计算量: {:.2f}G".format(flops / 1e9))
    out = models(x)

    # print(out)





