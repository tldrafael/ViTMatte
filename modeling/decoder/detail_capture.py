import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T


class Boosted_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        stride=2,
        padding=1,
    ):
        super().__init__()
        nlayers = 12
        ch_list = [in_chans] + [out_chans // 2] * (nlayers - 1) + [out_chans]
        strides = [2] + [1] * (nlayers - 1)
        paddings = [1] + ['same'] * (nlayers - 1)

        self.basics = nn.Sequential(*[
            Basic_Conv3x3(ch_list[i], ch_list[i+1], strides[i], paddings[i])
            for i in range(nlayers)
        ])

    def forward(self, x):
        x = self.basics(x)
        return x


class Basic_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    def __init__(
        self,
        in_chans = 4,
        out_chans = [48, 96, 192],
        block='basic',
    ):
        super().__init__()
        self.convs = nn.ModuleList()

        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)

        block = Basic_Conv3x3 if block == 'basic' else Boosted_Conv3x3

        for i in range(len(self.conv_chans)-1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i+1]
            self.convs.append(
                block(in_chan_, out_chan_)
            )

    def forward(self, x):
        out_dict = {'D0': x}
        for i, layer in enumerate(self.convs):
            x = layer(x)
            name_ = 'D'+str(i+1)
            out_dict[name_] = x

        return out_dict

class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
    ):
        self.in_chans = in_chans
        self.out_chans = out_chans

        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x, D):
        newsize = D.shape[-2:]
        F_up = T.functional.resize(
            x, size=newsize, interpolation=T.InterpolationMode.BILINEAR,
            antialias=False)
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)

        return out

class Matting_Head(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """
    def __init__(
        self,
        in_chans = 32,
        mid_chans = 16,
    ):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(True),
            nn.Conv2d(mid_chans, 1, 1, 1, 0)
            )

    def forward(self, x):
        x = self.matting_convs(x)

        return x

class Detail_Capture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    """
    def __init__(
        self,
        in_chans = 384,
        img_chans = 4,
        convstream_out = [48, 96, 192],
        fusion_out = [256, 128, 64, 32],
        patch_size = 16,
        block='basic',
    ):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1

        self.patch_size = patch_size
        self.convstream_out = convstream_out

        self.convstream = ConvStream(
            in_chans = img_chans, out_chans = convstream_out, block=block)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, in_chans)
        for i in range(len(self.fus_channs)-1):
            in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)]
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)],
                    out_chans = self.fus_channs[i+1],
                )
            )

        self.matting_head =  Matting_Head(
            in_chans = fusion_out[-1],
        )

    def forward(self, features, images):
        detail_features = self.convstream(images)
        # print(self.patch_size, self.convstream_out,
        #     [v.shape for k, v in detail_features.items()])

        for i, layer in enumerate(self.fusion_blks):
            d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
            features = layer(features, detail_features[d_name_])

        # phas = torch.sigmoid(self.matting_head(features))
        # return {'phas': phas}
        return self.matting_head(features)
