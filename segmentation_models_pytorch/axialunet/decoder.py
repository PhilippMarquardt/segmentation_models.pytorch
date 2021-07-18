import torch
import torch.nn as nn
import torch.nn.functional as F
from .axial import AxialImageTransformer
from ..base import modules as md
from ..transunet.vit import ViT

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
            image_size = 128
    ):
        super().__init__()
        self.image_size = image_size
        self.axial = AxialImageTransformer(dim = out_channels, depth = 4,axial_pos_emb_shape = (image_size,image_size), heads=8)

        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

            
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        x = self.axial(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class AxialUnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            image_size = 128
    ):
        
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels



        # self.trans = ViT(
        #     image_size = int(image_size / (2**(n_blocks - 1))),
        #     patch_size = 1,
        #     dim = head_channels,
        #     depth = 6,
        #     heads = 12,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1,
        #     channels = head_channels
        # )
        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, image_size=int(image_size / (2**(n_blocks - (cnt + 1)))), **kwargs)
            for cnt, (in_ch, skip_ch, out_ch) in enumerate(zip(in_channels, skip_channels, out_channels))
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        #head = self.trans(features[0])
        head = features[0]
        skips = features[1:]
        x = head
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x
