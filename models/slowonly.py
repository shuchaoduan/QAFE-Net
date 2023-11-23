# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from models.resnet3d import ResNet3d


class ResNet3dSlowOnly(nn.Module):
    """SlowOnly backbone based on ResNet3d.
    Args:
        conv1_kernel (tuple[int]): Kernel size of the first conv layer. Default: (1, 7, 7).
        inflate (tuple[int]): Inflate Dims of each block. Default: (0, 0, 1, 1).
        **kwargs (keyword arguments): Other keywords arguments for 'ResNet3d'.
    """

    def __init__(self, conv1_kernel=(1, 7, 7), inflate= (0, 1, 1),
                 in_channels = 3, base_channels = 32, num_stages = 3,
                 out_indices = (2,), stage_blocks = (4, 6, 3), conv1_stride = (1, 1),
                 pool1_stride=(1, 1),  spatial_strides=(2, 2, 2),
                 temporal_strides=(1, 1, 1), **kwargs):
        super().__init__()
        self.restnet3d = ResNet3d(conv1_kernel=conv1_kernel, inflate=inflate,
                         in_channels=in_channels, base_channels=base_channels, num_stages=num_stages,
                         out_indices=out_indices, stage_blocks=stage_blocks, conv1_stride=conv1_stride,
                         pool1_stride=pool1_stride, spatial_strides=spatial_strides,
                         temporal_strides=temporal_strides, **kwargs)
        self.avg_pool = torch.nn.AvgPool3d((16,7,7))

    def forward(self, x):
        batch = x.shape[0]
        feature = self.restnet3d(x)
        out = self.avg_pool(feature)
        out = out.reshape(batch, 512)
        return out

    def load_weights(self):
        weight = torch.load('./models/pose_only.pth', map_location='cpu')
        load_dict = {k[9:]: v for k,v in weight['state_dict'].items() }
        load_dict.pop('conv1.conv.weight')
        load_dict.pop('fc_cls.bias')
        load_dict.pop('fc_cls.weight')
        self.restnet3d.load_state_dict(load_dict, strict=False)
        print('slowonly weights loaded')


if __name__=='__main__':
    x = torch.randn((2,3,16,56,56))
    model = ResNet3dSlowOnly()
    model.load_weights()
    y = model(x)
    print(y)



