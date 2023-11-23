import torch
from torch import nn
from models.S_Former import spatial_transformer
from models.T_Former import temporal_transformer


class GenerateModel(nn.Module):
    def __init__(self,cls_num=7):
        super().__init__()
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        # self.fc = nn.Linear(512, cls_num)


    def forward(self, x):
        n_batch, frames, _, _, _ = x.shape
        n_clips = int(frames/16)
        # split video sequence into n segments and pack them
        if frames>16:
            data_pack = torch.cat([x[:,i:i+16] for i in range(0, frames-1, 16)])
            out_s= self.s_former(data_pack)
        else:
            # out_s = self.s_former(x)
            out_s= self.s_former(x)# []
        out_t = self.t_former(out_s)
        return out_t


if __name__ == '__main__':
    img = torch.randn((2, 80, 3, 224, 224))
    model = GenerateModel(cls_num=5)
    model(img)
