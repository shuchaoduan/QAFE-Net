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

    def load_weights(self, ckpt_path, model_state):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {}
        for k, v in ckpt.items():
            for i in range(1,4):
                if 'layer{}.0.'.format(i) in k:
                    state_dict.update({k.replace('layer{}.0.'.format(i), 'layer{}.blocks.0.'.format(i)): v})
                    break
                elif 'layer{}.1.'.format(i) in k:
                    state_dict.update({k.replace('layer{}.1.'.format(i), 'layer{}.last_cnn.'.format(i)): v})
                    break
                else:
                    state_dict.update({k: v})
                    break

        p2 = {k: v for k, v in state_dict.items() if
              k in model_state and model_state[k].shape == state_dict[k].shape}
        self.load_state_dict(p2, strict=False)

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

def freeze_s_former(model):
    model.s_former.conv1.requires_grad = False
    model.s_former.bn1.requires_grad = False
    model.s_former.relu.requires_grad = False
    model.s_former.maxpool.requires_grad = False
    model.s_former.layer1.requires_grad = False
    model.s_former.layer2.requires_grad = False
    model.s_former.layer3.requires_grad = False

    return model
    # for name, p in model.named_parameters():
    #     if 's_former' in name:
    #         p.requires_grad = False


if __name__ == '__main__':
    img = torch.randn((2, 80, 3, 224, 224))
    model = GenerateModel(cls_num=5)
    model(img)
