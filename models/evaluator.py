import torch.nn as nn



class MLP_block(nn.Module):

    def __init__(self, output_dim):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU()
        # self.softmax = nn.Softmax(dim=-1)

        self.layer1 = nn.Linear(512, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        # output = self.softmax(self.layer3(x))
        output = self.layer3(x)
        return output
    
class single_fc(nn.Module):

    def __init__(self, output_dim):
        super(single_fc, self).__init__()
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        output = self.fc(x)
        return output

class two_fc(nn.Module):

    def __init__(self, output_dim):
        super(two_fc, self).__init__()
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(512, 256)
        self.layer2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        output = self.layer2(x)
        return output


class Evaluator(nn.Module):

    def __init__(self, output_dim, model_type='MLP'):
        super(Evaluator, self).__init__()

        self.model_type = model_type

        if model_type == 'MLP':
            self.evaluator = MLP_block(output_dim=output_dim)
        else: # classification
            self.evaluator = single_fc(output_dim=output_dim)


    def forward(self, feats_avg):  # data: NCTHW

        probs = self.evaluator(feats_avg)  # Nxoutput_dim

        return probs

