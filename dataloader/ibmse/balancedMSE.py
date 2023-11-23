import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import joblib


class GAILoss(_Loss):
    def __init__(self, init_noise_sigma, gmm):
        super(GAILoss, self).__init__()
        self.gmm = joblib.load(gmm)
        self.gmm = {k: torch.tensor(self.gmm[k]).cuda() for k in self.gmm}
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = gai_loss(pred, target, self.gmm, noise_var)
        return loss


# def gai_loss(pred, target, gmm, noise_var):
#     gmm = {k: gmm[k].reshape(1, -1).expand(pred.shape[0], -1) for k in gmm}
#     mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var + 0.5 * noise_var.log()
#     sum_var = gmm['variances'] + noise_var
#     balancing_term = - 0.5 * sum_var.log() - 0.5 * (pred - gmm['means']).pow(2) / sum_var + gmm['weights'].log()
#     balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
#     loss = mse_term + balancing_term
#     loss = loss * (2 * noise_var).detach()

#     return loss.mean()


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var).detach()

    return loss


    
class FocalRLoss(_Loss):
    def __init__(self, weights=None, activate='sigmoid', beta=.2, gamma=1):
        super(FocalRLoss, self).__init__()
        self.weights = weights
        self.activate = activate
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        loss = weighted_focal_mse_loss(pred, target, self.weights, self.activate, self.beta, self.gamma)
        return loss
    
def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
    
if __name__=='__main__':
    loss = GAILoss(init_noise_sigma=1., gmm='./dataloader/gmm_PD_0.pkl')
    pred= torch.randn((2,1)).cuda()
    target = torch.randn((2,1)).cuda()
    p = loss(pred, target)
