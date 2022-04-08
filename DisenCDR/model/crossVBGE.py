import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GCN import GCN
from model.GCN import VGAE
from torch.autograd import Variable
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class crossVBGE(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(crossVBGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer(opt))
        self.encoder.append(LastLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        learn_user_source = source_ufea
        learn_user_target = target_ufea
        for layer in self.encoder[:-1]:
            learn_user_source = F.dropout(learn_user_source, self.dropout, training=self.training)
            learn_user_target = F.dropout(learn_user_target, self.dropout, training=self.training)
            learn_user_source, learn_user_target = layer(learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj)

        mean, sigma, = self.encoder[-1](learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj)
        return mean, sigma

class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.source_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.source_rate = torch.tensor(self.opt["rate"]).view(-1)

        if self.opt["cuda"]:
            self.source_rate = self.source_rate.cuda()

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho = self.gc3(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho = self.gc4(target_User_ho, target_UV_adj)

        source_User = torch.cat((source_User_ho , source_ufea), dim=1)
        source_User = self.source_user_union(source_User)
        target_User = torch.cat((target_User_ho, target_ufea), dim=1)
        target_User = self.target_user_union(target_User)

        return self.source_rate * F.relu(source_User) +  (1 - self.source_rate) * F.relu(target_User), self.source_rate * F.relu(source_User) + (1 - self.source_rate) * F.relu(target_User)


class LastLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(LastLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.source_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.source_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.source_rate = torch.tensor(self.opt["rate"]).view(-1)

        if self.opt["cuda"]:
            self.source_rate = self.source_rate.cuda()


    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho_mean = self.gc3_mean(source_User_ho, source_UV_adj)
        source_User_ho_logstd = self.gc3_logstd(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho_mean = self.gc4_mean(target_User_ho, target_UV_adj)
        target_User_ho_logstd = self.gc4_logstd(target_User_ho, target_UV_adj)

        source_User_mean = torch.cat(
            (source_User_ho_mean, source_ufea), dim=1)
        source_User_mean = self.source_user_union_mean(source_User_mean)

        source_User_logstd = torch.cat((source_User_ho_logstd, source_ufea), dim=1)
        source_User_logstd = self.source_user_union_logstd(source_User_logstd)

        target_User_mean = torch.cat(
            (target_User_ho_mean, target_ufea), dim=1)
        target_User_mean = self.target_user_union_mean(target_User_mean)

        target_User_logstd = torch.cat(
            (target_User_ho_logstd, target_ufea),
            dim=1)
        target_User_logstd = self.target_user_union_logstd(target_User_logstd)

        return self.source_rate * source_User_mean + (1 - self.source_rate) * target_User_mean, self.source_rate * source_User_logstd + (1 - self.source_rate) * target_User_logstd


