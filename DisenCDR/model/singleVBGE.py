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

class singleVBGE(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(singleVBGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer(opt))
        self.encoder.append(LastLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        learn_user = ufea
        learn_item = vfea
        for layer in self.encoder:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_item = F.dropout(learn_item, self.dropout, training=self.training)
            learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
        return learn_user, learn_item

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        learn_user = ufea
        for layer in self.encoder[:-1]:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_user = layer.forward_user_share(learn_user, UV_adj, VU_adj)
        mean, sigma = self.encoder[-1].forward_user_share(learn_user, UV_adj, VU_adj)
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
        self.user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        Item_ho = self.gc2(vfea, UV_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        Item = torch.cat((Item_ho, vfea), dim=1)
        User = self.user_union(User)
        Item = self.item_union(Item)
        return F.relu(User), F.relu(Item)

    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)

    def forward_item(self, ufea, vfea, UV_adj,VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        Item = torch.cat((Item_ho, vfea), dim=1)
        Item = self.item_union(Item)
        return F.relu(Item)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)


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
        self.user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

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

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        user, user_kld = self.forward_user(ufea, vfea, UV_adj,VU_adj)
        item, item_kld = self.forward_item(ufea, vfea, UV_adj, VU_adj)

        self.kld_loss = user_kld + item_kld

        return user, item


    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        user, kld_loss = self.reparameters(User_ho_mean, User_ho_logstd)
        return user, kld_loss

    def forward_item(self, ufea, vfea, UV_adj,VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)

        Item_ho_mean = self.gc4_mean(Item_ho, VU_adj)
        Item_ho_logstd = self.gc4_logstd(Item_ho, VU_adj)
        Item_ho_mean = torch.cat((Item_ho_mean, vfea), dim=1)
        Item_ho_mean = self.item_union_mean(Item_ho_mean)

        Item_ho_logstd = torch.cat((Item_ho_logstd, vfea), dim=1)
        Item_ho_logstd = self.item_union_logstd(Item_ho_logstd)

        item, kld_loss = self.reparameters(Item_ho_mean, Item_ho_logstd)
        return item, kld_loss

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        # user, kld_loss = self.reparameters(User_ho_mean, User_ho_logstd)
        return User_ho_mean, User_ho_logstd
        # return user, kld_loss