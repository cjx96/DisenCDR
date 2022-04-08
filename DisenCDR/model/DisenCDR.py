import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.singleVBGE import singleVBGE
from model.crossVBGE import crossVBGE
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class DisenCDR(nn.Module):
    def __init__(self, opt):
        super(DisenCDR, self).__init__()
        self.opt=opt

        self.source_specific_GNN = singleVBGE(opt)
        self.source_share_GNN = singleVBGE(opt)

        self.target_specific_GNN = singleVBGE(opt)
        self.target_share_GNN = singleVBGE(opt)

        self.share_GNN = crossVBGE(opt)

        self.dropout = opt["dropout"]

        # self.user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])

        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])
        self.source_user_embedding_share = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding_share = nn.Embedding(opt["target_user_num"], opt["feature_dim"])

        self.share_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.share_sigma = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)

        if self.opt["cuda"]:
            self.user_index = self.user_index.cuda()
            self.source_user_index = self.source_user_index.cuda()
            self.target_user_index = self.target_user_index.cuda()
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()

    def source_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.source_predict_1(fea)
        out = F.relu(out)
        out = self.source_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def target_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.target_predict_1(fea)
        out = F.relu(out)
        out = self.target_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output


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
        if self.share_mean.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, (1 - self.opt["beta"]) * kld_loss

    def forward(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        source_user_share = self.source_user_embedding_share(self.source_user_index)
        target_user_share = self.target_user_embedding_share(self.target_user_index)

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item, source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item, target_UV, target_VU)

        source_user_mean, source_user_sigma = self.source_share_GNN.forward_user_share(source_user, source_UV, source_VU)
        target_user_mean, target_user_sigma = self.target_share_GNN.forward_user_share(target_user, target_UV, target_VU)

        mean, sigma, = self.share_GNN(source_user_share, target_user_share, source_UV, source_VU, target_UV, target_VU)

        user_share, share_kld_loss = self.reparameters(mean, sigma)

        source_share_kld = self._kld_gauss(mean, sigma, source_user_mean, source_user_sigma)
        target_share_kld = self._kld_gauss(mean, sigma, target_user_mean, target_user_sigma)

        self.kld_loss =  share_kld_loss + self.opt["beta"] * source_share_kld + self.opt[
            "beta"] * target_share_kld

        # source_learn_user = self.source_merge(torch.cat((user_share, source_learn_specific_user), dim = -1))
        # target_learn_user = self.target_merge(torch.cat((user_share, target_learn_specific_user), dim = -1))
        source_learn_user = user_share + source_learn_specific_user
        target_learn_user = user_share + target_learn_specific_user

        return source_learn_user, source_learn_specific_item, target_learn_user, target_learn_specific_item

    def wramup(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item,
                                                                                          source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item,
                                                                                          target_UV, target_VU)
        self.kld_loss = 0
        return source_learn_specific_user, source_learn_specific_item, target_learn_specific_user, target_learn_specific_item
