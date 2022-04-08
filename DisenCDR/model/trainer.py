import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.DisenCDR import DisenCDR
from utils import torch_utils

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class CrossTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        if self.opt["model"] == "DisenCDR":
            self.model = DisenCDR(opt)
        else :
            print("please input right model name!")
            exit(0)

        self.criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.epoch_rec_loss = []

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
        else:
            inputs = [Variable(b) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
        return user_index, item_index

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user = inputs[0]
            source_pos_item = inputs[1]
            source_neg_item = inputs[2]
            target_pos_item = inputs[3]
            target_neg_item = inputs[4]
        else:
            inputs = [Variable(b) for b in batch]
            user = inputs[0]
            source_pos_item = inputs[1]
            source_neg_item = inputs[2]
            target_pos_item = inputs[3]
            target_neg_item = inputs[4]
        return user, source_pos_item, source_neg_item, target_pos_item, target_neg_item

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def source_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        user_feature = self.my_index_select(self.source_user, user_index)
        item_feature = self.my_index_select(self.source_item, item_index)

        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        score = self.model.source_predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def target_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        user_feature = self.my_index_select(self.target_user, user_index)
        item_feature = self.my_index_select(self.target_item, item_index)

        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        score = self.model.target_predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def evaluate_embedding(self, source_UV, source_VU, target_UV, target_VU, source_adj=None, target_adj=None):
        self.source_user, self.source_item, self.target_user, self.target_item = self.model(source_UV, source_VU,
                                                                                            target_UV, target_VU)

    def for_bcelogit(self, x):
        y = 1 - x
        return torch.cat((x,y), dim = -1)

    def reconstruct_graph(self, batch, source_UV, source_VU, target_UV, target_VU, source_adj=None, target_adj=None, epoch = 100):
        self.model.train()
        self.optimizer.zero_grad()

        user, source_pos_item, source_neg_item, target_pos_item, target_neg_item = self.unpack_batch(batch)

        if epoch<10:
            self.source_user, self.source_item, self.target_user, self.target_item = self.model.wramup(source_UV,source_VU,target_UV,target_VU)
        else:
            self.source_user, self.source_item, self.target_user, self.target_item = self.model(source_UV,source_VU, target_UV,target_VU)

        source_user_feature = self.my_index_select(self.source_user, user)
        source_item_pos_feature = self.my_index_select(self.source_item, source_pos_item)
        source_item_neg_feature = self.my_index_select(self.source_item, source_neg_item)

        target_user_feature = self.my_index_select(self.target_user, user)
        target_item_pos_feature = self.my_index_select(self.target_item, target_pos_item)
        target_item_neg_feature = self.my_index_select(self.target_item, target_neg_item)


        pos_source_score = self.model.source_predict_dot(source_user_feature, source_item_pos_feature)
        neg_source_score = self.model.source_predict_dot(source_user_feature, source_item_neg_feature)
        pos_target_score = self.model.target_predict_dot(target_user_feature, target_item_pos_feature)
        neg_target_score = self.model.target_predict_dot(target_user_feature, target_item_neg_feature)

        pos_labels, neg_labels = torch.ones(pos_source_score.size()), torch.zeros(
            pos_source_score.size())

        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()


        loss = self.criterion(pos_source_score, pos_labels) + \
               self.criterion(neg_source_score, neg_labels) + \
               self.criterion(pos_target_score, pos_labels) + \
               self.criterion(neg_target_score, neg_labels) + \
               self.model.source_specific_GNN.encoder[-1].kld_loss + \
               self.model.target_specific_GNN.encoder[-1].kld_loss + self.model.kld_loss

        loss.backward()
        self.optimizer.step()
        return loss.item()