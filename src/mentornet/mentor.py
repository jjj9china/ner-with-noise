# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2021-01-13

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from preprocess import config as DataConfig


OUTPUT_DIM = 1  # 二分类问题，判断当前token是否为噪声


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, OUTPUT_DIM)
        self.dropout = nn.Dropout(0.2)

        self.best_model = None

    def forward(self, x, lengths):
        """Forward process of model.

        Args:
            x: batch中每个word的features, [B, L]
            lengths: 没有使用，为了和bilstm接口保持一致
        """
        # x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)   # [B, L, 1]
        return x

    def cal_loss(self, logits, targets, tag2id, pos_weight):
        """计算损失

        Args:
            logits: [B, L, out_size]
            targets: [B, L]
            tag2id: dict
            pos_weight: positive weight. Note that default positive class is normal one.
        """
        PAD = tag2id.get(DataConfig.PAD_TOKEN)
        mask = (targets != PAD)  # [B, L]
        targets = targets[mask]  # get real target
        logits = logits.squeeze(2).masked_select(mask).contiguous().view(-1)

        assert logits.size(0) == targets.size(0)
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)

        return loss


class BiLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """initialization

        Args:
            input_dim: 输入的维数
            hidden_dim：隐向量的维数
        """
        super(BiLstm, self).__init__()

        self.bilstm = nn.LSTM(input_dim, hidden_dim,
                              batch_first=True,
                              bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim, OUTPUT_DIM)

        self.best_model = None

    def forward(self, inputs, lengths: list):
        """Forward process of model.

        Args:
            inputs: batch中每个word的id, [B, L]
            lengths: 该batch中每个句子的长度, [B]
        """
        packed = pack_padded_sequence(inputs, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.linear(rnn_out)  # [B, L, 1]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口
        """
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids

    def cal_loss(self, logits, targets, tag2id, pos_weight):
        """计算 BiLSTM 损失

        Args:
            logits: [B, L, out_size]
            targets: [B, L]
            tag2id: dict,
            pos_weight: positive weight. Note that default positive class is normal one.
        """
        PAD = tag2id.get(DataConfig.PAD_TOKEN)
        assert PAD is not None

        mask = (targets != PAD)  # [B, L]
        targets = targets[mask]  # get real target
        logits = logits.squeeze(2).masked_select(mask).contiguous().view(-1)

        assert logits.size(0) == targets.size(0)
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)

        return loss
