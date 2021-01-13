# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2021-01-07

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

    def forward(self, x):
        # x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)   # [B, L, 1]
        return x


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

    def cal_loss(self, logits, targets, tag2id):
        """计算 BiLSTM 损失

        Args:
            logits: [B, L, out_size]
            targets: [B, L]
            tag2id: dict
        """
        PAD = tag2id.get(DataConfig.PAD_TOKEN)
        assert PAD is not None

        mask = (targets != PAD)  # [B, L]
        targets = targets[mask]  # get real target
        out_size = logits.size(2)
        logits = logits.masked_select(mask.unsqueeze(2).expand(-1, -1, out_size)
                                      ).contiguous().view(-1, out_size)
        # 展开后的logits [-1, out_size]

        assert logits.size(0) == targets.size(0)
        loss = F.cross_entropy(logits, targets)

        return loss