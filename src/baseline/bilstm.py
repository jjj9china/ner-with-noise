# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2020-12-29

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from preprocess import config as DataConfig


class BiLstm(nn.Module):
    def __init__(self, vocab_size, out_size, emb_size=100, hidden_size=128, pretrain_word_embedding=None):
        """initialization

        Args:
            vocab_size: 字典的大小
            emb_size: 词向量的维数
            hidden_size：隐向量的维数
            out_size: 标注的种类大小
        """
        super(BiLstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        if pretrain_word_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrain_word_embedding))
        else:
            self.embedding.weight.data.copy_(torch.from_numpy(self._random_embedding(vocab_size, emb_size)))

        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.linear = nn.Linear(2*hidden_size, out_size)

        self.best_model = None

    def forward(self, sents_tensor, lengths: list):
        """Forward process of model.

        Args:
            sents_tensor: batch中每个word的id, [B, L]
            lengths: 该batch中每个句子的长度, [B]
        """
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        scores = self.linear(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids

    def cal_loss(self, logits, targets, tag2id):
        """计算BiLSTM损失
        参数:
            logits: [B, L, out_size]
            targets: [B, L]
            lengths: [B]
        """
        PAD = tag2id.get(DataConfig.PAD_TOKEN)
        assert PAD is not None

        mask = (targets != PAD)  # [B, L]
        targets = targets[mask]  # get real target
        out_size = logits.size(2)
        logits = logits.masked_select(mask.unsqueeze(2).expand(-1, -1, out_size)).contiguous().view(-1, out_size)
        # 展开后的logits [-1, out_size]

        assert logits.size(0) == targets.size(0)
        loss = F.cross_entropy(logits, targets)

        return loss

    """=========bilstm tools========="""
    def _random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

