# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2021-01-02

from typing import Optional

import torch
import torch.nn as nn
from itertools import zip_longest

from baseline.bilstm import BiLstm
from preprocess import config as DataConfig


class BiLstmPartialCrf(nn.Module):
    def __init__(self, vocab_size, out_size, emb_size=100, hidden_size=128, pretrain_word_embedding=None):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLstmPartialCrf, self).__init__()
        self.bilstm = BiLstm(vocab_size, out_size, emb_size, hidden_size, pretrain_word_embedding)

        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        # label_size = out_size - 1
        self.start_transitions = nn.Parameter(torch.randn(out_size))
        self.end_transitions = nn.Parameter(torch.randn(out_size))
        self.transitions = nn.Parameter(torch.ones(out_size, out_size) * 1 / out_size)

        self.best_model = None

    def forward(self, sents_tensor, lengths):
        # [B, L, out_size]
        emission = self.bilstm(sents_tensor, lengths)

        return emission

    def test(self, test_sents_tensor, lengths, mask: Optional[torch.ByteTensor] = None):
        """使用维特比算法进行解码
        Parameters:
            test_sents_tensor:
            lengths:
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            tags: (batch_size)
        """
        scores = self.forward(test_sents_tensor, lengths)
        emission = scores
        batch_size, sequence_length, _ = emission.shape
        if mask is None:
            mask = torch.ones([batch_size, sequence_length], dtype=torch.uint8, device=emission.device)

        emissions = emission.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        # Start transition and first emission score
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, sequence_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Add end transition score
        score += self.end_transitions

        # Compute the best path
        seq_ends = mask.long().sum(dim=0) - 1

        best_tags_list = []
        for i in range(batch_size):
            _, best_last_tag = score[i].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[i]]):
                best_last_tag = hist[i][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        best_tags_list = torch.Tensor(best_tags_list).long()

        return best_tags_list

    def cal_loss(self, scores, targets):
        """计算BiLSTM-Partial-CRF模型的损失

        Args:
            scores : list of [B, L, T], [T], [T, T], [T]
            targets : [B, L]
        """
        emission = scores
        mask = torch.ones(targets.size(0), targets.size(1), dtype=torch.uint8, device=targets.device)
        possible_tags = targets.clone()
        gold_score = self._numerator_score(emission, targets, mask, possible_tags)
        forward_score = self._denominator_score(emission, mask)
        return torch.sum(forward_score - gold_score)

    """=========bilstm-partail-crf tools========="""

    def _denominator_score(self, emissions: torch.Tensor,
                           mask: torch.ByteTensor) -> torch.Tensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            mask: Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """
        batch_size, sequence_length, num_tags = emissions.data.shape

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        # Start transition score and first emissions score
        alpha = self.start_transitions.view(1, num_tags) + emissions[0]

        for i in range(1, sequence_length):
            emissions_score = emissions[i].view(batch_size, 1, num_tags)  # (batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)  # (1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)  # (batch_size, num_tags, 1)

            inner = broadcast_alpha + emissions_score + transition_scores  # (batch_size, num_tags, num_tags)

            alpha = (self._log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        stops = alpha + self.end_transitions.view(1, num_tags)

        return self._log_sum_exp(stops)  # (batch_size,)

    def _numerator_score(self, emissions: torch.Tensor,
                         tags: torch.LongTensor,
                         mask: torch.ByteTensor,
                         possible_tags: torch.ByteTensor) -> torch.Tensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            tags:  (batch_size, sequence_length)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """

        batch_size, sequence_length, num_tags = emissions.data.shape

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        possible_tags = possible_tags.float().transpose(0, 1)

        # Start transition score and first emission
        first_possible_tag = possible_tags[0]

        alpha = self.start_transitions + emissions[0]  # (batch_size, num_tags)
        alpha[(first_possible_tag == 0)] = DataConfig.IMPOSSIBLE_SCORE

        for i in range(1, sequence_length):
            current_possible_tags = possible_tags[i - 1]  # (batch_size, num_tags)
            next_possible_tags = possible_tags[i]  # (batch_size, num_tags)

            # Emissions scores
            emissions_score = emissions[i]
            emissions_score[(next_possible_tags == 0)] = DataConfig.IMPOSSIBLE_SCORE
            emissions_score = emissions_score.view(batch_size, 1, num_tags)

            # Transition scores
            transition_scores = self.transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags,
                                                                                    num_tags).clone()
            transition_scores[(current_possible_tags == 0)] = DataConfig.IMPOSSIBLE_SCORE
            transition_scores.transpose(1, 2)[(next_possible_tags == 0)] = DataConfig.IMPOSSIBLE_SCORE

            # Broadcast alpha
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all scores
            inner = broadcast_alpha + emissions_score + transition_scores  # (batch_size, num_tags, num_tags)
            alpha = (self._log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        last_tag_indexes = mask.sum(0).long() - 1
        end_transitions = self.end_transitions.expand(batch_size, num_tags) \
                          * possible_tags.transpose(0, 1).view(sequence_length * batch_size, num_tags)[
                              last_tag_indexes + torch.arange(batch_size,
                                                              device=possible_tags.device) * sequence_length]
        end_transitions[(end_transitions == 0)] = DataConfig.IMPOSSIBLE_SCORE
        stops = alpha + end_transitions

        return self._log_sum_exp(stops)  # (batch_size,)

    def _log_sum_exp(self, tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        """Compute log sum exp in a numerically stable way for the forward algorithm
        """
        max_score, _ = tensor.max(dim, keepdim=keepdim)
        if keepdim:
            stable_vec = tensor - max_score
        else:
            stable_vec = tensor - max_score.unsqueeze(dim)
        return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()
