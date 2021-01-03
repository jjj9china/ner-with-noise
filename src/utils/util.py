# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2020-12-30

import pickle
import torch
import logging

from preprocess import config as DataConfig


__all__ = ['save_predict', 'save_model', 'load_model', 'sort_by_lengths',
           'tensorized_word', 'tensorized_label', 'get_logger']


def save_predict(test_word_lists, test_tag_lists, pred_tag_lists, path):
    # 因为test_word_lists最后每一个句子都加入了一个<end>, 因此长一个单位
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(test_tag_lists)):
            tag_list = test_tag_lists[i]
            for j in range(len(tag_list)):
                f.write(test_word_lists[i][j] + '\t' + test_tag_lists[i][j] + '\t' + pred_tag_lists[i][j])
                f.write('\n')
            f.write('\n')
    f.close()


def save_model(model, file_name: str):
    """用于保存模型
    """
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name: str):
    """用于加载模型
    """
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


def sort_by_lengths(word_lists: list, tag_lists: list) -> tuple:
    """Sort by seq length in this batch. desc
    """
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def tensorized_word(batch: list, maps: dict) -> tuple:
    """Maps batch of words to ids.

    Args:
        batch: [B ,L]
        maps: map (word_alphabet)
    """
    PAD = maps.get(DataConfig.PAD_TOKEN)
    UNK = maps.get(DataConfig.UNKNOWN_TOKEN)

    max_len = len(batch[0])  # 已排序，所以第一个就是该batch的最大长度
    batch_size = len(batch)

    word_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            word_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return word_tensor, lengths


def tensorized_label(batch: list, maps: dict, partial_crf=False) -> tuple:
    """Maps batch of labels to ids.

    Args:
        batch: [B ,L]
        maps: map (label_alphabet)
        partial_crf: use partail crf decode or not.
    """
    PAD = maps.get(DataConfig.PAD_TOKEN)
    # UNK = -1 if partial_crf else maps.get(DataConfig.UNKNOWN_TOKEN)

    max_len = len(batch[0])  # 已排序，所以第一个就是该batch的最大长度
    batch_size = len(batch)
    num_tags = len(maps)

    if not partial_crf:
        label_tensor = torch.ones(batch_size, max_len).long() * PAD
        # size: [batch size, max seq length]
        for i, l in enumerate(batch):
            for j, e in enumerate(l):
                if e in maps:
                    label_tensor[i][j] = maps.get(e)
                else:
                    raise ValueError("get unknown tag: %s!" % e)
    else:
        label_tensor = torch.zeros(batch_size, max_len, num_tags, dtype=torch.uint8)
        # size: [batch size, max seq length, num tags]
        for i, l in enumerate(batch):
            for j, e in enumerate(l):
                if DataConfig.MULTI_LABEL_SPLIT_TAG not in e:
                    # unknown-tag or real-tag
                    if e in maps:
                        label_tensor[i, j, maps.get(e)] = 1
                    elif e == DataConfig.UNKNOWN_LABEL:
                        label_tensor[i, j, :] = 1
                    else:
                        raise ValueError("get unknown tag: %s!" % e)
                else:
                    # multi-tag
                    e_tokens = e.split(DataConfig.MULTI_LABEL_SPLIT_TAG)
                    for token in e_tokens:
                        if token in maps:
                            label_tensor[i][j][maps.get(token)] = 1
                        else:
                            raise ValueError("get unknown tag: %s!" % e)
            # fill pad token
            label_tensor[i, (j + 1):, PAD] = 1

    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return label_tensor, lengths


def get_logger(log_file: str):
    """Init logger"""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def create_possible_tag_masks(num_tags: int, tags: torch.Tensor) -> torch.Tensor:
    """Create possible tag bitmaps for partial-crf

    Args:
        num_tags: num of tags.
        tags: : [B, L]

    Returns:
        Boolean tensor, size is [batch, seq_length, num_tags],
        each location set true if it can be used as ground truth label.
    """
    copy_tags = tags.clone()
    no_annotation_idx = (copy_tags == DataConfig.UNLABELED_INDEX)
    copy_tags[copy_tags == DataConfig.UNLABELED_INDEX] = 0

    tags_ = torch.unsqueeze(copy_tags, 2)
    masks = torch.zeros(tags_.size(0), tags_.size(1), num_tags, dtype=torch.uint8, device=tags.device)
    masks.scatter_(2, tags_, 1)
    masks[no_annotation_idx] = 1
    return masks
