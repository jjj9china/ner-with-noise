# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2021-01-03

import numpy as np
from preprocess import config as DataConfig

"""
This module contains functions about data pre-process utils.
"""

__all__ = ['get_masked_word', 'read_instance', 'generate_instance_with_gaz', 'build_pretrain_embedding',
           'norm2one', 'load_pretrain_emb', 'invalid_word', 'prepocess_data_for_lstmcrf', 
           'load_mentor_data']


def get_masked_word(word):
    """
    turn digit in word to special character '0'
    :param word:
    :return:
    """
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized, max_sent_length,
                  char_padding_size=-1):
    in_lines = open(input_file, 'r').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = get_masked_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [DataConfig.UNKNOWN_TOKEN] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instence_texts.append([words, chars, labels])
                instence_Ids.append([word_Ids, char_Ids, label_Ids])
            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def generate_instance_with_gaz(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet,
                               label_alphabet, char_padding_size=-1):
    """Generate instence_texts and instence_Ids.

    Args:
        input_file: Input file path.
        gaz: Gazetteer from word vector.
        word_alphabet: word alphabet from train/dev/test file.
        biword_alphabet: as above.
        char_alphabet: as above.
        label_alphabet: as above.
        gaz_alphabet: matched word alphabet from gaz according to train/dev/test file.
        char_padding_size: if char length is smaller than this value, padding it with char_padding_symbol

    Returns:
        instence texts: [words, biwords, chars, gazs, labels]
        instence ids: [word_ids, biword_ids, char_ids, gaz_ids, label_ids], gaz_ids=[id, length]
    """
    in_lines = open(input_file, 'r', encoding='utf-8').readlines()
    instence_texts = []
    instence_ids = []

    words = []
    biwords = []
    chars = []
    labels = []
    word_ids = []
    biword_ids = []
    char_ids = []
    label_ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx].strip()
        if len(line) > 0:
            pairs = line.strip().split()
            word = pairs[DataConfig.word_col]
            if DataConfig.word_masked and invalid_word(word, pairs):
                word = DataConfig.MASK_TOKEN
            label = pairs[DataConfig.label_col]

            if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                # assert current line is not the last line
                biword = word + in_lines[idx + 1].strip().split()[0]
            else:
                biword = word + DataConfig.UNKNOWN_TOKEN

            biwords.append(biword)  # save biword for current word
            words.append(word)  # save word for current word
            labels.append(label)  # save label for current word

            word_ids.append(word_alphabet.get_index(word))  # save wordIds for current word
            biword_ids.append(biword_alphabet.get_index(biword))  # save biwordIds for current word
            if label == DataConfig.UNKNOWN_LABEL:
                label_ids.append(-1)
            elif DataConfig.MULTI_LABEL_SPLIT_TAG in label:
                label_tokens = label.split(DataConfig.MULTI_LABEL_SPLIT_TAG)
                label_ids.append([label_alphabet.get_index(token) for token in label_tokens])
            else:
                label_ids.append(label_alphabet.get_index(label))  # save labelIds for current word

            char_list = []
            char_id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [DataConfig.UNKNOWN_TOKEN] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                # not padding
                pass
            for char in char_list:
                char_id.append(char_alphabet.get_index(char))
            chars.append(char_list)  # save chars for current word
            char_ids.append(char_id)  # save charIds for current word

        else:
            if (DataConfig.MAX_SENTENCE_LENGTH < 0 or len(words) < DataConfig.MAX_SENTENCE_LENGTH) and len(words) > 0:
                gazs = []
                gaz_ids = []
                w_length = len(words)
                for i in range(w_length):
                    matched_list = gaz.enumerate_match_list(words[i:])
                    matched_length = [len(a) for a in matched_list]
                    gazs.append(matched_list)
                    matched_id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_id is not None:
                        gaz_ids.append([matched_id, matched_length])  # gaz id and length
                    else:
                        gaz_ids.append([])

                instence_texts.append([words, biwords, chars, gazs, labels])
                instence_ids.append([word_ids, biword_ids, char_ids, gaz_ids, label_ids])

            # re-init
            words = []
            biwords = []
            chars = []
            labels = []
            word_ids = []
            biword_ids = []
            char_ids = []
            label_ids = []

    return instence_texts, instence_ids


def build_pretrain_embedding(embedding_path: str, word_alphabet, embedd_dim=100, norm=True, logger=None):
    """Build pretrain embedding file
    """
    embedd_dict = dict()
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])

    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    logger.info("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%.5s"
                % (pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / word_alphabet.size()))
    return pretrain_emb


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path: str):
    """Load pretrained embedding file

    Args:
        embedding_path: path

    Reutrns:
        Dict: {word: embedding}, embedd_dim
    """
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim


def invalid_word(word: str, pairs: list) -> bool:
    """Write your valid_word functions here, judge whether word should be masked or not.

    Return false if word should be masked.
    """
    is_digit = word.isdigit()

    entity = pairs[1]
    invalid_entity_map = {"arabic_number", "percent_number", "person_name", "datetime", "company_name",
                          "company_abbr_name", "serial_number", "chinese_number"}
    return entity in invalid_entity_map


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append(DataConfig.END_TOKEN)
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append(DataConfig.END_TOKEN)

    return word_lists, tag_lists


def load_mentor_data(path: str):
    """data loader for MenotNet.
    It at least has four columns: [word, noise-label, true-label, noise-or-not].
    And:
        noise-label: token origin label that might be noisy.
        true-label: corrected token label.
        noise-or-not: 0-1 label,
        `1` when `noise-label == true-label`, `0` when `noise-label != true-label`
    """
    in_lines = open(path, 'r', encoding='utf-8').readlines()

    sentences = []

    words = []
    s_labels = []  # origin noise-label (used for student)
    m_labels = []  # noise-or-not, 0-1 label (used for mentor)

    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[DataConfig.mentor_word_col]
            if DataConfig.word_masked and invalid_word(word, line.strip().split()):
                word = DataConfig.MASK_TOKEN
            words.append(word)
            s_labels.append(pairs[DataConfig.mentor_noise_label_col])
            m_labels.append(pairs[DataConfig.mentor_label_col])
        else:
            if (DataConfig.MAX_SENTENCE_LENGTH < 0 or len(words) < DataConfig.MAX_SENTENCE_LENGTH) and len(words) > 0:
                sentences.append([words, s_labels, m_labels])
            words = []
            s_labels = []
            m_labels = []

    return sentences

