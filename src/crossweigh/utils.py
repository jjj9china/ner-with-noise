# -*- coding:utf-8 -*-

import os
import random
import config


def is_postive_sent(labels):
    for label in labels:
        if label != 'O':
            return True
    return False


def merge_files(input_dir, output_file, sample_flag=True):
    ids = []
    path = 'D:\\myProject\\learning-with-noise\\test\\annoid_300.txt'
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            id = line.strip()
            ids.append(id)

    if os.path.exists(output_file):
        os.remove(output_file)
    assert not os.path.exists(output_file)

    sentences = []

    count = 0
    files_list = os.listdir(input_dir)
    for i in range(0, len(files_list)):

        # remove data in benchmark data set
        if files_list[i].split('.')[0] in ids:
            count += 1
            continue

        input_file = os.path.join(input_dir, files_list[i])
        with open(input_file, 'r', encoding='utf-8') as f:
            tokens = []
            labels = []
            for line in f.readlines() + ['']:
                if len(line) == 0 or line.isspace():
                    if len(tokens) > 1:
                        random_num = random.uniform(0, 1)

                        if sample_flag:
                            if not is_postive_sent(labels):
                                if random_num > config.SAMPLING_RATE:
                                    sentences.append((tokens, labels))
                            else:
                                sentences.append((tokens, labels))
                        else:
                            sentences.append((tokens, labels))

                    tokens = []
                    labels = []
                else:
                    splits = line.strip().split('\t')
                    token, label = '\t'.join(splits[:2]), splits[-1]
                    tokens.append(token)
                    labels.append(label)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for token, label in zip(*sentence):
                f.write(f'{token}\t{label}\n')
            f.write('\n')

    print(f'remove data count: {count}')


def load_data(path, schema='bio'):
    # type: (str, str) -> list
    """Loads a column dataset into list of (tokens, labels). assumes BIO(IOB2) labeling.

    Args:
        path: file path
        schema: file label schema

    Returns:
        List of sentence
    """
    with open(path, 'r', encoding='utf-8') as f:
        sentences = []
        tokens = []
        labels = []
        for line in f.readlines() + ['']:
            if len(line) == 0 or line.isspace():
                if config.MIN_SEQ_LEN <= len(tokens) <= config.MAX_SEQ_LEN:
                    if schema is not None and schema != 'none':
                        if schema == 'iob':
                            labels = iob2bio(labels)
                        elif schema == 'iobes':
                            labels = iobes2bio(labels)
                        elif schema == 'sio':
                            labels = sio2bio(labels)
                        validate_bio(labels)

                    sentences.append((tokens, labels))

                tokens = []
                labels = []
            else:
                splits = line.strip().split('\t')
                token, label = '\t'.join(splits[:-1]), splits[-1]
                tokens.append(token)
                labels.append(label)
    return sentences


def validate_bio(labels):
    """Make sure that current label is valid.
    """
    for cur_label, next_label in zip(labels, labels[1:] + ['O']):
        if cur_label[0] == 'O':
            assert next_label[0] == 'O' or next_label[0] == 'B'
            continue
        elif cur_label[0] == 'B':
            assert next_label[0] == 'O' or next_label[0] == 'B' or (
                    next_label[0] == 'I' and cur_label[1:] == next_label[1:])
        elif cur_label[0] == 'I':
            assert next_label[0] == 'O' or next_label[0] == 'B' or \
                   (next_label[0] == 'I' and cur_label[1:] == next_label[1:])
        else:
            assert False


def iob2bio(iob_labels):
    bio_labels = []
    for prev_label, cur_label in zip(['O'] + iob_labels[:-1], iob_labels):
        if (prev_label[0] == 'O' and cur_label[0] == 'I') or (prev_label[0] != 'O' and
                                                              cur_label[0] == 'I' and
                                                              prev_label[2:] != cur_label[2:]):
            bio_labels.append('B' + cur_label[1:])
        else:
            bio_labels.append(cur_label)
    return bio_labels


def iobes2bio(iobes_labels):
    bio_labels = []
    for label in iobes_labels:
        if label[0] == 'S':
            bio_labels.append('B' + label[1:])
        elif label[0] == 'E':
            bio_labels.append('I' + label[1:])
        else:
            bio_labels.append(label)
    return bio_labels


def sio2bio(sio_labels):
    bio_labels = []
    for label in sio_labels:
        if label[0] == 'S':
            bio_labels.append('B' + label[1:])
        else:
            bio_labels.append(label)
    return bio_labels


def prepare_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"make output folder {folder_path}.")
        os.makedirs(folder_path, exist_ok=True)


def save_data(path, data, indexs):
    with open(path, 'w', encoding='utf-8') as f:
        for x in indexs:
            for token, label in zip(*data[x]):
                f.write(f'{token}\t{label}\n')
            f.write('\n')
