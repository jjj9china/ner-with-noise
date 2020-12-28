# -*- coding:utf-8 -*-

import sys
import os
import random
import json
import argparse

import utils
import config


def load_all_data(input_files, schema):
    # type: (str, str) -> list
    """Load all data from input_files with label schema.

    Args:
        input_files: file path or a folder.
        schema: label schema.

    Returns:
        List of all data.
    """
    all_data = []

    if os.path.isdir(input_files):
        print('input files is a folder.')
        files_list = os.listdir(input_files)
        for j in range(0, len(files_list)):
            input_file = os.path.join(input_files, files_list[j])
            if os.path.isfile(input_file):
                all_data.extend(utils.load_data(input_file, schema))

    elif os.path.isfile(input_files):
        print('input files is a file.')
        all_data.extend(utils.load_data(input_files, schema))

    else:
        print('wrong input files!')
        sys.exit(1)

    return all_data


def label_to_entity(tokens, labels):
    # type: (list, list) -> list
    """Given tokens, labels, extract list of spans of entities as (TYPE, START inc, END exc, SURFACE)

    Args:
        tokens: List of a sentence' tokens
        labels: List of a sentence' labels

    Returns:
        List of a sentence' entities if its label in `FILTER_LABEL`
    """
    assert len(tokens) == len(labels)

    word_tokens = [token.split('\t')[0] for token in tokens]
    entities = []
    cur_entity = {}

    for index, (label, next_label) in enumerate(zip(labels, labels[1:] + ['O'])):
        if label[0] == 'B' and label[2:] in config.FILTER_LABEL:
            # the begin idx in label, we assume that you use 'BIO' schema,
            # thus positive label is like 'B_' or 'B-' or 'I_' or 'I-'.
            cur_entity['type'] = label[2:]
            cur_entity['start'] = index

        if next_label[0] != 'I' and len(cur_entity) > 0:
            # if next_label is not start with 'I', it means that label is end.
            cur_entity['end'] = index + 1
            cur_entity['surface'] = config.SENTENCE_SEP.join(word_tokens[cur_entity['start']: cur_entity['end']])
            entities.append(cur_entity)
            cur_entity = {}

    return entities


def create_kfolds(sentence_entities, folds, random_seed):
    # type: (list, int, int) -> dict
    """Create k folds. train and dev data have no overlap entities.

    Args:
        sentence_entities: [[e1, e2, e3], [e2, e4, e5], [e1, e3], ...]...
        folds:  int
        random_seed:  int

    Returns:
        Dict of split k-folds information.
    """
    random.seed(random_seed)
    data_size = len(sentence_entities)
    index = list(range(data_size))
    info = {'seed': random_seed, 'folds': folds, 'index': index}
    random.shuffle(index)
    for i in range(folds):
        dev_data_index = index[i::folds]
        train_data_index = [index[x::folds] for x in range(folds) if x != i]
        train_data_index = [x for y in train_data_index for x in y]
        forbid_entities = set().union(*[set(sentence_entities[x]) for x in dev_data_index])
        train_data_index = list(
            filter(lambda x: set(sentence_entities[x]).isdisjoint(forbid_entities), train_data_index))
        assert set(dev_data_index).isdisjoint(set(train_data_index))
        assert set().union(*[set(sentence_entities[x]) for x in dev_data_index]).isdisjoint(
            set().union(*[set(sentence_entities[x]) for x in train_data_index]))
        _info = {
            'train_index': train_data_index,
            'dev_index': dev_data_index,
            'train_sentences': len(train_data_index),
            'train_total_entities': sum(len(sentence_entities[x]) for x in train_data_index),
            'train_distinct_entities': len(set().union(*[set(sentence_entities[x]) for x in train_data_index])),
            'dev_sentences': len(dev_data_index),
            'dev_total_entities': sum(len(sentence_entities[x]) for x in dev_data_index),
            'dev_distinct_entities': len(set().union(*[set(sentence_entities[x]) for x in dev_data_index])),
        }
        info[f'fold-{i}'] = _info
        print(f"Set {i}")
        print(f"Train sentences: {_info['train_sentences']}")
        print(f"Train total entities: {_info['train_total_entities']}")
        print(f"Train distinct entities: {_info['train_distinct_entities']}")
        print(f"Dev sentences: {_info['dev_sentences']}")
        print(f"Dev total entities: {_info['dev_total_entities']}")
        print(f"Dev distinct entities: {_info['dev_distinct_entities']}")
    return info


def main(input_files, output_folder, folds, schema):
    """Split k-folds data, train-fold and dev-fold have no overlap entities.

    Args:
        input_files: origin input file; file path of folder path.
        output_folder: output folder use to save k-folds data.
        folds: the number of k-folds.
        schema: label schema used in input files.

    Returns:
        None
    """

    assert folds > 1
    utils.prepare_folder(output_folder)

    all_data = load_all_data(input_files, schema)
    sentence_entities = [list(map(lambda x: x['surface'],
                                  label_to_entity(tokens, labels))) for tokens, labels in all_data]

    seed = random.randint(111111, 999999)
    info = create_kfolds(sentence_entities, folds, seed)

    for i in range(folds):
        trainIndex = info[f'fold-{i}']['train_index']
        devIndex = info[f'fold-{i}']['dev_index']

        utils.prepare_folder(os.path.join(output_folder, f'fold-{i}'))

        # save train data
        utils.save_data(os.path.join(output_folder, f'fold-{i}', f'train.col'), all_data, trainIndex)
        utils.save_data(os.path.join(output_folder, f'fold-{i}', f'dev.col'), all_data, devIndex)

    output_folder_json = os.path.join(output_folder, 'info.json')
    if os.path.exists(output_folder_json):
        os.remove(output_folder_json)
    with open(output_folder_json, 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split input_file into output_folder.')
    parser.add_argument('--input_files', help='input path, files or folder', default='../../data/train.col', nargs='+')

    parser.add_argument('--output_folder', help='output folder, will create per-fold folder in itr',
                        default='../../data/')

    parser.add_argument('--splits', help='number of splits to make', type=int, default=5)

    parser.add_argument('--folds', help='number of folds to make', type=int, default=5)

    parser.add_argument('--schema', help='label typing schema', default="sio",
                        choices=["sio", "bio", "iob", "iobes", "none"])
    args = parser.parse_args()
    # print(vars(args))

    for i in range(args.splits):
        output_folder = args.output_folder + f'split-{i}/'
        main(args.input_files, output_folder, args.folds, args.schema)

