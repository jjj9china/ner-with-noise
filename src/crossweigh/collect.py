# -*- coding:utf-8 -*-

import argparse
import json
import os
import glob
from collections import defaultdict

from crossweigh import util
from crossweigh import config


def load_predict_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = []
        predicts = []
        labels = []
        for line in f.readlines() + ['']:
            if len(line) == 0 or line.isspace():
                if len(predicts) > 0:
                    sentences.append((predicts, labels))

                predicts = []
                labels = []
            else:
                splits = line.strip().split('\t')
                label, predict = splits[1], splits[2]
                predicts.append(predict)
                labels.append(label)
    return sentences


def load_from_splits(paths, model_predicted_filename, info_dict):
    """Compares `original dev label` with `model predicted label` under all paths, and merge the results.
    Paths could be like ['split-0/fold-0', 'split-0/fold-1', ..., 'split-k/fold-0'...]

    NOTE: We store all wrong predicts in Memory, so keep your eyes on the OOM problem.
    """
    sentence_potential_error_count = defaultdict(int)  # {index: error count}
    sentence_potential_error_detail = defaultdict(list)  # {index: [predict labels]}
    positive_sentence = 0
    negative_sentence = 0

    for path in paths:
        # =====load model predict file=====
        # you might should change this code-cell in your system
        model_predicted_path = os.path.join(path, model_predicted_filename)
        assert os.path.exists(model_predicted_path)
        split_index = path.split('\\')[1]
        fold_index = path.split('\\')[2]
        # =================================

        model_predicted = load_predict_file(model_predicted_path)
        for i in range(len(model_predicted)):
            predicts, labels = model_predicted[i]
            if ' '.join(predicts) != ' '.join(labels):
                # error count is calculated on the sample level.
                key = info_dict[split_index][fold_index]['dev_index'][i]
                sentence_potential_error_count[key] += 1
                sentence_potential_error_detail[key].append(predicts)
            if 'B' in ' '.join(labels):
                positive_sentence += 1
            else:
                negative_sentence += 1

    print(f'positive_sentence: {positive_sentence}, negative_sentence: {negative_sentence}')

    return sentence_potential_error_count, sentence_potential_error_detail


def data_filter(count):
    if count == 5:
        return True
    return False


def main(split_folders, split_info, splits, origin_files, origin_file_schema, output_folder,
         output_file, output_json, model_predicted_filename, eps):
    """Error count and data save.
    """
    if os.path.exists(os.path.join(output_folder, output_file)):
        os.remove(os.path.join(output_folder, output_file))
    if os.path.exists(os.path.join(output_folder, output_json)):
        os.remove(os.path.join(output_folder, output_json))

    # load origin data
    origin_data = util.load_data(origin_files, origin_file_schema)

    split_folder_list = glob.glob(split_folders)
    # load split info
    info_dict = {}
    for split_folder in split_folder_list:
        assert os.path.exists(split_folder)
        with open(os.path.join(split_folder, split_info), 'r') as f:
            info_dict['split-' + split_folder[-1]] = json.load(f)

    paths = []
    for split_folder in split_folder_list:
        paths.extend(glob.glob(os.path.join(split_folder, 'fold-*')))

    sentence_potential_error_count, sentence_potential_error_detail = load_from_splits(paths,
                                                                                       model_predicted_filename,
                                                                                       info_dict)

    # modify data label with UNKNOWN_TAG or MULTI_LABEL_SPLIT_TAG
    with open(os.path.join(output_folder, output_file), 'w', encoding='utf-8') as f:
        for i in range(len(origin_data)):

            """======== 1. save data after label correcting =========="""
            if i in sentence_potential_error_count:
                count = sentence_potential_error_count[i]
                save_count = 5
                if count == save_count:
                    # label correct
                    data = origin_data[i]  # [[word, ...], [label, ...]]
                    predicts = sentence_potential_error_detail[i]

                    for j in range(len(data[0])):
                        s = data[0][j] + '\t'

                        init_count = splits - len(predicts)
                        origin_label = data[1][j]
                        max_predict_label_count = init_count
                        max_predict_label = origin_label

                        predict_dict = defaultdict(int)
                        predict_dict[origin_label] = init_count  # original label and its count

                        for k in range(len(predicts)):
                            predict_dict[predicts[k][j]] += 1
                            if predict_dict[predicts[k][j]] > max_predict_label_count:
                                max_predict_label_count = predict_dict[predicts[k][j]]
                                max_predict_label = predicts[k][j]

                        if max_predict_label_count == save_count and max_predict_label != origin_label:
                            """==== 1. label correct with majority vote ======="""
                            # s = s + max_predict_label
                            """==== 2. multi label use origin and majority ===="""
                            # s = s + max_predict_label + config.MULTI_LABEL_SPLIT_TAG + origin_label
                            """==== 3. label correct with unknown tag ========="""
                            s = s + config.UNKNOWN_TAG
                        else:
                            s = s + origin_label
                        f.write(f'{s}\n')
                    f.write('\n')
                else:
                    for token, label in zip(*origin_data[i]):
                        f.write(f'{token}\t{label}\n')
                    f.write('\n')
            else:
                for token, label in zip(*origin_data[i]):
                    f.write(f'{token}\t{label}\n')
                f.write('\n')

            """========== 2. save data after data cleaning ==========="""
            # if i in sentence_potential_error_count and data_filter(sentence_potential_error_count[i]):
            #     continue
            # else:
            #     for token, label in zip(*origin_data[i]):
            #         f.write(f'{token}\t{label}\n')
            #     f.write('\n')

            """========== 3. save data have been cleaned ============="""
            # if i in sentence_potential_error_count and data_filter(sentence_potential_error_count[i]):
            #     data = origin_data[i]
            #     predict = sentence_potential_mistake_detail[i]
            #     for j in range(len(data[0])):
            #         s = data[0][j] + '\t' + origin_label
            #         for k in range(N):
            #             s = s + '\t' + predict[k][j]
            #         f.write(f'{s}\n')
            #     f.write('\n')

            """========== 4. save data after data reweighing ========="""
            # error_count = sentence_potential_error_count.get(i, 0)
            # weight = "%.2f" % (eps ** error_count)
            # for token, label in zip(*origin_data[i]):
            #     f.write(f'{token}\t{label}\t{weight}\n')
            # f.write('\n')

    # save error count information
    with open(os.path.join(output_folder, output_json), 'w', encoding='utf-8') as f:
        json.dump(sentence_potential_error_count, f, indent=2)


if __name__ == '__main__':
    """
    Before run this collect script, you should train models in /split-*/fold-*/ train data, 
    and predict in dev data. For convenience, you'd better save model predict files in the same folder.
    Predict file at least have three columns: words, origin-label, predict-label
    """
    parser = argparse.ArgumentParser(description="collect models' predicts data and save part of them")
    parser.add_argument('--split_folders', help='different split folders', default='../data/split-*', nargs='+')
    parser.add_argument('--split_info', default='info.json')
    parser.add_argument('--splits', help='number of splits to make', type=int, default=5)
    parser.add_argument('--folds', help='number of folds to make', type=int, default=5)
    parser.add_argument('--origin_files', default='../data/train.col')
    parser.add_argument('--origin_file_schema', default="sio", choices=["sio", "bio", "iob", "iobes", "none"])
    parser.add_argument('--output_folder', default='../data/')
    parser.add_argument('--output_file', default='unknown_label.col')
    parser.add_argument('--output_json', default='error_count_json.json')
    parser.add_argument('--model_predicted_filename', default='predict.col')
    parser.add_argument('--eps', help='clean data ration in data set', type=float, default=0.7)
    args = parser.parse_args()

    main(args.split_folders, args.split_info, args.splits, args.origin_files, args.origin_file_schema,
         args.output_folder, args.output_file, args.output_json, args.model_predicted_filename, args.eps)
