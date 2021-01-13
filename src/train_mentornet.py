# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2020-01-11

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from sklearn.model_selection import train_test_split

from mentornet import config as ModelConfig
from mentornet.student import BiLstmPartialCrf as studentModel
from mentornet.mentor import MLP as mentorModel
from preprocess.data import Data
from preprocess import config as DataConfig
from preprocess.util import prepocess_data_for_lstmcrf, load_mentor_data
from utils.metric import Metrics
from utils.util import *


def model_train(student, mentor, train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists,
                train_mentor_data, dev_mentor_data, word2id, tag2id, device, logger):
    # 对数据集按照长度进行排序
    train_word_lists, train_tag_lists, _ = sort_by_lengths(train_word_lists, train_tag_lists)
    dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

    # 整理mentor数据
    train_mentor_word = [data[0] for data in train_mentor_data]
    train_mentor_s_label = [data[1] for data in train_mentor_data]
    train_mentor_m_label = [data[2] for data in train_mentor_data]
    train_mentor_word, train_mentor_m_label, _ = sort_by_lengths(train_mentor_word, train_mentor_m_label)

    dev_mentor_word = [data[0] for data in dev_mentor_data]
    dev_mentor_s_label = [data[1] for data in dev_mentor_data]
    dev_mentor_m_label = [data[2] for data in dev_mentor_data]
    dev_mentor_word, dev_mentor_m_label, _ = sort_by_lengths(dev_mentor_word, dev_mentor_m_label)

    # 加载训练参数
    student_B = ModelConfig.student_batch_size
    mentor_B = ModelConfig.mentor_batch_size
    best_val_loss = ModelConfig.best_val_loss
    epoches = ModelConfig.epoches
    print_step = ModelConfig.print_step
    lr = ModelConfig.lr
    early_stop = ModelConfig.early_stop

    s_optimizer = optim.Adam(student.parameters(), lr=lr)
    m_optimizer = optim.Adam(mentor.parameters(), lr=lr)

    nepoch_no_iprv = 0
    for e in range(1, epoches + 1):
        student.train()
        mentor.train()

        # -----------------
        # 1. Train student
        # -----------------
        s_step = 0
        s_losses = 0.
        for ind in range(0, len(train_word_lists), student_B):

            batch_sents = train_word_lists[ind:ind + student_B]
            batch_tags = train_tag_lists[ind:ind + student_B]
            tensorized_sents, lengths = tensorized_word(batch_sents, word2id)  # [batch, max_length]
            tensorized_sents = tensorized_sents.to(device)
            tensorized_targets, lengths = tensorized_label(batch_tags, tag2id)
            tensorized_targets = tensorized_targets.to(device)

            s_optimizer.zero_grad()
            scores = student(tensorized_sents, lengths)
            features = student.features
            v_predict = F.sigmoid(mentor(features).detach())  # 切断mentor的梯度回传
            tag_bitmap = get_tag_bitmap(tensorized_sents, tensorized_targets, v_predict, tag2id)
            tag_bitmap = tag_bitmap.to(device)

            s_loss = student.cal_loss(scores, tag_bitmap).to(device)
            s_loss.backward()
            s_optimizer.step()

            s_losses += s_loss.item()
            s_step += 1

            if s_step % print_step == 0:
                total_step = (len(train_word_lists) // student_B + 1)
                logger.info("Epoch {}, s_step/total_step: {}/{} {:.2f}% s_loss:{:.4f}".format(
                    e, s_step, total_step, 100. * s_step / total_step, s_losses / print_step))
                s_losses = 0.

        # ----------------
        # 2. Train mentor
        # ----------------
        m_step = 0
        m_losses = 0.
        for ind in range(0, len(train_mentor_word), mentor_B):

            batch_sents = train_mentor_word[ind:ind + mentor_B]
            # batch_s_tags = train_mentor_s_label[ind:ind + mentor_B]
            batch_m_tags = train_mentor_m_label[ind:ind + mentor_B]
            tensorized_sents, lengths = tensorized_word(batch_sents, word2id)  # [batch, max_length]
            tensorized_sents = tensorized_sents.to(device)
            # tensorized_s_targets, lengths = tensorized_label(batch_s_tags, tag2id)
            # tensorized_s_targets = tensorized_s_targets.to(device)
            tensorized_m_targets, lengths = tensorized_label(batch_m_tags, {'0': 0, '1': 1, DataConfig.PAD_TOKEN: 2})
            tensorized_m_targets = tensorized_m_targets.to(device)

            m_optimizer.zero_grad()
            _ = student(tensorized_sents, lengths)
            features = student.features.detach()  # 切断student的梯度回传
            v_predict = mentor(features)

            PAD = 2
            mask = (tensorized_m_targets != PAD)  # [B, L]
            tensorized_m_targets = tensorized_m_targets[mask]  # get real target
            v_predict = v_predict.squeeze(2).masked_select(mask).contiguous().view(-1)
            assert v_predict.size(0) == tensorized_m_targets.size(0)
            pos_weight = torch.tensor([0.5], dtype=torch.float32, device=device)
            m_loss = F.binary_cross_entropy_with_logits(v_predict, tensorized_m_targets.float(), pos_weight=pos_weight)

            m_loss.backward()
            m_optimizer.step()

            m_losses += m_loss.item()
            m_step += 1

            if m_step % print_step == 0:
                total_step = (len(train_mentor_word) // mentor_B + 1)
                logger.info("Epoch {}, m_step/total_step: {}/{} {:.2f}% m_loss:{:.4f}".format(
                    e, m_step, total_step, 100. * m_step / total_step, m_losses / print_step))
                m_losses = 0.

        # 查看mentor在验证集上的效果
        logger.info("Epoch {}, Evaluate Mentor in Dev.".format(e))
        mentor_label, mentor_predict = mentor_validate(student, mentor, dev_mentor_word, dev_mentor_m_label,
                                                       word2id, tag2id, mentor_B, device)
        metrics = Metrics(mentor_label, mentor_predict, logger)
        metrics.report_scores()
        # metrics.report_confusion_matrix()

        # 每轮结束测试在验证集上的性能，保存最好的一个
        val_loss = validate(student, mentor, dev_word_lists, dev_tag_lists, word2id, tag2id, student_B, device)
        logger.info("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

        if val_loss < best_val_loss:
            logger.info("Get best student model.")
            student.best_model = deepcopy(student)
            best_val_loss = val_loss
        else:
            nepoch_no_iprv = nepoch_no_iprv + 1
            if nepoch_no_iprv == early_stop:
                logger.info("meet early stop condition={}. Stop Training!".format(early_stop))
                break


def validate(student, mentor, dev_word_lists, dev_tag_lists, word2id, tag2id, batch_size, device):
    student.eval()
    mentor.eval()
    with torch.no_grad():
        val_losses = 0.
        val_step = 0
        for ind in range(0, len(dev_word_lists), batch_size):
            val_step += 1
            # 准备batch数据
            batch_sents = dev_word_lists[ind:ind + batch_size]
            batch_tags = dev_tag_lists[ind:ind + batch_size]
            tensorized_sents, lengths = tensorized_word(batch_sents, word2id)
            tensorized_sents = tensorized_sents.to(device)
            tensorized_targets, lengths = tensorized_label(batch_tags, tag2id)
            tensorized_targets = tensorized_targets.to(device)

            # forward
            scores = student(tensorized_sents, lengths)
            features = student.features
            v_predict = F.sigmoid(mentor(features))
            tag_bitmap = get_tag_bitmap(tensorized_sents, tensorized_targets, v_predict, tag2id)
            tag_bitmap = tag_bitmap.to(device)

            s_loss = student.cal_loss(scores, tag_bitmap).to(device)

            val_losses += s_loss.item()
        val_loss = val_losses / val_step

        return val_loss


def mentor_validate(student, mentor, dev_mentor_word, dev_mentor_m_label, word2id, tag2id, mentor_B, device):
    student.eval()
    mentor.eval()

    mentor_predict = []
    mentor_label = []
    with torch.no_grad():
        for ind in range(0, len(dev_mentor_word), mentor_B):
            # 准备batch数据
            batch_sents = dev_mentor_word[ind:ind + mentor_B]
            batch_m_tags = dev_mentor_m_label[ind:ind + mentor_B]

            tensorized_sents, lengths = tensorized_word(batch_sents, word2id)  # [batch, max_length]
            tensorized_sents = tensorized_sents.to(device)
            tensorized_m_targets, lengths = tensorized_label(batch_m_tags, {'0': 0, '1': 1, DataConfig.PAD_TOKEN: 2})
            tensorized_m_targets = tensorized_m_targets.to(device)

            # forward
            _ = student(tensorized_sents, lengths)
            features = student.features
            v_predict = mentor(features)

            PAD = 2
            mask = (tensorized_m_targets != PAD)  # [B, L]
            tensorized_m_targets = tensorized_m_targets[mask].tolist()  # get real target
            v_predict = v_predict.squeeze(2).masked_select(mask).contiguous().view(-1).tolist()

            mentor_label.append([str(l) for l in tensorized_m_targets])
            mentor_predict.append(['0' if v < ModelConfig.predict_threshold else '1' for v in v_predict])

    return mentor_label, mentor_predict


def student_test(student, word_lists, tag_lists, word2id, tag2id, device):
    """返回最佳模型在测试集上的预测结果"""
    # 准备数据
    word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
    tensorized_sents, lengths = tensorized_word(word_lists, word2id)
    tensorized_sents = tensorized_sents.to(device)

    best_model = student.best_model
    best_model.eval()
    with torch.no_grad():
        batch_tagids = best_model.test(tensorized_sents, lengths)

    # 将id转化为标注
    pred_tag_lists = []
    id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
    for i, ids in enumerate(batch_tagids):
        tag_list = []
        for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
            tag_list.append(id2tag[ids[j].item()])

        pred_tag_lists.append(tag_list)

    # indices存有根据长度排序后的索引映射的信息
    # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
    # 索引为2的元素映射到新的索引是1...
    # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
    ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
    indices, _ = list(zip(*ind_maps))
    pred_tag_lists = [pred_tag_lists[i] for i in indices]
    tag_lists = [tag_lists[i] for i in indices]

    return pred_tag_lists, tag_lists


def train(train_file, dev_file, test_file, mentor_file, gaz_file, model_save_path, output_path, log_path):
    """Train model in train-data, evaluate it in dev-data, and finally test it in test-data

    Args:
        train_file:
        dev_file:
        test_file:
        mentor_file:
        gaz_file: Word vector file
        model_save_path:
        output_path: Predict file path.
        log_path:
    """
    current_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logger = get_logger(os.path.join(log_path, current_time + '-mentornet.log'))

    # ======== 1. 数据准备 ========
    data = Data(logger)
    data.initialization([train_file, dev_file, test_file], gaz_file)
    logger.info("Data initialization.")
    data.generate_data_instance(train_file, 'train')
    data.generate_data_instance(dev_file, 'dev')
    data.generate_data_instance(test_file, 'test')
    logger.info("Generate train/dev/test data.")

    train_texts, dev_texts, test_texts = data.train_texts, data.dev_texts, data.test_texts

    train_word_lists = [text[0] for text in train_texts]
    train_tag_lists = [text[-1] for text in train_texts]
    dev_word_lists = [text[0] for text in dev_texts]
    dev_tag_lists = [text[-1] for text in dev_texts]
    test_word_lists = [text[0] for text in test_texts]
    test_tag_lists = [text[-1] for text in test_texts]

    # crf特殊处理
    word2id, tag2id = data.extend_maps_for_crf()
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(train_word_lists, train_tag_lists)
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(dev_word_lists, dev_tag_lists)
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)

    # prepare word embdding
    data.build_pretrain_emb(gaz_file, 'word')

    # load mentor data
    mentor_data = load_mentor_data(mentor_file)
    train_mentor_data, dev_mentor_data = train_test_split(mentor_data, train_size=0.7, random_state=2021)

    # ======== 2. 模型构建 ========
    logger.info("Build mentorNet model.")
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = studentModel(vocab_size=vocab_size, out_size=out_size,
                           emb_size=ModelConfig.embedding_size, hidden_size=ModelConfig.hidden_size,
                           pretrain_word_embedding=data.pretrain_word_embedding).to(device)
    mentor = mentorModel(input_dim=2*ModelConfig.hidden_size).to(device)

    # ======== 3. 模型训练 ========
    logger.info("Training model.")
    model_train(student, mentor, train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists,
                train_mentor_data, dev_mentor_data, word2id, tag2id, device, logger)

    save_model(student, os.path.join(model_save_path, 'mentornet-model.pkl'))
    logger.info("train done, time consuming {} s.".format(int(time.time() - start)))

    # ======== 4. 模型评估 ========
    logger.info("Evaluating model.")
    pred_tag_lists, test_tag_lists = student_test(student, test_word_lists, test_tag_lists, word2id, tag2id, device)

    metrics = Metrics(test_tag_lists, pred_tag_lists, logger, remove_O=True)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    save_predict(test_word_lists, test_tag_lists, pred_tag_lists, os.path.join(output_path, 'mentornet-predict.col'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/train_demo.col')
    parser.add_argument('--dev_file', default='../data/dev_demo.col')
    parser.add_argument('--test_file', default='../data/test.col')
    parser.add_argument('--mentor_file', default='../data/train_mentor.col')
    parser.add_argument('--gaz_file', default='../data/wv_txt.txt')
    parser.add_argument('--model_save_path', default='../saved_model/')
    parser.add_argument('--output_path', default='../data/output/')
    parser.add_argument('--log_path', default='../log/')
    args = parser.parse_args()

    train(args.train_file, args.dev_file, args.test_file, args.mentor_file, args.gaz_file,
          args.model_save_path, args.output_path, args.log_path)

