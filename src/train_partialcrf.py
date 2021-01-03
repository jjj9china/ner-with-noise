# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2020-12-30

import os
import time
import argparse
import torch
import torch.optim as optim
from copy import deepcopy


from baseline import config as ModelConfig
from partialcrf.bilstm_partialcrf import BiLstmPartialCrf
from preprocess.data import Data
from preprocess.util import prepocess_data_for_lstmcrf
from utils.metric import Metrics
from utils.util import *


def model_train(model, word_lists, tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id,
                device, logger):
    # 对数据集按照长度进行排序
    word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
    dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

    # 加载训练参数
    B = ModelConfig.batch_size
    best_val_loss = ModelConfig.best_val_loss
    epoches = ModelConfig.epoches
    print_step = ModelConfig.print_step
    lr = ModelConfig.lr
    early_stop = ModelConfig.early_stop

    optimizer = optim.Adam(model.parameters(), lr=lr)

    nepoch_no_iprv = 0
    for e in range(1, epoches + 1):
        step = 0
        losses = 0.
        for ind in range(0, len(word_lists), B):
            batch_sents = word_lists[ind:ind + B]
            batch_tags = tag_lists[ind:ind + B]

            losses += train_step(model, optimizer, batch_sents, batch_tags, word2id, tag2id, device)
            step += 1

            if step % print_step == 0:
                total_step = (len(word_lists) // B + 1)
                logger.info("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                    e, step, total_step, 100. * step / total_step, losses / print_step))
                losses = 0.

        # 每轮结束测试在验证集上的性能，保存最好的一个
        val_loss = validate(model, dev_word_lists, dev_tag_lists, word2id, tag2id, B, device)
        logger.info("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

        if val_loss < best_val_loss:
            logger.info("Get best model...")
            model.best_model = deepcopy(model)
            best_val_loss = val_loss
        else:
            nepoch_no_iprv = nepoch_no_iprv + 1
            if nepoch_no_iprv == early_stop:
                logger.info("meet early stop condition={}. Stop Training!".format(early_stop))
                break


def train_step(model, optimizer, batch_sents, batch_tags, word2id, tag2id, device):
    model.train()

    # 准备数据
    tensorized_sents, lengths = tensorized_word(batch_sents, word2id)  # [batch, max_length]
    tensorized_sents = tensorized_sents.to(device)
    targets, lengths = tensorized_label(batch_tags, tag2id, partial_crf=True)
    targets = targets.to(device)

    # forward
    # scores size in bilstm: [B, L, out_size]
    # scores size in  bilstm-crf: [B, L, out_size, out_size]
    scores = model(tensorized_sents, lengths)

    # 计算损失 更新参数
    optimizer.zero_grad()
    loss = model.cal_loss(scores, targets).to(device)
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(model, dev_word_lists, dev_tag_lists, word2id, tag2id, batch_size, device):
    model.eval()
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
            targets, lengths = tensorized_label(batch_tags, tag2id, partial_crf=True)
            targets = targets.to(device)

            # forward
            scores = model(tensorized_sents, lengths)

            # 计算损失
            loss = model.cal_loss(scores, targets).to(device)
            val_losses += loss.item()
        val_loss = val_losses / val_step

        return val_loss


def model_test(model, word_lists, tag_lists, word2id, tag2id, device):
    """返回最佳模型在测试集上的预测结果"""
    # 准备数据
    word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
    tensorized_sents, lengths = tensorized_word(word_lists, word2id)
    tensorized_sents = tensorized_sents.to(device)

    best_model = model.best_model
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


def train(train_file, dev_file, test_file, gaz_file, model_save_path, output_path, log_path):
    """Train model in train-data, evaluate it in dev-data, and finally test it in test-data

    Args:
        train_file:
        dev_file:
        test_file:
        gaz_file: Word vector file
        model_save_path:
        output_path: Predict file path.
        log_path:
    """
    current_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logger = get_logger(os.path.join(log_path, current_time + '-partialCrf.log'))

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

    # ======== 2. 模型构建 ========
    logger.info("Build partialCrf model.")
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    partial_crf_model = BiLstmPartialCrf(vocab_size=vocab_size, out_size=out_size,
                                         emb_size=ModelConfig.embedding_size, hidden_size=ModelConfig.hidden_size,
                                         pretrain_word_embedding=data.pretrain_word_embedding).to(device)

    # ======== 3. 模型训练 ========
    logger.info("Training model.")
    model_train(partial_crf_model, train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id,
                device, logger)

    save_model(partial_crf_model, os.path.join(model_save_path, 'partialCrf-model.pkl'))
    logger.info("train done, time consuming {} s.".format(int(time.time() - start)))

    # ======== 3. 模型评估 ========
    logger.info("Evaluating model...")
    pred_tag_lists, test_tag_lists = model_test(partial_crf_model, test_word_lists, test_tag_lists, word2id, tag2id,
                                                device)

    metrics = Metrics(test_tag_lists, pred_tag_lists, logger, remove_O=True)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    save_predict(test_word_lists, test_tag_lists, pred_tag_lists, os.path.join(output_path, 'partialCrf-predict.col'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/train_demo.col')
    parser.add_argument('--dev_file', default='../data/dev_demo.col')
    parser.add_argument('--test_file', default='../data/test.col')
    parser.add_argument('--gaz_file', default='../data/wv_txt.txt')
    parser.add_argument('--model_save_path', default='../saved_model/')
    parser.add_argument('--output_path', default='../data/output/')
    parser.add_argument('--log_path', default='../log/')
    args = parser.parse_args()

    train(args.train_file, args.dev_file, args.test_file, args.gaz_file,
          args.model_save_path, args.output_path, args.log_path)

