# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2020-12-29

import time

from preprocess import config as DataConfig
from preprocess.alphabet import Alphabet
from preprocess.gazetteer import Gazetteer
from preprocess.util import invalid_word, build_pretrain_embedding, generate_instance_with_gaz
from utils.util import get_logger


class Data:
    def __init__(self, logger=None):
        if logger is None:
            current_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            self.logger = get_logger('../log/', current_time + '-data.log')
        else:
            self.logger = logger
        # word instances[] and instance2index{}
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', label=True)

        self.gaz_lower = DataConfig.gaz_lower
        # All word Trie from outer word embedding
        self.gaz = Gazetteer(self.gaz_lower)
        # word []{} input_file matched Gazetteer(Trie)
        self.gaz_alphabet = Alphabet('gaz')

        self.train_texts = []  # texts for train
        self.dev_texts = []  # texts for dev
        self.test_texts = []  # texts for test
        self.raw_texts = []

        self.train_ids = []  # ids for train
        self.dev_ids = []  # ids for dev
        self.test_ids = []  # ids for test
        self.raw_ids = []

        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None

    def initialization(self, input_files: list, gaz_file: str):
        for file in input_files:
            self.build_alphabet(file)  # build alphabet []{} for file

        if DataConfig.use_gaz:
            self.build_gaz_file(gaz_file)  # build gazetteer(Trie) for word vector lookup book
            for file in input_files:
                self.build_gaz_alphabet(file)  # build alphabet []{} for matched gazetteer

        self.fix_alphabet()
        self.save_alphabet()

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Word   alphabet  size: %s" % self.word_alphabet.size())
        print("     Biword  alphabet size: %s" % self.biword_alphabet.size())
        print("     Char  alphabet   size: %s" % self.char_alphabet.size())
        print("     Label  alphabet  size: %s" % self.label_alphabet.size())
        print("     Gaz   alphabet   size: %s" % self.gaz_alphabet.size())
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("DATA SUMMARY END.")

    def build_alphabet(self, input_file: str):
        """Build word/biword/char/label alphabet from file.

        Args:
            input_file: file path

        Returns:
            None
        """
        in_lines = open(input_file, 'r', encoding='utf-8').readlines()
        for idx in range(len(in_lines)):
            line = in_lines[idx].strip()
            if len(line) > 0:
                pairs = line.strip().split()
                word = pairs[DataConfig.word_col]
                if DataConfig.word_masked and invalid_word(word, pairs):
                    word = DataConfig.MASK_TOKEN
                label = pairs[DataConfig.label_col]

                # add real label to label_alphabet
                if label != DataConfig.UNKNOWN_LABEL and DataConfig.MULTI_LABEL_SPLIT_TAG not in label:
                    self.label_alphabet.add(label)
                self.word_alphabet.add(word)

                if DataConfig.use_char:
                    for char in word:
                        self.char_alphabet.add(char)

                if DataConfig.use_biword:
                    if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                        # assert current line is not the last line
                        biword = word + in_lines[idx + 1].strip().split()[DataConfig.word_col]
                    else:
                        biword = word + DataConfig.UNKNOWN_TOKEN

                    self.biword_alphabet.add(biword)

    def build_gaz_file(self, gaz_file: str):
        """Build gaz file, initial read gaz embedding file

        Args:
            gaz_file: word(gaz) embedding file
        """
        if gaz_file is not None:
            fins = open(gaz_file, 'r', encoding='utf-8').readlines()
            for fin in fins:
                fin = fin.strip().split()[0]
                if fin is not None:
                    self.gaz.insert(fin, "one_source")
            self.logger.info("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            self.logger.info("Gaz file is None, load nothing")

    def build_gaz_alphabet(self, input_file: str):
        """Build alphabet of gaz according to input file.
        """
        in_lines = open(input_file, 'r', encoding='utf-8').readlines()
        word_list = []
        for line in in_lines:
            line = line.strip()
            if len(line) > 0:
                word = line.split()[DataConfig.word_col]
                if DataConfig.word_masked and invalid_word(word, line.strip().split()):
                    word = DataConfig.MASK_TOKEN
                word_list.append(word)
            else:
                if DataConfig.use_char:
                    # if you set use_char=True, we assume that token in word_col is Chinese-word.
                    # So we just run precise search.
                    for idx in range(len(word_list)):
                        search_word = word_list[idx]
                        if search_word in self.gaz.ent2type:
                            # print (search_word, self.gaz.searchId(search_word), self.gaz.searchType(search_word))
                            self.gaz_alphabet.add(search_word)
                else:
                    # Here we assume that token in word_col is Chinese-character.
                    # So we run enumerate search.
                    for idx in range(len(word_list)):  # search from every start position
                        match_list = self.gaz.enumerate_match_list(word_list[idx:])
                        for match_word in match_list:
                            # print (match_word, self.gaz.searchId(match_word), self.gaz.searchType(match_word))
                            self.gaz_alphabet.add(match_word)

                word_list = []
        self.logger.info("gaz alphabet size:", self.gaz_alphabet.size())

    def fix_alphabet(self):
        """Close all alphabet, make sure that following codes will not add other words into them.
        """
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.gaz_alphabet.close()

    def save_alphabet(self, save_path=None):
        """Save word/biword/char/label alphabet to file.
        """
        if save_path is None:
            save_path = DataConfig.alphabet_save_path
        self.fix_alphabet()
        self.word_alphabet.save(save_path, name='word')
        self.biword_alphabet.save(save_path, name='biword')
        self.char_alphabet.save(save_path, name='char')
        self.label_alphabet.save(save_path, name='label')

    def build_pretrain_emb(self, emb_path: str, name: str):
        self.logger.info("Build %s pretrain emb..." % name)
        if name == 'word':
            self.pretrain_word_embedding = build_pretrain_embedding(emb_path, self.word_alphabet,
                                                                    DataConfig.word_emb_dim,
                                                                    DataConfig.norm_word_emb,
                                                                    self.logger)
        elif name == 'biword':
            self.pretrain_biword_embedding = build_pretrain_embedding(emb_path, self.biword_alphabet,
                                                                      DataConfig.biword_emb_dim,
                                                                      DataConfig.norm_biword_emb,
                                                                      self.logger)
        elif name == 'gaz':
            self.pretrain_gaz_embedding = build_pretrain_embedding(emb_path, self.gaz_alphabet,
                                                                   DataConfig.gaz_emb_dim,
                                                                   DataConfig.norm_gaz_emb,
                                                                   self.logger)
        else:
            self.logger.error("Error: you can only generate word/biword/gaz embedding! Illegal input:%s" % name)

    def generate_data_instance(self, input_file: str, name: str):
        """Generate train/dev/test data
        """
        self.fix_alphabet()  # close alphabet again!
        if name == "train":
            self.train_texts, self.train_ids = generate_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                          self.biword_alphabet, self.char_alphabet,
                                                                          self.gaz_alphabet, self.label_alphabet)
        elif name == "dev":
            self.dev_texts, self.dev_ids = generate_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                      self.biword_alphabet, self.char_alphabet,
                                                                      self.gaz_alphabet, self.label_alphabet)
        elif name == "test":
            self.test_texts, self.test_ids = generate_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                        self.biword_alphabet, self.char_alphabet,
                                                                        self.gaz_alphabet, self.label_alphabet)
        else:
            self.logger.info("Error: you can only generate train/dev/test instance! Illegal input:%s" % name)

    def extend_maps_for_crf(self):
        """加了CRF的<start>和<end> (解码的时候需要用到)
        """
        self.word_alphabet.add(DataConfig.START_TOKEN)
        self.word_alphabet.add(DataConfig.END_TOKEN)
        self.label_alphabet.add(DataConfig.START_TOKEN)
        self.label_alphabet.add(DataConfig.END_TOKEN)

        return self.word_alphabet.instance2index, self.label_alphabet.instance2index
