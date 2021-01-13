# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2020-12-29

# make sure that if you set `use_char=True`, the `word_col` in your input file should be Chinese-Word.
# example: `北京` is a Chinese-Word, `北` and `京` are Chinese-Character.
use_char = False
use_biword = False

MAX_SENTENCE_LENGTH = 100
MAX_WORD_LENGTH = -1

char_masked = True
word_masked = True


norm_word_emb = True
norm_char_emb = True
norm_biword_emb = True


use_gaz = False
norm_gaz_emb = False
gaz_lower = True
fix_gaz_emb = False
gaz_dropout = 0.5


tag_schema = "BIO"  # tag shema, BIO or SIO or BIOES or BMES


word_emb_dim = 100
biword_emb_dim = 50
char_emb_dim = 30
gaz_emb_dim = 100


UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
MASK_TOKEN = "<MASK>"
START_TOKEN = '<START>'
END_TOKEN = '<END>'


# label special tag
UNKNOWN_LABEL = 'unknown_label'
MULTI_LABEL_SPLIT_TAG = '|'

UNLABELED_INDEX = -1
IMPOSSIBLE_SCORE = -10000000.0

word_col = 0
label_col = -1

mentor_word_col = 0
mentor_noise_label_col = 2  # origin noise-label
mentor_true_label_col = 3   # corrected label by human
mentor_label_col = 4   # noise-label and true-label equal or not

alphabet_save_path = '../data/dic/'
