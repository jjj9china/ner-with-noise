# -*- coding:utf-8 -*-

# sampling rate
SAMPLING_RATE = 0.7

# min and max sentences length when loading data.
MIN_SEQ_LEN = 2
MAX_SEQ_LEN = 200

# special tag used in label correct
UNKNOWN_TAG = 'Unknown_Tag'
MULTI_LABEL_SPLIT_TAG = '|'

# default separator between words
SENTENCE_SEP = ''

# label need filter in train data when the corresponding words are in dev data.
# Example:
# when 'Jack Ma' are in dev data and its label is in FILTER_LABEL,
# we will remove all data contains 'Jack Ma' in train data.
FILTER_LABEL = ['PERSON_NAME']
