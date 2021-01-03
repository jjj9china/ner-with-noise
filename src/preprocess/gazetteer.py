# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2020-12-29

from preprocess.trie import Trie
from preprocess import config as DataConfig


class Gazetteer:
    def __init__(self, lower):
        self.trie = Trie()
        self.ent2type = {}  # word list to source type
        self.ent2id = {DataConfig.UNKNOWN_TOKEN: 0, DataConfig.PAD_TOKEN: 1}   # word list to id
        self.lower = lower
        self.space = ""

    def enumerate_match_list(self, word_list: list) -> list:
        """find macthing word in gaz according to word_list.

        Args:
            word_list: word list need to be sarched.

        Returns:
            List of matched word.
        """
        if self.lower:
            word_list = [word.lower() for word in word_list]
        match_list = self.trie.enumerate_match(word_list, self.space)
        return match_list

    def insert(self, word: str, source: str):
        """Insert word into trie-tree

        Args:
            word:
            source: source type of word
        """
        if self.lower:
            letter_list = [letter.lower() for letter in word]
        else:
            letter_list = [letter for letter in word]
        self.trie.insert(letter_list)

        string = self.space.join(letter_list)
        if string not in self.ent2type:
            self.ent2type[string] = source
        if string not in self.ent2id:
            self.ent2id[string] = len(self.ent2id)

    def search_id(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2id:
            return self.ent2id[string]
        return self.ent2id[DataConfig.UNKNOWN_TOKEN]

    def search_type(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2type:
            return self.ent2type[string]
        print("Error in finding entity type at gazetteer.py, exit program! String:", string)
        exit(0)

    def size(self):
        return len(self.ent2type)




