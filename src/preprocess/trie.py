# -*- coding: utf-8 -*-
# @Author: jjj
# @Date:   2020-12-29

import collections


class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, letter_list: list):
        """Insert a letter_list into Trie

        Args:
            letter_list: letter_list in input file word_col
        """
        current = self.root
        for letter in letter_list:
            current = current.children[letter]
        current.is_word = True

    def search(self, word: list) -> bool:
        current = self.root
        for letter in word:
            current = current.children.get(letter)

            if current is None:
                return False
        return current.is_word

    def starts_with(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    def enumerate_match(self, word, space="", backward=False):
        matched = []
        # while len(word) > 1 does not keep character itself, while word keep character itself
        while len(word) > 1:
            if self.search(word):
                matched.append(space.join(word[:]))
            del word[-1]  # search for every end position
        return matched
