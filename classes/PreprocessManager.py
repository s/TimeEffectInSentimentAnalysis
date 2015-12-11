# -*- coding: utf-8 -*-

import json
from config import *
from helpers.Preprocessor import Preprocessor
from helpers.GeneralHelpers import GeneralHelpers


class PreprocessManager(Preprocessor):

    def __init__(self):
        Preprocessor.__init__(self)
        self.__helper = GeneralHelpers()
        self.__root_cache = self.__helper.load_roots_cache()
        self.__suggestion_cache = self.__helper.load_suggestion_cache()
        self.__dictionaries_directory = PROJECT_ROOT_DIRECTORY + DICTIONARIES_DIR_NAME

    def remove_characters_in_string(self, text, characters=[]):
        """
        Removes specified characters in a string
        :param text: String
        :param characters: list, list of characters to remove
        :return: new string
        """

        if len(characters):
            for char in characters:
                text = text.replace(char, "")

        return text

    def correct_misspelling(self, word):
        """
        Suggest correct words for given word
        :param text: string, word
        :return: string, suggestion
        """
        has_special_keyword, special_keyword = self._has_special_keyword(word)

        if not has_special_keyword:
            if word in self.__suggestion_cache:
                return self.__suggestion_cache[word]
            else:
                corrected_word = self.__helper.correct_misspelling_from_zemberek(word)
                self.__suggestion_cache[word] = corrected_word
                return corrected_word
        else:
            self.__suggestion_cache[word] = special_keyword
            return special_keyword



    def find_root_of_word(self, word):
        """
        Returns root of word
        :param word: string, word
        :return: string, root of word
        """
        has_special_keyword, special_keyword = self._has_special_keyword(word)
        if not has_special_keyword:
            if word in self.__root_cache:
                return self.__root_cache[word]
            else:
                root_of_word = self.__helper.find_root_from_zemberek(word)
                self.__root_cache[word] = root_of_word
                return root_of_word
        else:
            self.__root_cache[word] = special_keyword
            return special_keyword

    def save_caches(self):
        """
        Saves suggestion and root finding caches' changes
        :return: void
        """
        self.__helper.save_changes_in_suggestion_cache(self.__suggestion_cache)
        self.__helper.save_changes_in_root_cache(self.__root_cache)

    def _has_special_keyword(self, word):
        """
        If model name is present in the word, returns the model name
        :param word: string, word
        :return: bool, string
        """
        has_special_keyword = False
        special_keyword = ""

        for keyword in SPECIAL_KEYWORDS:
            if keyword.lower() in word.lower():
                has_special_keyword = True
                special_keyword = keyword

        return has_special_keyword, special_keyword
