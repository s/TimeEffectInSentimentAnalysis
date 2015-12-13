# -*- coding: utf-8 -*-

import re
import arff
import json
import codecs
import random
import pprint
import numpy as np
import collections

from config import *
from glob import glob
from subprocess import Popen, PIPE, STDOUT

from classes.DBManager import DBManager

class GeneralHelpers:

    def __init__(self):
        self.__db_manager = DBManager()
        self.regexp_for_predict_lines = "\d{1,}\s{1,}\d{1}:\w{1,8}.{1,}"
        self.__dictionaries_directory = PROJECT_ROOT_DIRECTORY + DICTIONARIES_DIR_NAME

    def pretty_print_list(self, list_to_print, message):
        """

        :param list_to_print: List, list to print
        :param message: String, a message to print before printing list
        :return: void
        """
        if len(list_to_print):
            print(message)

            for element in list_to_print:
                print(element)

    def get_chunks_of_list(self, list_to_chunk, chunk_size):
        """
        Returns chunks of new list
        :param list_to_chunk: List, List to divide into chunks
        :param chunk_size: Int, size of a chunk
        :return:
        """

        n = max(1, chunk_size)
        return [list_to_chunk[i:i + n] for i in range(0, len(list_to_chunk), n)]

    def find_root_from_zemberek(self, word):
        """
        Finds root of given word from zemberek
        :param word: string, word
        :return: string, root of word
        """
        connection_output = self._make_jar_call(ZEMBEREK_ROOT_FINDER_JAR_FILE_NAME, word)
        return connection_output

    def correct_misspelling_from_zemberek(self, word):
        """
        Corrects misspelled words by asking to zemberek
        :param word: string, word to correct
        :return: string, corrected word
        """
        connection_output = self._make_jar_call(ZEMBEREK_SUGGESTION_FINDER_JAR_FILE_NAME, word)

        suggestions = connection_output.split(",")
        if len(suggestions):
            if word in suggestions:
                return word
            else:
                return suggestions[0]
        else:
            return word

    def save_changes_in_suggestion_cache(self, suggestions_cache):
        """
        Saves given suggestion cache to file
        :param suggestions_cache: dict, suggestions
        :return: void
        """
        suggestion_cache_file_path = self.__dictionaries_directory + MODEL_NAME + '/' + SUGGESTION_CACHE_FILE_NAME
        self._write_json_to_file(suggestions_cache, suggestion_cache_file_path)

    def save_changes_in_root_cache(self, roots_cache):
        """
        Saves given roots cache to file
        :param roots_cache: roots_cache, dict, roots
        :return: void
        """
        roots_file_path = self.__dictionaries_directory + MODEL_NAME + '/' + ROOTS_CACHE_FILE_NAME
        self._write_json_to_file(roots_cache, roots_file_path)

    def load_suggestion_cache(self):
        """
        Loads previously asked (to zemberek) suggestion cache
        :return: void
        """
        suggestion_cache_file_path = self.__dictionaries_directory + MODEL_NAME + '/' + SUGGESTION_CACHE_FILE_NAME
        if os.path.isfile(suggestion_cache_file_path):
            with open(suggestion_cache_file_path, "r") as suggestions_file:
                suggestion_cache = json.load(suggestions_file)
                return suggestion_cache

    def load_roots_cache(self):
        """
        Loads previously asked (to zemberek) word roots cache
        :return: void
        """
        roots_file_path = self.__dictionaries_directory + MODEL_NAME + '/' + ROOTS_CACHE_FILE_NAME
        if os.path.isfile(roots_file_path):
            with open(roots_file_path, "r") as roots_file:
                roots_cache = json.load(roots_file)
                return roots_cache

    def generate_arff_file(self, file_path, file_name, arff_data):
        """
        Generates arff file
        :param file_name: file_name for arff data
        :param arff_data: dict, arff_data
        :return: string, generated file path
        """

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        arff_file = codecs.open(file_path+file_name, 'w+', encoding='utf-8')
        arff.dump(arff_data, arff_file)
        arff_file.close()

    def generate_random_string(self, n):
        """
        Generates random string with size of n
        :param n: int, size
        :return: string, n-length random string
        """
        random_string = ''.join(random.choice('abcdefghijklmnoprstuvyzxw1234567890') for _ in range(n))
        return random_string

    def generate_random_file_name(self, file_name, extension):
        """
        Generates random file name with given file name
        :param file_name: string, desired file name root
        :return: string, randomized file name
        """
        random_file_appendix = self.generate_random_string(5)
        final_file_name = file_name + '_' + random_file_appendix + extension
        return final_file_name

    def _write_json_to_file(self, json_data, file_path):
        """
        Writes given data to given path
        :param data: list or dict, json_data
        :param file_path: string
        :return:
        """
        with open(file_path, 'w') as outfile:
            json.dump(json_data, outfile)

    def _make_jar_call(self, jar_file_name, word):
        """
        Makes a jar call with proper parameters
        :param jar_file_name: string, jar file to make call
        :param word: string, parameter word
        :return: string, connection output
        """
        jar_file_path = JAR_FILE_DIR_NAME + jar_file_name

        # Making the call
        process_call = Popen(['java', '-jar', jar_file_path, word], stdout=PIPE, stderr=STDOUT)

        # Getting output
        output = process_call.communicate()[0].decode('utf-8')
        return output

    def get_accuracy_scores_for_years_from_root_dir(self, root_dir):
        """
        Returns a dict of __years' classifier scores
        :param root_dir: string, path to root directory
        :return: dict, __years' classifier scores
        """
        years_scores = {}

        # Iterating over directories in root directory
        for root, dirs, files in os.walk(root_dir):

            # Iterating over files in a directory
            for file_name in files:

                # If it's a txt file we got it. E.g. TTNet_2015_SMO.txt
                if file_name.endswith('.txt'):

                    file_path = root + '/' + file_name
                    model_name, year, classifier_name = file_name.rstrip('.txt').split("_") #TTNet, 2015, SMO

                    if not year in years_scores:
                        years_scores[year] = {}

                    with open(file_path, 'r') as classifier_log_file:
                        file_content = classifier_log_file.read()
                        years_scores[year][classifier_name] = self.get_accuracy_score_from_log_file_content(file_content)

        # Calculating mean for each year and sorting
        for year, classifiers in years_scores.iteritems():
            all_classifier_scores = np.array(classifiers.values())
            years_scores[year]['MEAN'] = round(all_classifier_scores.mean(), 2)
            years_scores[year] = collections.OrderedDict(sorted(years_scores[year].items()))

        sorted_years_scores = collections.OrderedDict(sorted(years_scores.items()))

        return sorted_years_scores

    def get_accuracy_score_from_log_file_content(self, log_file_content):
        """
        Returns accuracy score of log file
        :param log_file_content: string
        :return: float
        """
        regexp_for_accuracy_lines = "Correctly Classified Instances.{1,}"

        classifier_accuracy_lines = re.findall(regexp_for_accuracy_lines, log_file_content, re.IGNORECASE)

        if len(classifier_accuracy_lines):
            accuracy_line = classifier_accuracy_lines[0]
            accuracy_line_components = accuracy_line.split() #['Correctly', 'Classified', 'Instances', '209', '41.8', '%']
            accuracy = accuracy_line_components[4]
            return float(accuracy)
        else:
            return -1.0
