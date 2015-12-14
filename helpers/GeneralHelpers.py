# -*- coding: utf-8 -*-

import os
import re
import arff
import json
import codecs
import random
import pprint
import datetime
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
        suggestion_cache_file_path = self.__dictionaries_directory + SUGGESTION_CACHE_FILE_NAME
        self._write_json_to_file(suggestions_cache, suggestion_cache_file_path)

    def save_changes_in_root_cache(self, roots_cache):
        """
        Saves given roots cache to file
        :param roots_cache: roots_cache, dict, roots
        :return: void
        """
        roots_file_path = self.__dictionaries_directory + ROOTS_CACHE_FILE_NAME
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

    def get_accuracy_scores_for_years_from_root_dir(self, root_dir):
        """
        Returns a dict of years' classifier scores
        :param root_dir: string, path to root directory
        :return: dict, years' classifier scores
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

    def get_accuracy_scores_for_experiment_years_from_root_dir(self, root_dir):
        """
        Returns a dict of scores
        :param root_dir: string, path to root directory
        :return: dict, years' classifier scores
        """
        lines_scores = {}

        # Iterating over directories in root directory
        for root, dirs, files in os.walk(root_dir):

            experiment_scores = []
            # Iterating over files in a directory
            for file_name in files:

                if file_name.endswith('.txt'):

                    file_path = root + '/' + file_name

                    with open(file_path, 'r') as classifier_log_file:
                        file_content = classifier_log_file.read()
                        accuracy_score_of_log_file = self.get_accuracy_score_from_log_file_content(file_content)
                        experiment_scores.append(accuracy_score_of_log_file)

            if len(experiment_scores):
                mean_of_experiment_scores = round(np.array(experiment_scores).mean(), 2)
                root_components = root.split("/")
                line_name = root_components[-2]
                point_name = root_components[-1]
                point_name = LINES_DIR_DICT[point_name]

                if not line_name in lines_scores:
                    lines_scores[line_name] = {}

                lines_scores[line_name][point_name] = mean_of_experiment_scores

        for line, points in lines_scores.iteritems():
            lines_scores[line] = collections.OrderedDict(sorted(points.items()))

        lines_scores = collections.OrderedDict(sorted(lines_scores.items()))

        return lines_scores

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

    def get_log_files_stats(self, root_dir):
        """
        Returns each classifier's monthly accuracy scores from log files.
        :param root_dir: string
        :return: dict
        """

        # Going to log files directory
        os.chdir(root_dir)

        all_log_files = {}

        # iterating over log files of a model (say, Turkcell)
        for txt_file in glob("*.txt"):
            # openning file
            with open(txt_file, 'r') as a_log_file:
                # getting log file's name
                file_name = a_log_file.name.split(".")[0]

                # updating data model with log file's content
                all_log_files.update({file_name: a_log_file.read()})

        # we got a model and we need it month's lengths (Say, we got 42 tweets for October)
        self.model_month_counts = self.__db_manager.get_months_lengths()

        all_accuracy_scores = {}

        # iterating log files we previously read
        for log_file_name, log_file_content in all_log_files.iteritems():
            accuracy_scores = self.get_accuracy_scores_per_month_from_log_file(log_file_content)

            if not log_file_name in all_accuracy_scores:
                all_accuracy_scores[log_file_name] = accuracy_scores

        return all_accuracy_scores

    def get_accuracy_scores_per_month_from_log_file(self, log_file_content):
        """
        Returns each year's [true, total] predictions for given log file content
        :param log_file_content: string
        :return: dict
        """
        predict_lines_in_file = re.findall(self.regexp_for_predict_lines, log_file_content)

        correct_vs_total_values = {}
        start_idx = 0
        # e.g. { 2014: [42, 42, ...., 38] }
        for year, months_lengths in self.model_month_counts.iteritems():

            # e.g. 42
            for month_length in months_lengths:

                end_idx = start_idx + month_length

                # a months' stats
                a_months_stats = [0, month_length]  # (Correct, Total)

                # slicing months' lines to find lines like: "1 3:positive 3:positive          0      0     *1"
                months_lines = predict_lines_in_file[start_idx:end_idx]

                # iterating over a month's lines to find accuracy
                for month_line in months_lines:

                    # if line contains + it's an error
                    if not '+' in month_line:
                        a_months_stats[0] += 1

                if not year in correct_vs_total_values:
                    correct_vs_total_values[year] = []

                correct_vs_total_values[year].append(a_months_stats)

                start_idx += month_length

        """
        Example correct_vs_total_values:
        {
            2013: [[25, 43], [26, 42], [21, 42], [31, 44], [25, 42], [24, 40], [27, 42], [23, 42], [23, 43], [28, 41], [25, 42], [21, 39]],
            2014: [[35, 55], [19, 39], [25, 46], [32, 43], [23, 42], [25, 42], [27, 42], [28, 45], [25, 45], [27, 45], [23, 42], [15, 27]],
            2015: [[25, 43], [32, 56], [34, 57], [36, 56], [31, 54], [35, 55], [37, 68], [38, 57], [27, 51]]
        }
        """
        # Let's find accuracies now.

        accuracy_scores = {}

        for year, all_months_predictions in correct_vs_total_values.iteritems():

            if not year in accuracy_scores:
                accuracy_scores[year] = []

            for month_predictions in all_months_predictions:
                correct_predictions = month_predictions[0]
                total_predictions = month_predictions[1]

                one_acc_score = float(correct_predictions) / total_predictions
                one_acc_score *= 100
                one_acc_score = round(one_acc_score, 2)


                accuracy_scores[year].append(one_acc_score)

        return accuracy_scores

    def find_key_of_given_value_in_dict(self, dictionary, value_to_search):
        """
        Returns key of given value
        :param dictionary: dict
        :param value_to_search: string
        :return: string
        """
        for k, v in dictionary.iteritems():
            if v == value_to_search:
                return k
        return -1

    def find_most_distinct_n_samples(self, document_one, document_two, n_samples):
        """
        Returns most distinct n_samples from document_two comparing to document_one (documents are arff data)
        :param document_one: list
        :param document_two: list
        :param n_samples: int
        :return: list
        """
        pass

    def cumulate_years_scores(self, years_scores):
        """
        Cumulates and finds averages of all years' scores
        :param years_scores: list
        :return: dict
        """
        final_result_of_all_experiments = {}
        for one_experiments_scores in years_scores:

            for line_name, one_lines_scores in one_experiments_scores.iteritems():

                if not line_name in final_result_of_all_experiments:
                    final_result_of_all_experiments[line_name] = {}

                for setup_name, score_or_list in one_lines_scores.iteritems():

                    if not setup_name in final_result_of_all_experiments[line_name]:
                        final_result_of_all_experiments[line_name][setup_name] = []

                    final_result_of_all_experiments[line_name][setup_name].append(score_or_list)

        for line_name, one_lines_scores in final_result_of_all_experiments.iteritems():
            if line_name != "line2":

                for setup_name, scores_list in one_lines_scores.iteritems():
                    final_result_of_all_experiments[line_name][setup_name] = round(np.mean(scores_list), 2)

            else:

                for setup_name, scores_list in one_lines_scores.iteritems():
                    np_array_scores_list = np.array(scores_list)
                    min_mean_max_mean = np.mean(scores_list, axis=0)
                    final_result_of_all_experiments[line_name][setup_name] = min_mean_max_mean

        print('Relative scores:')
        pprint.pprint(final_result_of_all_experiments, width=2)
        for line_name, score in final_result_of_all_experiments['line3'].iteritems():

            target_line1_key = self.get_line1_key_from_line3_key(line_name)
            new_line3_score = final_result_of_all_experiments['line1'][target_line1_key] + score
            final_result_of_all_experiments['line3'][line_name] = new_line3_score

        print('Absolute scores:')
        pprint.pprint(final_result_of_all_experiments, width=2)

        return final_result_of_all_experiments

    def calculate_relative_scores(self, all_scores):
        """
        Calculates relative scores of line3 according to line1
        :return: void
        """

        line1_scores = all_scores['line1']

        for setup_name, score in all_scores['line3'].iteritems():

            target_line1_key = self.get_line1_key_from_line3_key(setup_name)
            difference = score - line1_scores[target_line1_key]
            all_scores['line3'][setup_name] = difference

        return all_scores


    def get_line1_key_from_line3_key(self, line3_key):
        """
        Returns relevant line1 key from line3 key
        :param line3_key: string
        :return: string
        """
        regexp_for_target_keys = "\+(2012|2013|2014|2015)_50"
        target_line1_key = re.sub(regexp_for_target_keys, "", line3_key)
        return target_line1_key

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