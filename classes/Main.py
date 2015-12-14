# -*- coding: utf-8 -*-

import numpy as np
from config import *
from random import randint

from DBManager import DBManager
from PlotManager import PlotManager
from ImportManager import ImportManager
from FeatureManager import FeatureManager
from ExperimentManager import ExperimentManager

from helpers.GeneralHelpers import GeneralHelpers

class Main:

    def __init__(self):
        self.__db_manager = DBManager()
        self.__helper = GeneralHelpers()
        self.__plot_manager = PlotManager()
        self.__feature_manager = FeatureManager()

        self.years = ("2012", "2013", "2014", "2015")

    def retrieve_tweets(self, file_path_of_ids):
        """
        Runs Import Manager to retrieve and import tweets
        :param file_path_of_ids: String, file path of tweets to import
        :return: void
        """
        importer = ImportManager(file_path_of_ids)
        importer.run()

    def extract_features_and_generate_arff(self, n=3, analyzer='char', year='2012'):
        """
        Makes necessary function calls to extract features for given year and to generate arff file
        :param n: int, ngram count
        :param analyzer: string, word or char
        :param year: string, 2012, 2013, 2014, 2015 or ALL
        :return: string, path of generated arff file
        """

        # Getting tweets with year
        print("Getting tweets for year "+ year)
        tweets_for_given_year = self.__db_manager.get_tweets_for_year(year)

        print("Generating document and classes of tweets.")
        document, classes = self.__feature_manager.create_document_and_classes_for_tweets(tweets_for_given_year, True)

        print("Fitting the data, finding ngrams and frequencies.")
        ngrams, arff_data, vectorizer, X = self.__feature_manager.fit_data(document, classes, n, analyzer)

        print("Formatting the data for arff lib format.")
        formatted_arff_data = self.__feature_manager.format_data_for_arff(ngrams, arff_data)

        print("Generating file.")
        # Experiment name, 1grams, 2grams, 3grams.. or words
        experiment_name = str(n)+'Gram' if analyzer == 'char' else 'Word'

        # File name, TTNet_3grams_2012
        file_name = MODEL_NAME + '_' + experiment_name + '_' + year

        # File name randomized TTNet_3grams_2012_asfas12.arff
        file_name = self.__helper.generate_random_file_name(file_name, ARFF_FILE_EXTENSION)

        # Arff file path ...../DataSet-ARFF/3Gram/TTNet/TTNet_3grams_2012_asfas12.arff
        arff_file_path = PROJECT_ROOT_DIRECTORY + DATASET_ARFF_DIR_NAME + experiment_name + '/' + MODEL_NAME + '/'

        # Generating the file with data
        self.__helper.generate_arff_file(arff_file_path, file_name, formatted_arff_data)

        print("Arff file generated at path:"+arff_file_path+file_name)

    def plot_years_scores(self, root_dir):
        """
        Makes necessary function calls to plot years scores
        :param dir: string
        :return: void
        """
        self.__plot_manager.plot_years_scores_from_root_directory(root_dir)

    def plot_2012_vs_rest(self, root_dir):
        """
        Makes necessary function calls to plot 2012 vs REST scores
        :param root_dir: string
        :return: void
        """
        self.__plot_manager.plot_2012_vs_rest(root_dir)

    def plot_top_feature_frequencies_in_years(self):
        """
        Makes necessary function calls to plot top features frequencies' in years
        :return: void
        """
        years_features_counts = {}

        for year in self.years:
            years_features_counts[year] = self.find_frequency_dictionary_for_year(year)

        self.__plot_manager.plot_top_feature_frequencies_in_years(years_features_counts)

    def find_frequency_dictionary_for_year(self, year):
        """
        Finds frequencies of each feature for given year
        :param year: string
        :return: dict
        """
        # For this particular method, find_roots=True, n=1, analyzer=word because we're working with top info gain words

        tweets_for_the_year = self.__db_manager.get_tweets_for_year(year)
        document, classes = self.__feature_manager.create_document_and_classes_for_tweets(tweets_for_the_year, find_roots=True)
        ngrams, arff_data, vectorizer, X = self.__feature_manager.fit_data(document, classes, n=1, analyzer='word')

        terms = vectorizer.get_feature_names()
        freqs = X.sum(axis=0).A1

        result = sorted(zip(freqs, terms), reverse=True)

        freqs = [elm[0] for elm in result]
        terms = [elm[1] for elm in result]

        final_result = dict(zip(terms, freqs))

        return final_result

    def plot_years_intersection_scores(self):
        """
        Makes necessary function callst to plot a matrix which shows years' vocabularies similarities
        :return: void
        """
        years_features_counts = {}

        for year in self.years:
            years_features_counts[year] = self.find_frequency_dictionary_for_year(year)

        self.__plot_manager.plot_years_intersection_scores(years_features_counts)

    def run_experiment_with_scikit_learn(self, n=1, analyzer='word'):
        """
        Makes necessary method calls to run the experiment on scikit learn.
        :param n: int, count n in n-gram
        :param analyzer: string, either 'word' or 'char'
        :return: void
        """
        # Retrieving all tweets from database
        print("Retrieving all tweets from database.")
        tweets_for_all_years = {}
        # Iterating over all years
        for year in self.years:
            # Retrieving tweets for the year
            tweets_for_year = self.__db_manager.get_tweets_for_year(year)
            tweets_for_all_years[year] = tweets_for_year

        # Creating a big list of tweets
        print("Creating a big list of tweets.")
        all_tweets = []
        # Appending all tweets together
        for year, tweets in tweets_for_all_years.iteritems():
            all_tweets += tweets

        # Generating document
        print("Generating document and classes by preprocessing")
        # Preprocessing and generation of document
        document, classes = self.__feature_manager.create_document_and_classes_for_tweets(all_tweets, True)

        # Getting years' tweets counts
        print("Getting years' tweets counts.")
        years_tweets_counts = {}
        for year in self.years:
            years_tweets_counts[year] = len(tweets_for_all_years[year])

        experiment_manager = ExperimentManager(years_tweets_counts, n, analyzer)
        print("Running experiments.")
        for i in range(0, N_EXPERIMENTS):
            print("Experiment:"+str(i))
            experiment_manager.run_experiment(document, classes)
            print("-"*20)
