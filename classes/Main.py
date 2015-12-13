# -*- coding: utf-8 -*-

import numpy as np
from config import *
from random import randint

from DBManager import DBManager
from ImportManager import ImportManager
from PlotManager import PlotManager
from FeatureManager import FeatureManager

from helpers.GeneralHelpers import GeneralHelpers

class Main:

    def __init__(self):
        self.__db_manager = DBManager()
        self.__helper = GeneralHelpers()
        self.__plot_manager = PlotManager()
        self.__feature_manager = FeatureManager()

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