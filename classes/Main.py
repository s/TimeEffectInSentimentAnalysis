# -*- coding: utf-8 -*-

import numpy as np
from config import *
from random import randint

from DBManager import DBManager
from ImportManager import ImportManager
from FeatureManager import FeatureManager

from helpers.GeneralHelpers import GeneralHelpers

class Main:

    def __init__(self):
        self.__db_manager = DBManager()
        self.__helper = GeneralHelpers()
        self.__feature_manager = FeatureManager()

    def retrieve_tweets(self, file_path_of_ids):
        """
        Runs Import Manager to retrieve and import tweets
        :param file_path_of_ids: String, file path of tweets to import
        :return: void
        """
        importer = ImportManager(file_path_of_ids)
        importer.run()