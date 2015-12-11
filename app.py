# -*- coding: utf-8 -*-

from config import *
from classes.Main import Main
from helpers.Preprocessor import Preprocessor

if __name__ == "__main__":
    main = Main()

    """
    Example code to import tweets
    """
    #file_path = os.path.realpath(PROJECT_ROOT_DIRECTORY+DATASET_TXT_DIR_NAME+"TTNetTweets2012.txt")
    #main.retrieve_tweets(file_path)

    """
    Example code to generate arff file with given feature parameters
    """
    main.extract_features_and_generate_arff(n=3, analyzer='char', year='2012')
