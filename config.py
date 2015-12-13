# -*- coding: utf-8 -*-
import os

PROJECT_ROOT_DIRECTORY = os.path.realpath("../")

MODEL_NAME = "TTNet" # Turkcell or TTNet

JAR_FILE_DIR_NAME = "jars/"
CHARTS_DIR_NAME = "/Charts/"
DATASET_TXT_DIR_NAME = "/DataSet-TXT/"
DATASET_ARFF_DIR_NAME = "/DataSet-ARFF/"
DATASET_LOGS_DIR_NAME = "/DataSet-Logs/"
DICTIONARIES_DIR_NAME = "/Dictionaries/"

ROOTS_CACHE_FILE_NAME = "roots.json"
SUGGESTION_CACHE_FILE_NAME = "suggestions.json"

ZEMBEREK_ROOT_FINDER_JAR_FILE_NAME = "ZemberekWordStemFinder.jar"
ZEMBEREK_SUGGESTION_FINDER_JAR_FILE_NAME = "ZemberekSuggestionFinder.jar"

ARFF_FILE_RELATION = "Ngrams"
ARFF_FILE_EXTENSION = ".arff"
ARFF_FILE_TWEET_Y_NAME = "sentiment"
ARFF_FILE_DESCRIPTION = "Ngrams of Tweets"

LOGS_YEARS_ITSELF_DIR_NAME = "/YearsOnly/"
LOGS_2012_VS_REST = "/2012vsREST/"

TWITTER_CONSUMER_TOKEN = ""
TWITTER_CONSUMER_SECRET = ""
TWITTER_ACCESS_TOKEN_KEY = ""
TWITTER_ACCESS_TOKEN_SECRET = ""

FEATURE_TYPE = "Word" # or #3Gram

SPECIAL_KEYWORDS = ['TTNet', 'Turkcell', '3G'] # TODO

if "TTNet" == MODEL_NAME:
    INFO_GAIN_ATTRIBUTES = ["sev", "çek", "bedava", "internet", "iyi", "teşekkür", "söyle", "bebek", "yavaş", "indir",
                            "genç", "san", "bu", "tatlı", "ede", "nefret", "gangnam", "çekim", "para", "siz"]
elif "Turkcell" == MODEL_NAME:
    INFO_GAIN_ATTRIBUTES = ["müzik", "arena", "internet", "şarkı", "teşekkür", "böyle", "cif", "yavaş", "güzel", "gibi",
                            "net","sen", "hız", "para", "aracı", "ver", "daha", "kes", "sev"]
