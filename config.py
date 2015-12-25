# -*- coding: utf-8 -*-
import os

PROJECT_ROOT_DIRECTORY = os.path.realpath("../")

MODEL_NAME = "TurkcellMerged" # Turkcell or TTNet

JAR_FILE_DIR_NAME = "jars/"
CHARTS_DIR_NAME = "/Charts/"
DATASET_CSV_DIR_NAME = "/DataSet-CSV/"
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

LINES_DIR_DICT = {
    '2012-500,2013-300':'L1A',
    '2012-500,2014-300':'L1B',
    '2012-500,2015-300':'L1C',

    '2012-500+2013-R50,2013-300':'L2A',
    '2012-500+2014-R50,2014-300':'L2B',
    '2012-500+2015-R50,2015-300':'L2C',

    '2012-500+2013-S50,2013-300':'L3A',
    '2012-500+2014-S50,2014-300':'L3B',
    '2012-500+2015-S50,2015-300':'L3C',

    '2012-500+2013-200,2013-300':'L4A',
    '2012-500+2014-200,2014-300':'L4B',
    '2012-500+2015-200,2015-300':'L4C'
}

if "Turkcell" == MODEL_NAME:
    INFO_GAIN_ATTRIBUTES = ["sev", "çek", "bedava", "internet", "iyi", "teşekkür", "söyle", "bebek", "yavaş", "indir",
                            "genç", "san", "bu", "tatlı", "ede", "nefret", "gangnam", "çekim", "para", "siz"]
elif "TTNet" == MODEL_NAME:
    INFO_GAIN_ATTRIBUTES = ["müzik", "arena", "internet", "şarkı", "teşekkür", "böyle", "cif", "yavaş", "güzel", "gibi",
                            "net","sen", "hız", "para", "aracı", "ver", "daha", "kes", "sev"]

#ALE CONSTANTS
N_EXPERIMENTS = 3
BASE_YEAR = '2012'
MOST_DISTINCT = -998
RANDOM_SAMPLE_SIZE = 400 # OR 50
EXPERIMENT_DIR_NAME = 'ALE' # Active Learning Experiment
ALE_PARTITION_50_KEY = '400' # OR 50
ALE_PARTITION_500_KEY = '1600' # OR 500
ALE_PARTITION_300_KEY = '800TE' # OR 300
ALE_PARTITION_200_KEY = '800TR' # OR 200
MOST_DISTINCT_SAMPLE_SIZE = 400 # OR 50
NOT_INCLUDE_YEARS_TWEETS = '-999'
LINE3_CHOOSING_SAMPLES_SIZE = 10
LINE2_RANDOM_ITERATION_NUMBER = 10
TEST_YEARS = ('2013', '2014', '2015')
ALE_EACH_YEAR_TWEET_LIMIT_COUNT = 1600 # OR 500
ALE_LINE3_KEYS = ["L0", "L1", "L2", "L3"]
LINE3_CHOOSING_SAMPLES_ITERATION_COUNT = 5
PLOT_DECISION_BOUNDARIES_FOR_LINE_3 = False
BASE_YEAR_FOR_DISTINCTION_PROBABILITIES = '2012'
SENTIMENT_CLASSES = ["positive", "negative", "neutral"]

LINES_SETUPS = {
    'line1': {
        'train': [
            {
                '2012':ALE_PARTITION_500_KEY
            },
            {
                '2012':ALE_PARTITION_500_KEY
            },
            {
                '2012':ALE_PARTITION_500_KEY
            }
        ],
        'test': [
            {
                '2013':ALE_PARTITION_300_KEY
            },
            {
                '2014':ALE_PARTITION_300_KEY
            },
            {
                '2015':ALE_PARTITION_300_KEY
            }
        ]
    },
    'line2': {
        'train': [
            {
                '2012':ALE_PARTITION_500_KEY,
                '2013':str(RANDOM_SAMPLE_SIZE)
            },
            {
                '2012':ALE_PARTITION_500_KEY,
                '2014':str(RANDOM_SAMPLE_SIZE)
            },
            {
                '2012':ALE_PARTITION_500_KEY,
                '2015':str(RANDOM_SAMPLE_SIZE)
            }
        ],
        'test': [
            {
                '2013':ALE_PARTITION_300_KEY
            },
            {
                '2014':ALE_PARTITION_300_KEY
            },
            {
                '2015':ALE_PARTITION_300_KEY
            }
        ]
    },
    'line3': {
        'train': [
            {
                '2012':ALE_PARTITION_500_KEY,
                '2013':MOST_DISTINCT_SAMPLE_SIZE
            },
            {
                '2012':ALE_PARTITION_500_KEY,
                '2014':MOST_DISTINCT_SAMPLE_SIZE
            },
            {
                '2012':ALE_PARTITION_500_KEY,
                '2015':MOST_DISTINCT_SAMPLE_SIZE
            }
        ],
        'test': [
            {
                '2013':ALE_PARTITION_300_KEY
            },
            {
                '2014':ALE_PARTITION_300_KEY
            },
            {
                '2015':ALE_PARTITION_300_KEY
            }
        ]
    },
    'line4':{
        'train': [
            {
                '2012':ALE_PARTITION_500_KEY,
                '2013':ALE_PARTITION_200_KEY
            },
            {
                '2012':ALE_PARTITION_500_KEY,
                '2014':ALE_PARTITION_200_KEY
            },
            {
                '2012':ALE_PARTITION_500_KEY,
                '2015':ALE_PARTITION_200_KEY
            }
        ],
        'test': [
            {
                '2013':ALE_PARTITION_300_KEY
            },
            {
                '2014':ALE_PARTITION_300_KEY
            },
            {
                '2015':ALE_PARTITION_300_KEY
            }
        ]
    }
}
LINE3_PROB_SETUP = [
    {
        'train':('2012', ALE_PARTITION_500_KEY),
        'test':('2013', ALE_PARTITION_200_KEY)
    },
    {
        'train':('2012', ALE_PARTITION_500_KEY),
        'test':('2014', ALE_PARTITION_200_KEY)
    },
    {
        'train':('2012', ALE_PARTITION_500_KEY),
        'test':('2015', ALE_PARTITION_200_KEY)
    }
]
