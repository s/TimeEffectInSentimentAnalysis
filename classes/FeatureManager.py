# -*- coding: utf-8 -*-
import numpy as np
import PreprocessManager

from config import *
from sklearn.feature_extraction.text import CountVectorizer

class FeatureManager:
    """
    This class manages feature-related functionalities
    """

    def __init__(self):
        pass

    def create_document_and_classes_for_tweets(self, tweets, find_roots):
        """
        Creates document with irrelevant information removed and classes of them
        :param tweets: list, list of tweets
        :param find_roots: bool, whether or not to find roots of words
        :return: list, list, document and classes
        """

        classes = []
        document = []

        preprocess_manager = PreprocessManager.PreprocessManager()

        # Iterating over all tweets
        for tweet_index, tweet in enumerate(tweets):

            # Removing irrelevant features
            preprocessed_tweet = preprocess_manager.clean_all(tweet.text)

            words_of_tweets = preprocessed_tweet.split(" ")

            # Iterating over all words in tweet to correct misspellings
            for word_index, word in enumerate(words_of_tweets):

                # Correcting misspellings
                spell_corrected_word = preprocess_manager.correct_misspelling(word)
                words_of_tweets[word_index] = spell_corrected_word

            # If researcher wants to use root of words as features
            if find_roots:

                # Iterating over all words in tweet to find roots
                for word_index, word in enumerate(words_of_tweets):

                    # Finding roots
                    root_of_word = preprocess_manager.find_root_of_word(word)
                    words_of_tweets[word_index] = root_of_word

            processed_tweet = ' '.join(words_of_tweets)

            # Appending to document
            document.append(processed_tweet)
            classes.append(tweet.tweet_class)

        preprocess_manager.save_caches()
        return document, classes

    def fit_data(self, document, classes, n, analyzer):
        """
        Fits data and returns n-grams for given analyzer
        :param document: list, document of tweets
        :param classes: list, classes of tweets
        :return: list, dict, CountVectorizer, X
        """
        # matrix_freq = np.asarray(X.sum(axis=0)).ravel()
        # final_matrix = np.array([matrix_terms,matrix_freq])
        # print(len(X.toarray()))

        # Creating the vectorizer
        vectorizer = CountVectorizer(ngram_range=(n, n), analyzer=analyzer)

        # Fitting document
        X = vectorizer.fit_transform(document)

        # Getting our n-grams list
        ngrams = np.array(vectorizer.get_feature_names()).tolist()

        # And our arff data
        arff_data = X.toarray().tolist()

        # Appending classes to each row
        for idx, cl in enumerate(classes):
            arff_data[idx].append(cl)

        return ngrams, arff_data, vectorizer, X

    def format_data_for_arff(self, ngrams, arff_data):
        """
        Formats given n-grams and arff data compatible with arff library
        :param ngrams: list, ngrams
        :param arff_data:
        :return: dict
        """

        # Example arff data
        # xor_dataset = {
        #     'description': 'XOR Dataset',
        #     'relation': 'XOR',
        #     'attributes': [
        #         ('input1', 'REAL'),
        #         ('input2', 'REAL'),
        #         ('y', 'REAL'),
        #     ],
        #     'data': [
        #         [0.0, 0.0, 0.0],
        #         [0.0, 1.0, 1.0],
        #         [1.0, 0.0, 1.0],
        #         [1.0, 1.0, 0.0]
        #     ]
        # }

        attributes = [(ngram, "NUMERIC") for ngram in ngrams]
        attributes.append((ARFF_FILE_TWEET_Y_NAME, ["negative", "neutral", "positive"]))

        formatted_data = {
            'description': ARFF_FILE_DESCRIPTION,
            'relation': ARFF_FILE_RELATION,
            'attributes': attributes,
            'data': arff_data
        }

        return formatted_data