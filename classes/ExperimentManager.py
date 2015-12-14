import sys
import numpy as np
from config import *
from random import randint


from scipy import *
from scipy import sparse
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier

class ExperimentManager:
    """

    """
    def __init__(self, years_tweets_counts, n=1, analyzer='word'):
        self.__n = n
        self.__analyzer = analyzer
        self.__years_tweets_counts= years_tweets_counts
        self.__label_encoder = preprocessing.LabelEncoder()
        self.__label_encoder.fit(SENTIMENT_CLASSES)
        self.__feature_count = 0
        self.__all_scores = {}

    def run_experiment(self, document, classes):
        """

        :param document:
        :param classes:
        :return:
        """
        # Fitting document
        X_sparse, features = self._fit_document(document)

        self.__feature_count = len(features)
        self.__features = features
        print("Feature count:"+str(self.__feature_count))

        # Getting in ndarray format
        X = X_sparse.toarray()

        # Splitting document for years
        years_X, years_X_sparse, years_y = self._split_dataset_to_years(X, X_sparse, classes)

        # Shuffling them for the experiment
        years_X, years_X_sparse, years_y = self._shuffle_years(years_X, years_X_sparse, years_y)

        # Creating 500, 300, 200 and 50 chunks of data
        partitioned_X_sparse, partitioned_y = self._create_years_partitions(years_X_sparse, years_y)

        # Iterating over lines' setups dict
        self._go_over_lines_setups(partitioned_X_sparse, partitioned_y)



    def _fit_document(self, document):
        """

        :param document:
        :return:
        """
        vectorizer = CountVectorizer(ngram_range=(self.__n, self.__n), analyzer=self.__analyzer)
        X = vectorizer.fit_transform(document)
        features = vectorizer.get_feature_names()
        return X, features

    def _split_dataset_to_years(self, X, X_sparse, y):
        """

        :param X:
        :return:
        """

        start_index = 0
        years_X = {}
        years_y = {}
        years_X_sparse = {}

        for year, tweet_count in self.__years_tweets_counts.iteritems():
            end_index = start_index + tweet_count
            years_X[year] = X[start_index:end_index]
            years_y[year] = y[start_index:end_index]
            years_X_sparse[year] = X_sparse[start_index:end_index]
            start_index += tweet_count

        return years_X, years_X_sparse, years_y

    def _create_years_partitions(self, years_X, years_y):
        """

        :param arff_data_for_years:
        :return:
        """
        splitted_X = {}
        splitted_y = {}

        for (year_X, year_X_data), (year_y, year_y_data) in zip(years_X.iteritems(), years_y.iteritems()):

            splitted_X[year_X] = {}
            splitted_y[year_y] = {}
            if not year_X in TEST_YEARS and not year_y in TEST_YEARS:
                splitted_X[year_X]['500'] = year_X_data
                splitted_y[year_y]['500'] = year_y_data

            else:
                start_index = 0
                end_index = year_X_data.shape[0]

                splitted_X[year_X]['300'] = year_X_data[start_index:start_index+300]
                splitted_X[year_X]['200'] = year_X_data[start_index+300:end_index]
                splitted_X[year_X]['50']  = []

                splitted_y[year_y]['300'] = year_y_data[start_index:start_index+300]
                splitted_y[year_y]['200'] = year_y_data[start_index+300:end_index]
                splitted_y[year_y]['50']  = []

        for test_year in TEST_YEARS:

            for j in range(0,10):
                random_X_set = []
                random_y_set = []

                test_year_200_length = splitted_X[test_year]['200'].shape[0]
                random.seed()
                random_start_index = randint(0, test_year_200_length-RANDOM_SAMPLE_SIZE)
                random_end_index = random_start_index+RANDOM_SAMPLE_SIZE

                random_X_set = splitted_X[test_year]['200'][random_start_index:random_end_index]
                random_y_set = splitted_y[test_year]['200'][random_start_index:random_end_index]

                splitted_X[test_year]['50'].append(random_X_set)
                splitted_y[test_year]['50'].append(random_y_set)

        return splitted_X, splitted_y

    def _shuffle_years(self, years_X, years_X_sparse, years_y):
        """

        :param years_X:
        :param years_y:
        :return:
        """
        for year_name, year_data in years_X_sparse.iteritems():
            normal = years_X[year_name]
            sparse = years_X_sparse[year_name]
            labels = years_y[year_name]
            years_X[year_name], years_X_sparse[year_name], years_y[year_name] = shuffle(normal, sparse, labels)

        return years_X, years_X_sparse, years_y

    def _go_over_lines_setups(self, years_X_sparse, years_y):
        """

        :param years_X_sparse:
        :param years_y:
        :return:
        """

        # Iterating over lines
        for line_name, line_value in LINES_SETUPS.iteritems():

            if line_name == "line1" or line_name == "line4":

                # Itearating over each setup( say 500-2012, 200-2013 / 300-2013)
                print(line_name)
                for iteration_index, (train_set_setup, test_set_setup) in enumerate(zip(line_value['train'],
                                                                                        line_value['test'])):

                    X_train, X_test, y_train, y_test, train_set_name, test_set_name = \
                        self._create_train_and_test_sets_from_setup_dict(years_X_sparse, years_y, train_set_setup, test_set_setup, line_name, -1)

                    acc_score, probabilities = self._classify(X_train, X_test, y_train, y_test)
                    print(train_set_name, test_set_name, acc_score)

            elif line_name == "line2":

                print(line_name)
                for setup_iteration_index, (train_set_setup, test_set_setup) in enumerate(zip(line_value['train'],
                                                                                              line_value['test'])):
                    for random_50_iteration_index in range(0, LINE2_RANDOM_ITERATION_NUMBER):

                        X_train, X_test, y_train, y_test, train_set_name, test_set_name = \
                        self._create_train_and_test_sets_from_setup_dict(years_X_sparse, years_y, train_set_setup, test_set_setup, line_name, random_50_iteration_index)

                        acc_score, probabilities = self._classify(X_train, X_test, y_train, y_test)
                        print(train_set_name, test_set_name, acc_score)


    def _create_train_and_test_sets_from_setup_dict(self, years_X_sparse, years_y, train_setup, test_setup, line_name, iteration_number):
        """

        :param years_X_sparse:
        :param years_y:
        :return:
        """
        # Old code
        # X_train_flag = 0
        # X_test_flag = 0
        #
        # if X_train_flag == 0:
        #     X_train = new_x_train
        #     X_train_flag = 1
        # else:
        #     X_train = sparse.vstack((new_x_train, X_train))
        # if X_test_flag == 0:
        #     X_test = new_x_test
        #     X_test_flag = 1
        # else:
        #     X_test = sparse.vstack((new_x_test, X_test))

        X_train = []
        y_train = []

        X_test  = []
        y_test  = []

        train_set_name = ""
        test_set_name = ""

        for train_set_year, tweet_to_take_from_train_year in train_setup.iteritems():

            if NOT_INCLUDE_YEARS_TWEETS == tweet_to_take_from_train_year:
                pass

            elif MOST_DISTINCT == tweet_to_take_from_train_year:
                pass

            else:
                new_x_train = years_X_sparse[train_set_year][tweet_to_take_from_train_year]

                if line_name == "line2" and isinstance(new_x_train, list) and len(new_x_train)==10:
                    new_x_train = new_x_train[iteration_number]

                new_x_train_dense = new_x_train.toarray().tolist()
                X_train += new_x_train_dense

                new_y_train = years_y[train_set_year][tweet_to_take_from_train_year]

                if line_name == "line2" and isinstance(y_train, list) and len(new_y_train)==10:
                    y_train += new_y_train[iteration_number]
                else:
                    y_train += new_y_train

                train_set_name += train_set_year + '_' + tweet_to_take_from_train_year + '+'

        X_train_sparse = sparse.csr_matrix(X_train)


        train_set_name = train_set_name.rstrip('+')

        for test_set_year, tweet_to_take_from_test_year in test_setup.iteritems():

            new_x_test = years_X_sparse[test_set_year][tweet_to_take_from_test_year]
            new_x_test_dense = new_x_test.toarray().tolist()
            X_test += new_x_test_dense

            y_test += years_y[test_set_year][tweet_to_take_from_test_year]

            test_set_name += test_set_year + '_' + tweet_to_take_from_test_year + '+'

        sparse_X_test = sparse.csr_matrix(X_test)
        test_set_name = test_set_name.rstrip('+')

        return X_train_sparse, sparse_X_test, y_train, y_test, train_set_name, test_set_name


    def _classify(self, X_train, X_test, y_train, y_test):
        """

        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :return:
        """
        # Creating SVM instance
        classifier = SVC(C=1.0, kernel='poly', probability=True, degree=1.0, cache_size=250007, tol=0.001, shrinking=False)
        classifier = OneVsRestClassifier(classifier)

        y_train = self.__label_encoder.transform(y_train)
        y_test  = self.__label_encoder.transform(y_test)

        # Fitting model
        classifier.fit(X_train, y_train)

        # Predicting
        predictions = classifier.predict(X_test)

        # Getting each sample's class probabilities
        probabilities = classifier.predict_proba(X_test)

        # Getting accuracy score
        acc_score = accuracy_score(y_test, predictions)

        return acc_score, predictions