import traceback
from config import *
from random import randint

import numpy as np

from scipy import *
from scipy import sparse

import xgboost as xgb

from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import *
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


class ExperimentManager:
    """
    This class consists core methods of an active learning experiment.
    """

    def __init__(self, experiment_number, years_tweets_counts, n=1, analyzer='word'):
        """
        Constructor method
        :param experiment_number: int
        :param years_tweets_counts: dict
        :param n: int
        :param analyzer: string
        :return: ExperimentManager
        """

        self.__n = n
        self.__all_scores = {}
        self.__feature_count = 0
        self.__analyzer = analyzer
        self.__experiment_number = experiment_number
        self.__years_tweets_counts= years_tweets_counts

        self.__label_encoder = preprocessing.LabelEncoder()
        self.__label_encoder.fit(SENTIMENT_CLASSES)

    def run_experiment(self, document, classes):
        """
        Main method for using resources and making method calls in order.
        :param document: list
        :param classes: list
        :return: dict
        """
        try:
            # Fitting document
            X_sparse, features = self._fit_document(document)

            self.__feature_count = len(features)
            self.__features = features

            # Getting in ndarray format
            X = X_sparse.toarray()

            # Splitting document for __years
            years_X, years_X_sparse, years_y = self._split_dataset_to_years(X, X_sparse, classes)

            # Shuffling them for the experiment
            years_X, years_X_sparse, years_y = self._shuffle_years(years_X, years_X_sparse, years_y)

            # Creating 500, 300, 200 and 50 chunks of data
            partitioned_X_sparse, partitioned_y = self._create_years_partitions(years_X_sparse, years_y)

            # Iterating over lines' setups dict
            self._go_over_lines_setups(partitioned_X_sparse, partitioned_y)

            # Now let's cumulate line2's scores
            self._cumulate_scores_of_line2()

        except Exception:

            print("Exception in experiment:#"+str(self.__experiment_number))
            traceback.print_exc()

        return self.__all_scores

    def _fit_document(self, document):
        """
        Fits document and generates n-grams and features
        :param document: list
        :return: csr_matrix, list
        """
        vectorizer = CountVectorizer(ngram_range=(self.__n, self.__n), analyzer=self.__analyzer)
        X = vectorizer.fit_transform(document)
        features = vectorizer.get_feature_names()
        return X, features

    def _split_dataset_to_years(self, X, X_sparse, y):
        """
        Splits dataset to each year respectively
        :param X: dense data
        :param X_sparse: scipy.sparse
        :param y: list
        :return: dense matrice, scipy.sparse, list
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

    def _shuffle_years(self, years_X, years_X_sparse, years_y):
        """
        Shuffles __years' tweets
        :param years_X: dense data
        :param years_X_sparse: scipy.sparse
        :param years_y: list
        :return: dense matrice, scipy.sparse, list
        """
        for year_name, year_data in years_X_sparse.iteritems():
            normal = years_X[year_name]
            sparse = years_X_sparse[year_name]
            labels = years_y[year_name]
            years_X[year_name], years_X_sparse[year_name], years_y[year_name] = shuffle(normal, sparse, labels)

        return years_X, years_X_sparse, years_y

    def _create_years_partitions(self, years_X, years_y):
        """
        Creates partitions of each year
        :param years_X: scipy.sparse
        :param years_y: list
        :return: scipy.sparse, list
        """
        splitted_X = {}
        splitted_y = {}

        for (year_X, year_X_data), (year_y, year_y_data) in zip(years_X.iteritems(), years_y.iteritems()):

            splitted_X[year_X] = {}
            splitted_y[year_y] = {}

            if not year_X in TEST_YEARS and not year_y in TEST_YEARS:
                splitted_X[year_X][ALE_PARTITION_500_KEY] = year_X_data
                splitted_y[year_y][ALE_PARTITION_500_KEY] = year_y_data

            else:
                start_index = 0
                end_index = year_X_data.shape[0]

                splitted_X[year_X][ALE_PARTITION_300_KEY] = year_X_data[start_index:start_index+800]
                splitted_X[year_X][ALE_PARTITION_200_KEY] = year_X_data[start_index+800:end_index]
                splitted_X[year_X][ALE_PARTITION_50_KEY]  = []

                splitted_y[year_y][ALE_PARTITION_300_KEY] = year_y_data[start_index:start_index+800]
                splitted_y[year_y][ALE_PARTITION_200_KEY] = year_y_data[start_index+800:end_index]
                splitted_y[year_y][ALE_PARTITION_50_KEY]  = []

        for test_year in TEST_YEARS:

            for j in range(0,LINE2_RANDOM_ITERATION_NUMBER):
                random_X_set = []
                random_y_set = []

                test_year_200_length = splitted_X[test_year][ALE_PARTITION_200_KEY].shape[0]
                random.seed()
                random_start_index = randint(0, test_year_200_length-RANDOM_SAMPLE_SIZE)
                random_end_index = random_start_index+RANDOM_SAMPLE_SIZE

                random_X_set = splitted_X[test_year][ALE_PARTITION_200_KEY][random_start_index:random_end_index]
                random_y_set = splitted_y[test_year][ALE_PARTITION_200_KEY][random_start_index:random_end_index]

                splitted_X[test_year][ALE_PARTITION_50_KEY].append(random_X_set)
                splitted_y[test_year][ALE_PARTITION_50_KEY].append(random_y_set)

        return splitted_X, splitted_y

    def _go_over_lines_setups(self, years_X_sparse, years_y):
        """
        Iterates over LINES_SETUPS dictionary to run classifications
        :param years_X_sparse: scipy.sparse
        :param years_y: list
        :return: void
        """

        # Iterating over lines
        for line_name, line_value in LINES_SETUPS.iteritems():
            print('Currently running on Experiment #'+str(self.__experiment_number)+', '+line_name)
            self.__all_scores[line_name] = {}

            # Iterating over each setup( say 500-2012, 200-2013 / 300-2013)
            for iteration_index, (train_set_setup, test_set_setup) in enumerate(zip(line_value['train'],
                                                                                    line_value['test'])):

                if line_name == "line1" or line_name == "line4":

                    X_train, X_test, y_train, y_test, train_set_name, test_set_name = \
                            self._create_train_and_test_sets_from_setup_dict(years_X_sparse, years_y, train_set_setup, test_set_setup, line_name, -1)

                    acc_score = self._classify(X_train, X_test, y_train, y_test)
                    self._save_accuracy_score(line_name, train_set_name, test_set_name, acc_score)

                elif line_name == "line2":

                    for random_50_iteration_index in range(0, LINE2_RANDOM_ITERATION_NUMBER):

                        X_train, X_test, y_train, y_test, train_set_name, test_set_name = \
                        self._create_train_and_test_sets_from_setup_dict(years_X_sparse, years_y, train_set_setup,
                                                                         test_set_setup, line_name, random_50_iteration_index)

                        acc_score = self._classify(X_train, X_test, y_train, y_test)
                        self._save_accuracy_score(line_name, train_set_name, test_set_name, acc_score)

                elif line_name == "line3":
                    # Find train set and test set - preperation
                    probability_setup = LINE3_PROB_SETUP[iteration_index]
                    prob_train_setup = probability_setup['train']
                    prob_test_setup = probability_setup['test']

                    prob_train_year, prob_train_count = prob_train_setup[0], prob_train_setup[1]
                    prob_test_year, prob_test_count = prob_test_setup[0], prob_test_setup[1]

                    prob_X_train = years_X_sparse[prob_train_year][prob_train_count]
                    prob_y_train = years_y[prob_train_year][prob_train_count]

                    prob_X_test = years_X_sparse[prob_test_year][prob_test_count]
                    prob_y_test = np.array(years_y[prob_test_year][prob_test_count])

                    final_X_test_year = test_set_setup.keys()[0] #2013
                    final_X_test_tweet_count = test_set_setup[final_X_test_year] #300

                    final_X_test = years_X_sparse[final_X_test_year][final_X_test_tweet_count]
                    final_y_test = years_y[final_X_test_year][final_X_test_tweet_count]

                    test_set_name = final_X_test_year + "_" + final_X_test_tweet_count
                    train_set_name_appendix = prob_train_year + "_" + prob_train_count + "+" + prob_test_year + "_" + str(MOST_DISTINCT_SAMPLE_SIZE)


                    #Active Learning Method - I
                    print('Active Learning Method - I')
                    samples_closest_to_decision_boundary_X, samples_closest_to_decision_boundary_y, indexes_of_samples_closest_to_decision_boundary = \
                        self._choose_ale_samples_closest_to_decision_boundary(prob_X_train, prob_X_test, prob_y_train, prob_y_test, MOST_DISTINCT_SAMPLE_SIZE)

                    acc_score_for_ale_one = self._combine_train_sets_and_run_classification(prob_X_train, final_X_test,
                                                                                            samples_closest_to_decision_boundary_X,
                                                                                            prob_y_train, final_y_test,
                                                                                            samples_closest_to_decision_boundary_y)

                    train_set_name_one = "L0-" + train_set_name_appendix
                    self._save_accuracy_score(line_name, train_set_name_one, test_set_name, acc_score_for_ale_one)

                    if PLOT_DECISION_BOUNDARIES_FOR_LINE_3:
                        # Plot decision boundary of probabilities with PCA
                        plot_title = train_set_name_one + '/' + test_set_name
                        self._plot_decision_boundary(prob_X_train, prob_X_test, prob_y_train, prob_y_test,
                                                     indexes_of_samples_closest_to_decision_boundary, plot_title)


                    # Active Learning Method - II
                    print('Active Learning Method - II')
                    samples_closest_to_cluster_centroids_org_X, samples_closest_to_cluster_centroids_org_y, indices_of_closest_samples_to_centroids = \
                        self._choose_ale_samples_from_cluster_centroids_with_original_features(prob_X_test, prob_y_test)

                    acc_score_for_ale_two = self._combine_train_sets_and_run_classification(prob_X_train, final_X_test,
                                                                                            samples_closest_to_cluster_centroids_org_X,
                                                                                            prob_y_train, final_y_test,
                                                                                            samples_closest_to_cluster_centroids_org_y)

                    train_set_name_two = "L1-" + train_set_name_appendix
                    self._save_accuracy_score(line_name, train_set_name_two, test_set_name, acc_score_for_ale_two)


                    # Active Learning Method - III
                    print('Active Learning Method - III')
                    probabilities = self._predict_probabilities(prob_X_train, prob_X_test, prob_y_train)

                    samples_closest_to_cluster_centroids_cmb_X, samples_closest_to_cluster_centroids_cmb_y, indices_of_closest_samples_to_centroids = \
                        self._choose_ale_samples_from_cluster_centroids_with_combined_features(probabilities, prob_X_test, prob_y_test)

                    acc_score_for_ale_three = self._combine_train_sets_and_run_classification(prob_X_train, final_X_test,
                                                                                            samples_closest_to_cluster_centroids_cmb_X,
                                                                                            prob_y_train, final_y_test,
                                                                                            samples_closest_to_cluster_centroids_cmb_y)

                    train_set_name_three = "L2-" + train_set_name_appendix
                    self._save_accuracy_score(line_name, train_set_name_three, test_set_name, acc_score_for_ale_three)



                    # Active Learning Method - IV
                    print('Active Learning Method - IV')
                    it_X_train, it_X_test, it_y_train, it_y_test = \
                        self._choose_ale_samples_closest_to_decision_boundary_with_iteration(prob_X_train, prob_X_test, prob_y_train, prob_y_test)
                    acc_score_for_ale_four = self._classify(it_X_train, final_X_test, it_y_train, final_y_test)
                    train_set_name_three = "L3-" + train_set_name_appendix
                    self._save_accuracy_score(line_name, train_set_name_three, test_set_name, acc_score_for_ale_four)

    def _create_train_and_test_sets_from_setup_dict(self, years_X_sparse, years_y, train_setup, test_setup, line_name, iteration_number):
        """
        Creates necessary train set and test set from given setup dictionary
        :param years_X_sparse: scipy.sparse
        :param years_y: list
        :param train_setup: dict
        :param test_setup: dict
        :param line_name: string
        :param iteration_number: int
        :return: scipy.sparse, scipy.sparse, list, list, string, string
        """

        X_train = []
        y_train = []

        X_test  = []
        y_test  = []

        train_set_name = ""
        test_set_name = ""

        # Train setup and test setup may have different number of elements so we can't zip() and iterate simultaniously

        for train_set_year, tweet_to_take_from_train_year in train_setup.iteritems():

            new_x_train = years_X_sparse[train_set_year][tweet_to_take_from_train_year]

            if line_name == "line2" and isinstance(new_x_train, list) and len(new_x_train)==LINE2_RANDOM_ITERATION_NUMBER:
                new_x_train = new_x_train[iteration_number]

            new_x_train_dense = new_x_train.toarray().tolist()
            X_train += new_x_train_dense

            new_y_train = years_y[train_set_year][tweet_to_take_from_train_year]

            if line_name == "line2" and isinstance(y_train, list) and len(new_y_train)==LINE2_RANDOM_ITERATION_NUMBER:
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
        Makes a classification with given train and test sets
        :param X_train: scipy.sparse
        :param X_test: scipy.sparse
        :param y_train: list
        :param y_test: list
        :return: float
        """
        # Creating SVM instance
        classifier = self._get_new_model_for_classification()

        y_train = self.__label_encoder.transform(y_train)
        y_test  = self.__label_encoder.transform(y_test)

        # Fitting model
        classifier.fit(X_train, y_train)

        # Predicting
        predictions = classifier.predict(X_test)

        # Getting accuracy score
        acc_score = accuracy_score(y_test, predictions)
        #acc_score = round(acc_score, 3)

        return acc_score

    def _save_accuracy_score(self, line_name, train_set_name, test_set_name, score):
        """
        Saves given accuracy score to appropriate key
        :param line_name: string
        :param point_name: string
        :param score: float
        :return: void
        """
        score_dict_key = train_set_name + '/' + test_set_name

        if line_name == "line2":
            if not score_dict_key in self.__all_scores[line_name]:
                self.__all_scores[line_name][score_dict_key] = []
            self.__all_scores[line_name][score_dict_key].append(score)
        else:
            self.__all_scores[line_name][score_dict_key] = score

    def _cumulate_scores_of_line2(self):
        """
        Cumulates line2's scores from LINE2_RANDOM_ITERATION_NUMBER experiments into an array like: [min, mean, max]
        :return: void
        """
        # Iterating over line2's scores
        for train_test_set, scores_list in self.__all_scores['line2'].iteritems():

            min_ = np.min(scores_list)
            mean_= np.mean(scores_list)
            max_ = np.max(scores_list)

            #min_, mean_, max_ = round(min_, 3), round(mean_, 3), round(max_, 3),

            min_mean_max = []

            for m in (min_, mean_, max_):
                min_mean_max.append(m)

            self.__all_scores['line2'][train_test_set] = min_mean_max

    def _predict_probabilities(self, X_train, X_test, y_train):
        """
        This method calculates the probabilities of test set samples belogning each sentiment class using a model.

        :param X_train: scipy.sparse
        :param X_test: scipy.sparse
        :param y_train: list
        :param y_test: list
        :return: list
        """
        # Getting new model instance
        classifier = self._get_new_model_for_logical_selection_with_classification()

        # Fitting
        classifier.fit(X_train, y_train)

        # Getting probabilities
        probabilities = classifier.predict_proba(X_test)

        return probabilities

    def _get_new_model_for_classification(self):
        """
        Returns new classifier instance
        :return: OneVsRestClassifier
        """
        model_for_classification = SVC(C=1.0, kernel='poly', probability=True, degree=1.0, cache_size=250007)
        model_for_classification = OneVsRestClassifier(model_for_classification)
        #model_for_classification = RandomForestClassifier(n_estimators=100)
        #model_for_classification = GradientBoostingClassifier(n_estimators=100)
        #model_for_classification = GaussianNB()
        #model_for_classification = xgb.XGBClassifier(n_estimators=100)

        return model_for_classification

    def _get_new_model_for_logical_selection_with_classification(self):
        """
        Returns new classifier instance for logical selection
        :return:
        """
        model_for_logical_selection_with_classification = SVC(C=1.0, kernel='poly', probability=True, degree=1.0, cache_size=250007)
        model_for_logical_selection_with_classification = OneVsRestClassifier(model_for_logical_selection_with_classification)
        #model_for_logical_selection_with_classification = RandomForestClassifier(n_estimators=100)
        #model_for_logical_selection_with_classification = GradientBoostingClassifier(n_estimators=100)
        #model_for_logical_selection_with_classification = MultinomialNB()
        #model_for_logical_selection_with_classification = xgb.XGBClassifier(n_estimators=100)

        return model_for_logical_selection_with_classification

    def _get_new_model_for_logical_selection_with_clustering(self):
        """
        Returns new model for clustering in active learning experiment II and III
        :return: KMeans
        """
        model_for_logical_selection_with_clustering = KMeans(n_clusters = MOST_DISTINCT_SAMPLE_SIZE)

        return model_for_logical_selection_with_clustering

    def _get_sample_indexes_closest_to_decision_boundary(self, samples_probabilities, sample_size):
        """
        Returns samples' indexes which are closest to decision boundary
        :param samples: list
        :return: list
        """
        # Finding elements which have minimum standart deviations
        np_array = self._find_standart_deviations_of_samples_probabilities(samples_probabilities)

        indices_of_minimum_stds = np_array.argsort()[:sample_size]
        # for indice in indices_of_minimum_stds:
        #     print(samples_probabilities[indice], standart_deviations[indice])
        return indices_of_minimum_stds

    def _plot_decision_boundary(self, X_train, X_test, y_train, y_test, highlighted_samples_indexes, plot_title):
        """
        Plots decision boundary of a point in line3 using dimensionality reduction with PCA(TruncatedSVD)
        :param X_train: scipy.sparse
        :param X_test: scipy.sparse
        :param y_train: list
        :param y_test: list
        :param highlighted_samples_indexes: list
        :param plot_title: string
        :return: void
        """

        # Creating classifiers
        classifier = self._get_new_model_for_classification()
        svd = TruncatedSVD(n_components=2)

        # Encoding labeles to float
        y_train = self.__label_encoder.transform(y_train)
        y_test = self.__label_encoder.transform(y_test)

        # Splitting normal and highlighted samples
        probability_ranges = np.arange(X_test.shape[0])
        normal_samples_indexes = np.setdiff1d(probability_ranges, highlighted_samples_indexes)

        normal_samples_y = y_test[normal_samples_indexes]
        highlighted_samples_y = y_test[highlighted_samples_indexes]

        # Dimensionality reduction
        svd_X_train = svd.fit_transform(X_train)
        svd_X_test = svd.fit_transform(X_test)

        # Training the model
        classifier.fit(svd_X_train, y_train)

        x_min= min(svd_X_train[:, 0].min(), svd_X_test[:, 0].min()) - 2
        x_max = max(svd_X_train[:, 0].max(), svd_X_test[:, 0].max()) + 2

        y_min= min(svd_X_train[:, 1].min(), svd_X_test[:, 1].min()) - 2
        y_max = max(svd_X_train[:, 1].max(), svd_X_test[:, 1].max()) + 2


        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        plt.figure(110)

        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Train set
        #plt.scatter(svd_X_train[:, 0], svd_X_train[:, 1], c=y_train, cmap=plt.cm.Paired)

        # 150 samples from test set those are normal samples
        plt.scatter(svd_X_test[normal_samples_indexes][:, 0], svd_X_test[normal_samples_indexes][:, 1], c=normal_samples_y, cmap=plt.cm.Paired, s=5)

        # 50 samples from test set those are chosen
        plt.scatter(svd_X_test[highlighted_samples_indexes][:, 0], svd_X_test[highlighted_samples_indexes][:, 1], c=highlighted_samples_y, cmap=plt.cm.Paired, marker='D', s=50)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        info_text = "Reduced to 2 dims from %s dims. Dataset: %s\n" \
                    "Setup: %s. Explained Variance: %s" % (str(self.__feature_count), MODEL_NAME, plot_title, str(svd.explained_variance_ratio_.sum()))

        plt.title("PCA Projection on Decision Boundary of SVM.")

        plt.text(xx.mean()+2, yy.max()-0.5,  info_text, size=12, rotation=0.,
         ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

        plt.show()

    def _choose_ale_samples_closest_to_decision_boundary(self, prob_X_train, prob_X_test, prob_y_train, prob_y_test, sample_size):
        """
        Returns closest samples to the decision boundary.
        :param prob_X_train: scipy.sparse
        :param prob_X_test: scipy.sparse
        :param prob_y_train: list
        :param prob_y_test: list
        :return: scipy.sparse, list, list
        """

        probabilities = self._predict_probabilities(prob_X_train, prob_X_test, prob_y_train)

        # Find closest samples to the decision boundary
        indexes_of_samples_closest_to_decision_boundary = self._get_sample_indexes_closest_to_decision_boundary(probabilities, sample_size)

        samples_closest_to_decision_boundary_X = prob_X_test[indexes_of_samples_closest_to_decision_boundary]
        samples_closest_to_decision_boundary_y = prob_y_test[indexes_of_samples_closest_to_decision_boundary]

        return samples_closest_to_decision_boundary_X, samples_closest_to_decision_boundary_y, indexes_of_samples_closest_to_decision_boundary

    def _choose_ale_samples_closest_to_decision_boundary_with_iteration(self, prob_X_train, prob_X_test, prob_y_train, prob_y_test):
        """

        :param prob_X_train:
        :param prob_X_test:
        :param prob_y_train:
        :param prob_y_test:
        :return:
        """

        iteration_X_train = prob_X_train
        iteration_X_test = prob_X_test

        iteration_y_train = prob_y_train[:]
        iteration_y_test = prob_y_test[:]

        for i in range(0, LINE3_CHOOSING_SAMPLES_ITERATION_COUNT):

            samples_X, samples_y, indexes = self._choose_ale_samples_closest_to_decision_boundary(iteration_X_train,
                                                                                                  iteration_X_test,
                                                                                                  iteration_y_train,
                                                                                                  iteration_y_test,
                                                                                                  LINE3_CHOOSING_SAMPLES_SIZE)

            dense_X_train = iteration_X_train.toarray().tolist()
            dense_X_train += samples_X.toarray().tolist()
            iteration_X_train = sparse.csr_matrix(dense_X_train)

            iteration_y_train += samples_y.tolist()

            iteration_y_test = np.delete(iteration_y_test, indexes)
            dense_X_test = iteration_X_test.toarray().tolist()
            dense_X_test = np.delete(dense_X_test, indexes, axis=0)


            iteration_X_test = sparse.csr_matrix(dense_X_test, shape=(len(iteration_y_test), iteration_X_train.shape[1]))

        return iteration_X_train, iteration_X_test, iteration_y_train, iteration_y_test

    def _choose_ale_samples_from_cluster_centroids_with_original_features(self, prob_X_test, prob_y_test):
        """
        Returns samples closest to the cluster centroids with original features.
        :param prob_X_test: scipy.sparse
        :param prob_y_test: scipy.sparse
        :return: scipy.sparse, list, list
        """
        indices_of_closest_samples_to_centroids = []
        clustering_model = self._get_new_model_for_logical_selection_with_clustering()
        clustering_model.fit(prob_X_test)
        distances_matrix = clustering_model.transform(prob_X_test)

        indices_of_closest_samples_to_centroids = np.argmin(distances_matrix, axis=0)

        samples_closest_to_centroids_X = prob_X_test[indices_of_closest_samples_to_centroids]
        samples_closest_to_centroids_y = prob_y_test[indices_of_closest_samples_to_centroids]

        return samples_closest_to_centroids_X, samples_closest_to_centroids_y, indices_of_closest_samples_to_centroids

    def _choose_ale_samples_from_cluster_centroids_with_combined_features(self, probabilities, prob_X_test, prob_y_test):
        """
        Returns samples closest to cluster centroids with three features(probabilities)
        :param probabilities: list
        :param prob_X_test: scipy.sparse
        :param prob_y_test: scipy.sparse
        :return: scipy.sparse, list, list
        """
        N_SAMPLES_WITH_MINIMUM_STDS_TO_CLUSTER = 100

        standart_deviations = self._find_standart_deviations_of_samples_probabilities(probabilities)
        indices_of_minimum_stds = standart_deviations.argsort()[:N_SAMPLES_WITH_MINIMUM_STDS_TO_CLUSTER]

        clustering_model = self._get_new_model_for_logical_selection_with_clustering()
        clustering_model.fit(probabilities)
        distances_matrix = clustering_model.transform(probabilities)

        indices_of_closest_samples_to_centroids = np.argmin(distances_matrix, axis=0)

        samples_closest_to_centroids_X = prob_X_test[indices_of_closest_samples_to_centroids]
        samples_closest_to_centroids_y = prob_y_test[indices_of_closest_samples_to_centroids]

        return samples_closest_to_centroids_X, samples_closest_to_centroids_y, indices_of_closest_samples_to_centroids

    def _combine_train_sets_and_run_classification(self, base_train_X, base_test_X, new_train_X, base_train_y, base_test_y, new_train_y):
        """
        Combines original train set with additional train set for active learning experiments. Then runs classification
        with final train and test sets.
        :param base_train_X: scipy.sparse
        :param base_test_X: scipy.sparse
        :param new_train_X: scipy.sparse
        :param base_train_y: list
        :param base_test_y: list
        :param new_train_y: list
        :return:
        """
        # Find final train and test set
        final_X_train = base_train_X.toarray().tolist()
        final_X_train += new_train_X.toarray().tolist()
        final_sparse_X_train = sparse.csr_matrix(final_X_train)

        final_y_train = []
        final_y_train = base_train_y[:]
        final_y_train += new_train_y.tolist()

        # Test model and save the score
        acc_score = self._classify(final_sparse_X_train, base_test_X, final_y_train, base_test_y)

        return acc_score

    def _find_standart_deviations_of_samples_probabilities(self, probabilities):
        """
        Returns standart deviations of given probabilities list to find samples closest to the decision boundary in one
        step further.
        :param probabilities: list
        :return: np.array
        """

        standart_deviations = []
        for one_samples_probabilities in probabilities:

            mean_of_samples_probabilities = np.std(one_samples_probabilities)
            standart_deviations.append(mean_of_samples_probabilities)

        # Finding elements which have minimum standart deviations
        np_standart_deviations = np.array(standart_deviations)

        return np_standart_deviations