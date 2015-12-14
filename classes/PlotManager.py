# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from config import *
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from helpers.GeneralHelpers import GeneralHelpers

class PlotManager:
    """
    This class does the necessary works for visualizing data
    """
    def __init__(self):
        self.__first_year = 2012
        self.__helper = GeneralHelpers()
        self.__colors = ['r', 'b', 'y', 'm', 'g', 'c', 'k']
        self.__years = ('2012', '2013', '2014', '2015')
        self.__regexp_for_predict_lines = "\d{1,}\s{1,}\d{1}:\w{1,8}.{1,}"


    def plot_years_scores_from_root_directory(self, root_dir):
        """
        Plots years' scores for given classifiers and mean of them
        :param root_dir: string, root directory to scan
        :return: void
        """
        bar_width = 0.10

        # Getting scores from helper
        years_classifier_scores_list = []
        years_classifier_scores_dict = self.__helper.get_accuracy_scores_for_years_from_root_dir(root_dir)

        # Making them lists
        for year, classifiers_scores in years_classifier_scores_dict.iteritems():
            years_classifier_scores_list.append(classifiers_scores.values())

        years_classifier_scores_list = np.array(years_classifier_scores_list)
        classifier_names = years_classifier_scores_dict['2012'].keys()

        indexes = np.arange(len(years_classifier_scores_dict.keys()))  # [0,1,2,3] +

        # Iterating over J48 for 2012, 2013, 2014, 2015, MEAN for 2012, 2013, 2014, 2015 an so on..
        for iteration_number, (color_name, classifier_name, classifier_scores) in enumerate(zip(self.__colors, classifier_names, years_classifier_scores_list.T)):
            bar_offset = indexes + (iteration_number * bar_width)
            plt.bar(bar_offset, classifier_scores, bar_width, color=color_name, label=classifier_name)

        plt.xlabel('Years')
        plt.ylabel('Scores %')
        plt.title('Scores by year and classifier(' + MODEL_NAME + ', CV=4)')
        plt.xticks(indexes + bar_width, self.__years)
        plt.legend(loc=4)
        plt.show()

    def plot_2012_vs_rest(self, root_dir):
        """
        Plots results of classifications of using 2012 as train set, 2013, 2014, 2015 as test set.
        :param root_dir: string
        :return: void
        """
        all_accuracy_scores = self.__helper.get_log_files_stats(root_dir)
        """
        Example accuracy scores:
        {
            'SMO':{
                2013: [62.79, 66.67, 50.0, 70.45, 57.14, 60.0, 64.29, 66.67, 62.79, 73.17, 57.14, 66.67],
                2014: [65.45, 58.97, 54.35, 72.09, 47.62, 66.67, 71.43, 66.67, 57.78, 64.44, 71.43, 59.26],
                2015: [62.79, 57.14, 63.16, 62.5, 59.26, 67.27, 61.76, 66.67, 68.63]
            },
            'IB1': {
                2013: [37.21, 40.48, 38.1, 43.18, 45.24, 47.5, 45.24, 35.71, 32.56, 24.39, 28.57, 51.28],
                2014: [38.18, 43.59, 41.3, 39.53, 47.62, 50.0, 33.33, 44.44, 26.67, 24.44, 40.48, 37.04],
                2015: [37.21, 41.07, 43.86, 39.29, 51.85, 30.91, 39.71, 26.32, 45.1]
            }
            ...
            ...
            ...
        }
        """
        self._plot_2012_vs_rest_monthly(all_accuracy_scores)
        self._plot_2012_vs_rest_yearly(all_accuracy_scores)

    def plot_top_feature_frequencies_in_years(self, years_features_counts):
        """
        Plots top features' frequencies in years
        :return: void
        """

        plot_feature_counts = {}
        bar_width = 0.20

        for feature_name in INFO_GAIN_ATTRIBUTES:
            if not feature_name in plot_feature_counts:
                plot_feature_counts[feature_name] = []

            f_key = feature_name.decode('utf-8')

            for year in self.__years:
                if not f_key in years_features_counts[year]:
                    years_features_counts[year][f_key] = 0

            plot_feature_counts[feature_name] = [years_features_counts["2012"][f_key],
                                                  years_features_counts["2013"][f_key],
                                                  years_features_counts["2014"][f_key],
                                                  years_features_counts["2015"][f_key]
                                                ]
        print(plot_feature_counts)

        indexes = np.arange(len(plot_feature_counts.keys()))

        for first_iteration_number, (feature_name, feature_counts) in enumerate(plot_feature_counts.iteritems()):
            for second_iteration_number, (color, feature_count) in enumerate(zip(self.__colors, feature_counts)):
                x_coord = first_iteration_number + (second_iteration_number*bar_width)
                plt.bar(x_coord, feature_count, bar_width, color=color)

        xticks = [key.decode('utf-8') for key in plot_feature_counts.keys()]

        plt.xlabel('Features')
        plt.ylabel('Frequencies in __years')
        plt.title('InfoGain features by year and features(' + MODEL_NAME + ')')
        plt.xticks(indexes + bar_width*2, xticks)

        handles = []
        for idx, (year, color) in enumerate(zip(self.__years, years_colors)):
            patch = Patch(color=color, label=year)
            handles.append(patch)

        plt.legend(loc=1, handles=handles)
        plt.show()

    def plot_years_intersection_scores(self, years_features_counts):
        """
        Plots a matrix which shows years' vocabularies similarities
        :param years_features_counts: dict
        :return: void
        """

        years_intersection_scores = np.zeros((len(self.__years),len(self.__years)))
        feature_frequencies = years_features_counts

        for first_iteration_number, (x_year, x_years_features) in enumerate(feature_frequencies.iteritems()):
            features_of_x = x_years_features.keys()
            total_count = np.sum(x_years_features.values())

            for second_iteration_number, (y_year, y_years_features) in enumerate(feature_frequencies.iteritems()):

                if x_year == y_year:
                    pass

                else:
                    features_of_y = y_years_features.keys()
                    intersect = list(set(features_of_x) & set(features_of_y))


                    intersect_count = 0
                    for intersect_item in intersect:
                        intersect_count = intersect_count + y_years_features[intersect_item]

                    ratio = float(intersect_count)/total_count

                    i_index = int(x_year) - self.__first_year #0
                    j_index = int(y_year) - self.__first_year #1
                    years_intersection_scores[i_index][j_index] = ratio


        all_scores_df = pd.DataFrame(years_intersection_scores, self.__years, self.__years)

        print(MODEL_NAME+'\'s __years\' vocabulary similarities:')
        print(all_scores_df)

    def plot_experiments_results_with_scikit_learn(self, lines_scores):
        """
        Plots experiments' results from scikit learn
        :param lines_scores: dict
        :return: void
        """
        test_years = ['13', '14', '15']
        markers = ['o','D','h','*','+']
        plot_types = ['-','--','-.',':', ',']
        legend_line_names = {
            'line1':'LINE1',
            'line2':'LINE2',
            'line3L0':'LINE3-RF DB',
            'line3L1':'LINE3-kMEANS CLUSTERING',
            'line3L2':'LINE3-kMEANS CLUSTERING(probabilities)',
            'line4':'LINE4'
        }
        # -(2012-500)/(YEAR-300)
        # -(2012-500)+(YEAR-R50)/(YEAR-300)
        # -(2012-500)+(YEAR-L50)/(YEAR-300)
        # -(2012-500)+(YEAR-L50)/(YEAR-300)
        # -(2012-500)+(YEAR-200)/(YEAR-300)
        fig, ax = plt.subplots(figsize=(20,9))
        ax.set_autoscale_on(False)
        ax.set_xlim([12.5,15.5])

        all_of_min = 100
        all_of_max = 0

        handles = []
        color_index = 0
        for first_iteration_number, (line_name, line_points) in enumerate(lines_scores.iteritems()):
            line_max, line_min = 0, 100

            if line_name == "line2":
                line_points_array = np.array(line_points.values())
                ys = line_points_array[:,1]
                mins = line_points_array[:,0]
                maxs = line_points_array[:,2]
                line_max, line_min = np.max(maxs), np.min(mins)

                for sub_iteration_number, (a_min, a_max) in enumerate(zip(mins, maxs)):
                    ax.plot((int(test_years[sub_iteration_number])-0.05,int(test_years[sub_iteration_number])+0.05),(a_min, a_min),'k-')
                    ax.plot((int(test_years[sub_iteration_number])-0.05,int(test_years[sub_iteration_number])+0.05),(a_max, a_max),'k-')
                    ax.plot((int(test_years[sub_iteration_number]),int(test_years[sub_iteration_number])),(a_min, a_max),'k-')

                ax.plot(test_years, ys, self.__colors[color_index], marker= markers[first_iteration_number], linestyle=plot_types[first_iteration_number], linewidth=3.0)
                patch = Patch(color=self.__colors[color_index], label=legend_line_names[line_name])
                color_index+=1
                handles.append(patch)

            elif line_name == "line3":
                for sub_iteration_number, (ale_experiment_key) in enumerate(ALE_LINE3_KEYS):
                    proper_dict_values = [line_points[dict_key] for dict_key in line_points.keys() if dict_key.startswith(ale_experiment_key)]
                    ys = proper_dict_values

                    line_max, line_min = np.max(ys), np.min(ys)
                    ax.plot(test_years, ys, self.__colors[color_index], marker= markers[first_iteration_number], linestyle=plot_types[first_iteration_number], linewidth=3.0)
                    patch = Patch(color=self.__colors[color_index], label=legend_line_names[line_name+ale_experiment_key])
                    handles.append(patch)
                    color_index+=1
            else:
                ys = line_points.values()
                line_max, line_min = np.max(ys), np.min(ys)
                ax.plot(test_years, ys, self.__colors[color_index], marker= markers[first_iteration_number], linestyle=plot_types[first_iteration_number], linewidth=3.0)
                patch = Patch(color=self.__colors[color_index], label=legend_line_names[line_name])
                handles.append(patch)
                color_index+=1
            all_of_min = min(line_min, all_of_min)
            all_of_max = max(line_max, all_of_max)


        ymin = all_of_min-0.01
        ymax = all_of_max+0.01

        plt.legend(handles=handles)

        ax.set_ylim([ymin, ymax])
        plt.yticks(np.arange(ymin, ymax, 0.01))
        ax.set_xticklabels(["","13","","14","","15"])
        plt.xlabel('Years')
        plt.ylabel('Scores %')
        plt.title('Scores by year with changing training sets. Classifier=RF Feature=Word.')
        plt.tight_layout()
        plt.grid()
        plt.show()

    def _plot_2012_vs_rest_monthly(self, all_accuracy_scores):
        """
        Plots 2012 vs REST graphic in monthly basis.
        :param all_accuracy_scores: dict
        :return: void
        """

        date_ranges = pd.date_range(start='1/1/2013', periods=33, freq='M')
        date_ranges = np.array([date_obj.strftime('%b-%y') for date_obj in date_ranges])

        xs = date_ranges

        for iteration_number, classifier_scores in enumerate(all_accuracy_scores.values()):
            ys = []
            fig = plt.figure(iteration_number)
            for year, year_scores in classifier_scores.iteritems():
                ys += year_scores

            xs = np.arange(1, 34, 1)

            plt.xlabel("Months")
            plt.ylabel("Scores%")
            plt.title(all_accuracy_scores.keys()[iteration_number])
            plt.plot(xs, ys)

    def _plot_2012_vs_rest_yearly(self, all_accuracy_scores):
        """
        Plots 2012 vs REST graphic in yearly basis.
        :param all_accuracy_scores: dict
        :return: void
        """
        date_ranges = pd.date_range(start='1/1/2013', periods=3, freq='365D')
        date_ranges = np.array([date_obj.strftime('%y') for date_obj in date_ranges])

        xs = date_ranges
        yearly_scores = {}


        fig, ax = plt.subplots()
        names_of_classifiers = all_accuracy_scores.keys()

        for iteration_number, classifier_scores in enumerate(all_accuracy_scores.values()):
            ys = []
            for year, year_scores in classifier_scores.iteritems():
                ys.append(np.mean(year_scores))

            plt.xlabel('Years')
            plt.ylabel('Scores %')
            plt.title('Scores by year and classifier(' + MODEL_NAME + ', train=2012, test=2013, 2014, 2015)')
            ax.set_xticklabels(xs)
            plt.xticks(rotation=90)
            ax.plot(xs, ys, self.__colors[iteration_number], label=names_of_classifiers[iteration_number])
            plt.legend()
        plt.show()