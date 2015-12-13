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
        self.__helper = GeneralHelpers()
        self.__colors = ['r', 'b', 'y', 'm', 'g', 'c', 'k']
        self.__years = ('2012', '2013', '2014', '2015')

    def plot_years_scores_from_root_directory(self, root_dir):
        """
        Plots __years' scores for given classifiers and mean of them
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