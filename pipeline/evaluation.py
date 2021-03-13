import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import plot_confusion_matrix

from utils.utils import create_output_hierarchy
from utils.utils import save_to_file


def evaluate_classifiers(results, output, class_names, frequencies, models, k_fold):
    """
    Evaluates the scores of various metrics for each split of each classifier. Plots various
    charts to allow a better visualisation.

    :param results: dataframe of results
    :param output: root output directory
    :param class_names: list of class labels
    :param frequencies: list of frequencies
    :param models: list of models
    :param k_fold: number of fold in the cross-validation
    """

    # Create output hierarchy
    output_folder = create_output_hierarchy(output, frequencies, models)

    # Evaluates classifiers
    results = calculates_scores(results)

    # Plots charts
    plot_cnf_matrix(results, output_folder, class_names, k_fold)
    plot_baw(results, output_folder, frequencies, models, k_fold)
    plot_variation_over_frequency(results, output_folder, frequencies, models, k_fold)

    # Saves scores to file
    save_to_file(output_folder, results)


def calculates_scores(results):
    """
    Calculates the scores of various metrics for each split of each classifier.

    :param results: dataframe of results
    :return: dataframe of scores
    """

    scores = []

    # Evaluates each k-split of each classifier
    for i in results.index:
        y_test = results['y_test'][i]
        y_pred = results['y_pred'][i]

        # Calculates various metrics
        auroc = roc_auc_score(y_test, y_pred if y_pred.shape[1] > 2 else y_pred[:, 1], average='macro', multi_class='ovo')
        y_pred = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_test, y_pred)
        sp = specificity_score(y_test, y_pred)
        se = recall_score(y_test, y_pred, average='macro')
        pre = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Merge scores
        score = {'accuracy': acc, 'specificity': sp, 'sensitivity': se, 'precision': pre, 'f1': f1, 'auroc': auroc}
        scores.append(score)

    # return dataframe of scores
    scores = pd.DataFrame(scores)
    return pd.concat([results, scores], axis=1)


def plot_cnf_matrix(results, output_folder, class_names, k_fold):
    """
    Plots the confusion matrices for each k-fold of each classifier.

    :param results: dataframe of scores
    :param output_folder: output directory
    :param class_names: list of class labels
    :param k_fold: number of fold in the cross-validation
    """

    # Plots a chart for each k-split of each classifier
    for i in results.index:

        # Retrieves relevant results
        name = results['name'][i]
        abbreviation = results['abbreviation'][i]
        frequency = str(results['frequency'][i]) + 'Hz'
        clf = results['classifier'][i]
        x_test = results['x_test'][i]
        y_test = results['y_test'][i]

        # Creates and configures plot
        disp = plot_confusion_matrix(clf, x_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)
        disp.ax_.set_title(name + ": K-split " + str(i % k_fold + 1) + ' (' + frequency + ')')

        # Saves and shows figure
        save_folder = output_folder + '/plots/' + frequency + '/cnf/' + abbreviation
        file_location = save_folder + '/cnf_' + frequency + '_' + abbreviation + '_split_' + str(i % k_fold + 1) + '.png'
        plt.savefig(file_location)
        plt.show()


def plot_baw(results, output_folder, frequencies, models, k_fold, axes_ylim=None):
    """
    Plots box and whisker charts of each classifier and their cross-validation (k-split) . Creates one plot
    per metric per frequency which compares the performance of each model.

    :param results: dataframe of scores
    :param output_folder: output directory
    :param frequencies: list of frequencies
    :param models: list of classifiers
    :param k_fold: number of fold in the cross-validation
    :param axes_ylim: specific y-axis limits
    """

    # Plots a chart for each metric of each frequency
    for frequency in frequencies:
        for column in results.columns[8:]:

            # Retrieves relevant results
            result = results.loc[results['frequency'] == frequency][column]
            freq = str(frequency) + 'Hz'
            scores = pd.DataFrame(result.values.reshape(k_fold, -1, order='F'))

            # Creates and configures plot
            plt.boxplot(scores, labels=models)
            axes = plt.gca()
            if axes_ylim is not None:
                axes.set_ylim(axes_ylim)

            # Saves and shows figure
            save_folder = output_folder + '/plots/' + freq + '/baw/'
            file_location = save_folder + 'baw_' + column + '_' + freq + '.png'
            plt.savefig(file_location)
            plt.show()


def plot_variation_over_frequency(results, output_folder, frequencies, models, k_fold, axes_ylim=None):
    """
    Plot the metrics' variations over the frequencies. Creates one plot per metric which
    compares the performance of each model across the frequencies.

    :param results: dataframe of scores
    :param output_folder: output directory
    :param frequencies: list of frequencies
    :param models: list of classifiers
    :param k_fold: number of fold in the cross-validation
    :param axes_ylim: specific y-axis limits
    """

    markers = ['o', ',', 'd', 's', 'v']

    # Plots a chart for each metric
    for column in results.columns[8:]:
        result = results[column]

        # Calculates mean for each model
        scores = pd.DataFrame(result.values.reshape(k_fold, -1, order='F')).mean(axis=0)
        scores = pd.DataFrame(scores.values.reshape(len(frequencies), -1, order='C'))
        scores.columns = models
        scores.index = frequencies

        # Creates and configures plot
        plt.figure()
        scores.plot(kind='line', linestyle='-', style=markers[:len(models)])
        plt.xlabel('Sampling rate [Hz]')
        plt.ylabel(column.capitalize())
        plt.legend()
        plt.grid(axis='y')
        axes = plt.gca()
        if axes_ylim is not None:
            axes.set_ylim(axes_ylim)

        # Plot a line for the max average across models
        peak_idx = scores.mean(axis=1).idxmax()
        plt.axvline(peak_idx, color='grey', linewidth=2, linestyle='--')

        # Saves and shows figure
        save_folder = output_folder + '/plots/'
        file_location = save_folder + 'freq_' + column + '.png'
        plt.savefig(file_location)
        plt.show()


def specificity_score(y_test, y_pred):
    """
    Calculates the specificity scores based on the true and predicted labels.

    :param y_test: true labels
    :param y_pred: predicted labels
    :return: specificity scores
    """

    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0]
    # tp = mcm[:, 1, 1]
    # fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    sp = tn / (tn + fp)
    return sum(sp) / 4



