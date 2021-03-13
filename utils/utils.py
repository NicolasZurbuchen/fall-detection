import os
import openpyxl
import pandas as pd

from datetime import datetime


def create_output_hierarchy(output_folder, frequencies, models):
    """
    Creates the hierarchy of folders for saving the plots and scores.

    :param output_folder: root output directory
    :param frequencies: list of frequencies
    :param models: list of models
    :return: path to output directory
    """

    # Retrieves the data to ensure unique output folder
    now = datetime.now()
    dt_string = now.strftime('%Y%m%d_%H%M%S')

    # Creates root output directory
    results_directory = output_folder + '/results_' + dt_string
    os.mkdir(results_directory)

    # Creates directory for plots
    plot_directory = results_directory + '/plots'
    os.mkdir(plot_directory)

    # Creates sub directories for each plot type
    for frequency in frequencies:
        frequency_directory = plot_directory + '/' + str(frequency) + 'Hz'
        os.mkdir(frequency_directory)

        # Creates box and whisker directory
        baw_directory = frequency_directory + '/baw'
        os.mkdir(baw_directory)

        # Creates confusion matrix directory
        cnf_directory = frequency_directory + '/cnf'
        os.mkdir(cnf_directory)

        # Creates directory for each model (only for cnf)
        for model in models:
            model_directory = cnf_directory + '/' + model
            os.mkdir(model_directory)

    return results_directory


def save_to_file(output_folder, results):
    """
    Saves the calculated scores to an excel file for data persistence

    :param output_folder: output directory
    :param results: dataframe of scores
    """

    # Creates and open excel file
    file_location = output_folder + '/results.xlsx'
    writer = pd.ExcelWriter(file_location, engine='openpyxl')
    if os.path.isfile(output_folder):
        book = openpyxl.load_workbook(output_folder)
        writer.book = book

    # Writes the results to the file
    results.drop(['classifier', 'x_test', 'y_test', 'y_pred'], axis=1).to_excel(writer, index=False)

    # Saves and closes the file
    writer.save()
    writer.book.close()


def with_magnitude(sensors):
    """
    Verifies if the magnitude axe can be used for extracting features.

    :param sensors: sensors' axes
    :return: True if magnitude can be used
    """

    sensors.sort()

    for i in range(0, 9, 3):
        if not (sensors.contains(i) == sensors.contains(i+1) == sensors.contains(i+2)):
            return False
    return True



