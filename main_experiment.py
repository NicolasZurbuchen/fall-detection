#!/usr/bin/env python3

import sys
import argparse
import pandas as pd

from pipeline.acquisition import load_sisfall_data
from pipeline.preprocessing import change_activity_duration
from pipeline.preprocessing import change_activity_sampling
from pipeline.preprocessing import divide_fall
from pipeline.feature_extraction import extract_features
from pipeline.processing import fit_and_test_classifiers
from pipeline.evaluation import evaluate_classifiers

from utils.validation import validates_main_experiment_arguments


# Default values
SENSORS_AXES = [0, 1, 2, 3, 4, 5]
IGNORED_SUBJECTS = ['SA17', 'SA20', 'SA23', 'SE01', 'SE02', 'SE03', 'SE04', 'SE05', 'SE06', 'SE07', 'SE08', 'SE09', 'SE10', 'SE11', 'SE12', 'SE13', 'SE14', 'SE15']
DURATION = 10000
FREQUENCIES = [1, 2, 5, 10, 20, 50, 100, 200]
PRE_TIME = 1500
POST_TIME = 500
CLASSIFICATION = 'binary'
MODELS = ['knn', 'svm', 'dt', 'rf', 'gb']
K_FOLD = 5


parser = argparse.ArgumentParser(description="This script fits and tests various machine learning algorithms to differenciate between falls and activities of daily living and then output various results.")
parser.add_argument('dataset_folder', type=str, help="The path of the folder containing the SisFall data set.")
parser.add_argument('output_folder', type=str, help="The path of the folder where all the results will be saved.")
parser.add_argument('-se', '--sensors', type=int, default=SENSORS_AXES, nargs='+', help="The list of sensors axes as numbers from 0 to 8 included.")
parser.add_argument('-is', '--ignored_subjects', type=str, default=IGNORED_SUBJECTS, nargs='+', help="The list of ignored subjects as subjects names from SA01 to SA23 and SE01 to SE15.")
parser.add_argument('-du', '--duration', type=int, default=DURATION, help="The duration of the sample in [ms] as a number between 1000 and 12000 included.")
parser.add_argument('-fr', '--frequencies', type=int, default=FREQUENCIES, nargs='+', help="The list of frequencies of the sampling [Hz] as numbers from 1 to 200 included and divisor of 200.")
parser.add_argument('-pr', '--pre_time', type=int, default=PRE_TIME, help="The duration after the impact in [ms] (must be between 100 and 5000, only available with multi-class).")
parser.add_argument('-po', '--post_time', type=int, default=POST_TIME, help="The duration before the impact in [ms] (must be between 100 and 5000, only available with multi-class).")
parser.add_argument('-cl', '--classification', type=str, default=CLASSIFICATION, help="The classification type (either binary or multi-class).")
parser.add_argument('-mo', '--models', type=str, default=MODELS, nargs='+', help="The list of machine learning algorithms to use (either knn, svm, dt, rg or gb).")
parser.add_argument('-kf', '--k_fold', type=int, default=K_FOLD, help="The number of folds to use (must be between 2 and 10).")
args = parser.parse_args()


if __name__ == '__main__':

    # Gets script parameters
    dataset_folder = args.dataset_folder
    output_folder = args.output_folder
    sensors = args.sensors
    ignored_subjects = args.ignored_subjects
    duration = args.duration
    frequencies = args.frequencies
    pre_time = args.pre_time
    post_time = args.post_time
    classification = args.classification
    models = args.models
    k_fold = args.k_fold

    # Validates arguments
    errors = validates_main_experiment_arguments(args)
    if len(errors) != 0:
        print("Problems with script arguments. Please check the following arguments:")
        [print(e) for e in errors]
        sys.exit("Invalid arguments. Aborted.")

    # Loads SisFall dataset
    raw_dataset = load_sisfall_data(dataset_folder, ignored_subjects, sensors)
    all_results = []

    # Preprocesses the dataset for each frequency
    for frequency in frequencies:
        dataset = pd.DataFrame()
        labels = []

        for i in raw_dataset.index:
            d = raw_dataset['data'][i]
            d = change_activity_duration(d, duration)
            d = change_activity_sampling(d, frequency)

            is_fall = raw_dataset['activity'][i].startswith('F')
            if classification == 'binary':
                dataset = dataset.append(extract_features(d, True))
                labels.append(1 if is_fall else 0)
            else:
                activity, pre_fall, post_fall = divide_fall(d, is_fall, pre_time, post_time)
                dataset = dataset.append(extract_features(activity, True))
                labels.append(1 if is_fall else 0)
                if is_fall:
                    dataset = dataset.append(extract_features(pre_fall, True))
                    labels.append(2)
                    dataset = dataset.append(extract_features(post_fall, True))
                    labels.append(3)

        # Fits and tests models
        results = fit_and_test_classifiers(dataset, labels, models, k_fold)
        results.insert(0, 'frequency', [frequency] * len(models) * k_fold)
        all_results.append(results)

    all_results = pd.concat(all_results, sort=False)
    all_results.index = list(range(0, all_results.shape[0]))

    class_names = ['ADL', 'Fall'] if classification == 'binary' else ['ADL', 'Fall', 'Pre-fall', 'Post-fall']
    evaluate_classifiers(all_results, output_folder, class_names, frequencies, models, k_fold)

    print()
