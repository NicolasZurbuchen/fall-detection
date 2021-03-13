import os.path as path


def validates_main_experiment_arguments(args):
    """
    Validates the main_experiment script arguments

    :param args: list of arguments
    :return: list of errors
    """

    errors = []

    validates_dataset_folder(errors, args.dataset_folder)
    validates_output_folder(errors, args.output_folder)
    validates_sensors(errors, args.sensors)
    validates_ignored_subjects(errors, args.ignored_subjects)
    validates_duration(errors, args.duration)
    validates_frequencies(errors, args.frequencies)
    validates_pre_time(errors, args.pre_time, args.duration)
    validates_post_time(errors, args.post_time, args.duration)
    validates_classification(errors, args.classification)
    validates_models(errors, args.models)
    validates_k_fold(errors, args.k_fold)

    return errors


def validates_main_plot_sample_arguments(args):
    """
    Validates the main_plot_sample script arguments

    :param args: list of arguments
    :return: list of errors
    """

    errors = []

    validates_data_file(errors, args.data_file)
    validates_output_folder(errors, args.output_folder)
    validates_sensors(errors, args.sensors)
    validates_duration(errors, args.duration)
    validates_pre_time(errors, args.pre_time, args.duration)
    validates_post_time(errors, args.post_time, args.duration)

    return errors


def validates_dataset_folder(errors, dataset_folder):
    """
    Validates the dataset location. Performs the following checks:
        - is valid path
        - is folder

    :param errors:
    :param dataset_folder:
    :return:
    """

    if not path.exists(dataset_folder) or not path.isdir(dataset_folder):
        errors.append("Invalid dataset folder argument.")


def validates_data_file(errors, data_file):
    """
    Validates the data file. Performs the following checks:
        - is valid path
        - is file

    :param errors:
    :param data_file:
    :return:
    """

    if not path.exists(data_file) or not path.isfile(data_file):
        errors.append("Invalid data file argument.")


def validates_output_folder(errors, output_folder):
    """
    Validates the output folder location. Performs the following checks:
        - is valid path
        - is folder

    :param errors:
    :param output_folder:
    :return:
    """

    if not path.exists(output_folder) or not path.isdir(output_folder):
        errors.append("Invalid output folder argument.")


def validates_sensors(errors, sensors):
    """
    Validates the sensors' axes. Performs the following checks:
        - is not empty
        - has no duplicates
        - is valid axe

    :param errors:
    :param sensors:
    :return:
    """

    error = False
    valid_sensors = list(range(0, 9))

    if len(sensors) == 0:
        error = True

    unique = set(sensors)
    if len(unique) != len(sensors):
        error = True

    for s in sensors:
        if s not in valid_sensors:
            error = True
            break

    if error:
        errors.append("Invalid sensors argument.")


def validates_ignored_subjects(errors, ignored_subjects):
    """
    Validates the list of ignored subjects. Performs the following checks:
        - is valid subject

    :param errors:
    :param ignored_subjects:
    :return:
    """

    valid_subjects = ['SA01', 'SA02', 'SA03', 'SA04', 'SA05', 'SA06', 'SA07', 'SA08', 'SA09', 'SA10',
                      'SA11', 'SA12', 'SA13', 'SA14', 'SA15', 'SA16', 'SA17', 'SA18', 'SA19', 'SA20',
                      'SA21', 'SA22', 'SA23', 'SE01', 'SE02', 'SE03', 'SE04', 'SE05', 'SE06', 'SE07',
                      'SE08', 'SE09', 'SE10', 'SE11', 'SE12', 'SE13', 'SE14', 'SE15']

    for s in ignored_subjects:
        if s not in valid_subjects:
            errors.append("Invalid ignored_subjects argument.")
            break


def validates_duration(errors, duration):
    """
    Validates the list of frequencies. Performs the following checks:
        - is within valid range

    :param errors:
    :param duration:
    :return:
    """

    if duration < 1000 or duration > 12000:
        errors.append("Invalid duration argument.")


def validates_frequencies(errors, frequencies):
    """
    Validates the list of frequencies. Performs the following checks:
        - is not empty
        - has no duplicates
        - is valid frequency

    :param errors:
    :param frequencies:
    :return:
    """

    error = False
    valid_frequencies = []
    for f in range(1, 201):
        if 200 % f == 0:
            valid_frequencies.append(f)

    if len(frequencies) == 0:
        error = True

    unique = set(frequencies)
    if len(unique) != len(frequencies):
        error = True

    for f in frequencies:
        if f not in valid_frequencies:
            error = True

    if error:
        errors.append("Invalid frequencies argument.")


def validates_pre_time(errors, pre_time, duration):
    """
    Validates the time before the impact point. Performs the following checks:
        - is within valid range

    :param errors:
    :param pre_time:
    :param duration:
    :return:
    """

    if pre_time < 1 or pre_time > duration / 3:
        errors.append("Invalid pre_time argument.")


def validates_post_time(errors, post_time, duration):
    """
    Validates the time after the impact point. Performs the following checks:
        - is within valid range

    :param errors:
    :param post_time:
    :param duration:
    :return:
    """

    if post_time < 1 or post_time > duration / 3:
        errors.append("Invalid post_time argument.")


def validates_classification(errors, classification):
    """
    Validates the classification method. Performs the following checks:
        - is valid classification

    :param errors:
    :param classification:
    :return:
    """

    valid_classifications = ['binary', 'multi-class']

    if classification not in valid_classifications:
        errors.append("Invalid classification argument.")


def validates_models(errors, models):
    """
    Validates the list of models. Performs the following checks:
        - is not empty
        - has no duplicates
        - has valid model names

    :param errors:
    :param models:
    :return:
    """

    error = False
    valid_models = ['knn', 'svm', 'dt', 'rf', 'gb']

    if len(models) == 0:
        error = True

    unique = set(models)
    if len(unique) != len(models):
        error = True

    for m in models:
        if m not in valid_models:
            error = True

    if error:
        errors.append("Invalid models argument.")


def validates_k_fold(errors, k_fold):
    """
    Validates the k_fold value. Performs the following checks:
        - is within valid range

    :param errors:
    :param k_fold:
    :return:
    """

    if k_fold < 2 or k_fold > 20:
        errors.append("Invalid k_fold argument.")
