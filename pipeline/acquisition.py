import os
import pandas as pd


# Theses activities have only one trial and therefore need a different pre-processing
SPECIAL_ACTIVITIES = ['D01', 'D02', 'D03', 'D04']
SPECIAL_ACTIVITIES_STARTS = [1000, 5000, 9000, 13000, 17000]


def load_sisfall_data(folder_path, ignored_subjects, sensors_axes):
    """
    Load the data contained in the SisFall dataset into a DataFrame.

    :param folder_path: path to the sisfall dataset
    :param ignored_subjects: list of subjects to ignore
    :param sensors_axes: list of sensors' axes to use
    :return: DataFrame containing all data
    """

    # DataFrame containing the whole dataset
    dataset = []

    # Lists all subjects
    subjects = os.listdir(folder_path)
    subjects.sort()

    # Reads data from all subjects
    for subject in subjects:
        subject_path = folder_path + '/' + subject

        # Ensures only wanted subjects are used
        if subject in ignored_subjects or not os.path.isdir(subject_path):
            continue

        # Lists all subject's samples
        activities = os.listdir(subject_path)
        activities.sort()

        # Reads data for all activities
        for activity in activities:
            if activity.endswith('.txt'):
                if activity[0:3] in SPECIAL_ACTIVITIES:
                    for i in SPECIAL_ACTIVITIES_STARTS:
                        data = read_file(subject_path + '/' + activity, sensors_axes)
                        data = data.iloc[i:i + 2000, :]
                        data = {'subject': subject, 'activity': activity[0:3], 'trial': activity[9:12], 'data': data}
                        dataset.append(data)
                else:
                    data = read_file(subject_path + '/' + activity, sensors_axes)
                    data = {'subject': subject, 'activity': activity[0:3], 'trial': activity[9:12], 'data': data}
                    dataset.append(data)

        # Used for test purposes
        """if subject.startswith('SA01'):
            break"""

    return pd.DataFrame(dataset)


def read_file(file_path, sensors_axes):
    """
    Reads the data from an activity and convert them into a DataFrame with a corresponding time series to the frequency
    of the sensor.

    :param file_path: the path of the file containing the data of the activity
    :param sensors_axes: the data from which sensors' axes is wanted
    :return: a DataFrame containing the data for one activity
    """

    # Reads the file and add the defined columns names
    names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_2_x', 'acc_2_y', 'acc_2_z']
    data = pd.read_csv(file_path, header=None, names=names, usecols=names, sep=',|;', engine='python')

    # Converts the analog data in gravity
    d1 = data.iloc[:, 0:3] * ((2 * 16) / (2 ** 13))
    d2 = data.iloc[:, 3:6] * ((2 * 2000) / (2 ** 16)) * (3.14159 / 180)
    d3 = data.iloc[:, 6:9] * ((2 * 8) / (2 ** 14))
    data = pd.concat([d1, d2, d3], axis=1)

    # Selects sensors
    data = data.iloc[:, sensors_axes]
    index = pd.date_range('1/1/2000', periods=len(data), freq='5ms')
    data.set_index(index, inplace=True)

    return data
