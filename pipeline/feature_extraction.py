import itertools
import numpy as np
import pandas as pd


def extract_features(data, with_magnitude):
    """
    Extracts various features from the time and frequency domains from a given sample of activity. Also constructs
    features by combining the raw data.

    :param data: the data from the activity
    :param with_magnitude: calculate the magnitude of the sensors
    :return: list with all the features extracted from the activity
    """

    # Calculates the acceleration and rotation magnitudes
    if with_magnitude:
        for i in range(0, data.shape[1], 3):
            magnitude = np.linalg.norm(data.iloc[:, i:i+3], axis=1)
            name = 'mag_' + data.columns[i][0:len(data.columns[i])-2]
            data[name] = magnitude

    # Creates features vector name
    names = ['mean', 'var', 'std', 'median', 'max', 'min', 'ptp', 'centile25', 'centile75', 'psd', 'pse']
    columns = list('_'.join(n) for n in itertools.product(names, data.columns.tolist()))

    # Time domain features
    features = np.mean(data, axis=0)
    features = np.hstack((features, np.var(data, axis=0)))
    features = np.hstack((features, np.std(data, axis=0)))
    features = np.hstack((features, np.median(data, axis=0)))
    features = np.hstack((features, np.max(data, axis=0)))
    features = np.hstack((features, np.min(data, axis=0)))
    features = np.hstack((features, np.ptp(np.asarray(data), axis=0)))
    features = np.hstack((features, np.percentile(data, 25, axis=0)))
    features = np.hstack((features, np.percentile(data, 75, axis=0)))

    # Frequency domain features
    psd = np.abs(np.fft.fft(data)) ** 2
    psd = psd / data.shape[0]
    pse = psd * np.log(psd)
    features = np.hstack((features, np.sum(psd, axis=0)))
    features = np.hstack((features, -np.sum(pse, axis=0)))

    # Creates a DataFrame
    features = pd.DataFrame([features], columns=columns)
    return features
