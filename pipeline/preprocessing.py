import numpy as np


def change_activity_duration(data, duration):
    """
    Cuts the sample of an activity to match the wanted duration. The same duration at the start and end of the sample
    will be cut. This allows to remove outliers from the activity or test another solution.

    :param data: data from the activity in format of a DataFrame
    :param duration: the duration of the activity
    :return: the data activity shortened to match to given duration
    """

    # Calculates the number of samples to remove at the start and end
    total_samples = int(duration * 200 / 1000)
    no_samples_to_remove = len(data) - total_samples
    no_samples_to_remove_by_side = int(no_samples_to_remove / 2)

    # Removes the samples (by handling the specific cases)
    if (total_samples + len(data)) % 2 == 0:
        data = data.drop(data.tail(no_samples_to_remove_by_side).index)
    else:
        data = data.drop(data.tail(int(no_samples_to_remove / 2 + 0.5)).index)
    data = data.drop(data.head(no_samples_to_remove_by_side).index)

    return data


def change_activity_sampling(data, frequency):
    """
    Changes the frequency of the activity sample which allows to simulate sensor with a lower or higher sampling
    measure. Uses linear interpolation when upsampling. Applies FFT if wanted to resample the activity.

    :param data: data from the activity in format of a DataFrame
    :param frequency: the frequency in which to change the sampling
    :return: the data activity resampled to match the given parameters
    """

    sampling = 1000 / frequency
    data = data.asfreq(str(sampling) + 'ms')
    data = data.interpolate()

    return data


def divide_fall(data, is_fall, pre_time, post_time):
    """
    Divides falls into its three defined phases which are pre-fall, fall and post-fall. The ADL samples are
    not divided.

    :param data: the data from the activity whose features must be extracted
    :param is_fall: is the current data a fall or an ADL
    :param pre_time: time before the impact point
    :param post_time: time after the impact point
    :return:
    """

    # Determines the peak value of the magnitude
    d = np.square(data)
    d = np.sqrt(np.sum(d, axis=1))
    i = np.argmax(d, axis=0)

    if is_fall:

        # Determines where to split the fall sample
        size_l = int(len(data) * (pre_time / 1000))
        size_h = int(len(data) * (post_time / 1000))
        low = i - size_l if i - size_l >= 0 else 0
        high = i + size_h if i + size_h < len(data) else len(data)

        # Ensures to have at least one sample per phase
        if low == 0:
            low += 1
        if high == len(data):
            high -= 1

        # Extracts fall, pre-fall and post-fall phases
        fall = data.iloc[low:high, :]
        pre_fall = data.iloc[0:low, :]
        post_fall = data.iloc[high:, :]
        return fall, pre_fall, post_fall

    else:
        adl = data
        return adl, None, None
