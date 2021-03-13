#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from utils.validation import validates_main_plot_sample_arguments
from pipeline.acquisition import read_file
from pipeline.preprocessing import change_activity_duration


# Default values
SENSORS_AXES = [0, 1, 2]
DURATION = 10000
PRE_TIME = 1500
POST_TIME = 500


parser = argparse.ArgumentParser(description="This script fits and tests various machine learning algorithms to differenciate between falls and activities of daily living and then output various results.")
parser.add_argument('data_file', type=str, help="The path of the file containing the sample to plot.")
parser.add_argument('output_folder', type=str, help="The path of the folder where all the results will be saved.")
parser.add_argument('-se', '--sensors', type=int, default=SENSORS_AXES, nargs='+', help="The list of sensors axes as numbers from 0 to 8 included.")
parser.add_argument('-du', '--duration', type=int, default=DURATION, help="The duration of the sample in [ms] as a number between 1000 and 12000 included.")
parser.add_argument('-pr', '--pre_time', type=int, default=PRE_TIME, help="The duration after the impact in [ms] (must be between 100 and 5000, only available with multi-class).")
parser.add_argument('-po', '--post_time', type=int, default=POST_TIME, help="The duration before the impact in [ms] (must be between 100 and 5000, only available with multi-class).")
args = parser.parse_args()


if __name__ == '__main__':

    # Gets script parameters
    data_file = args.data_file
    output_folder = args.output_folder
    sensors = args.sensors
    duration = args.duration
    pre_time = args.pre_time
    post_time = args.post_time

    # Validates arguments
    errors = validates_main_plot_sample_arguments(args)
    if len(errors) != 0:
        print("Problems with script arguments. Please check the following arguments:")
        [print(e) for e in errors]
        sys.exit("Invalid arguments. Aborted.")

    # Retrieves the data to ensure unique output folder
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")

    # Reads and preprocesses data
    data = read_file(data_file, sensors)
    data = change_activity_duration(data, duration)

    # Finds the peak magnitude
    d = np.square(data)
    d = np.sqrt(np.sum(d, axis=1))
    i = np.argmax(d, axis=0)

    # Determines where to split the fall sample
    size_l = int(len(data) * (pre_time / 10000))
    size_h = int(len(data) * (post_time / 10000))
    low = i - size_l if i - size_l >= 0 else 0
    high = i + size_h if i + size_h < len(data) else len(data)

    # Specifies markers
    markers = ['o', ',', 'd', 's', 'v', 'P', 'X', 'H', '<']
    markers_on = np.linspace(0, len(data) - 1, 6).astype(int)
    markers_on_zoom = np.linspace(0, len(data) - 1, 12).astype(int)

    # Without zoom
    plt.figure()
    data.plot(kind='line', linestyle='-', style=markers[:len(sensors)], markevery=markers_on)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [g]')
    plt.legend()
    plt.grid(axis='y')
    plt.axvline(data.index[low], color='grey', linewidth=2, linestyle='--')
    plt.axvline(data.index[high], color='grey', linewidth=2, linestyle='--')
    plt.savefig(output_folder + '/' + dt_string + '_' + data_file[-16:-4] + '.png')

    # With zoom
    plt.figure()
    data.plot(kind='line', linestyle='-', style=markers[:len(sensors)], markevery=markers_on_zoom)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [g]')
    plt.legend()
    plt.grid(axis='y')
    shift_low = int(low * 0.05) if low * 0.05 >= 0 else 0
    shift_high = int(high * 0.05) if high * 0.05 <= len(data) else 0
    plt.axvline(data.index[low], color='grey', linewidth=2, linestyle='--')
    plt.axvline(data.index[high], color='grey', linewidth=2, linestyle='--')
    plt.axvline(data.index[i], color='grey', linewidth=2, linestyle='--')
    plt.axes().set(xlim=(data.index[low - shift_low], data.index[high + shift_high]))
    plt.savefig(output_folder + '/' + dt_string + '_' + data_file[-16:-4] + '_zoom.png')

    plt.show()
