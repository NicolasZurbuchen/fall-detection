# Fall Detection for the Elderly

This is a cleaned version of the experiment I developped during my Master's Thesis to detect falls based on wereable sensors. It resulted in two publications:

* Zurbuchen, N., Wilde, A., & Bruegger, P. (2021). [“A Machine Learning Multi-Class Approach for Fall Detection Systems Based on Wearable Sensors with a Study on Sampling Rates Selection”](https://www.mdpi.com/1424-8220/21/3/938). In: Sensors 21.3. ISSN: 1424-8220. DOI: 10.3390/s21030938.
* Zurbuchen, N., Bruegger, P., & Wilde, A. (2020). [“A Comparison of Machine Learning Algorithms for Fall Detection using Wearable Sensors”](https://ieeexplore.ieee.org/document/9065205). In: 2020 International Conference on Artificial Intelligence in Information and Communication (ICAIIC), pp. 427–431. DOI: 10.1109/ICAIIC48513.2020.9065205.


## Dataset

This experiment only works with the publicly available [SisFall](https://www.mdpi.com/1424-8220/17/1/198) dataset which is a a fall and movement dataset based on wearable sensors.


## Usage

The experiment is in the file `main_experiment.py` which requires the following input parameters:

* `dataset_folder` : The path of the folder containing the SisFall data set.
* `output_folder` : The path of the folder where all the results will be saved.

The following list defines the optional parameters which all have default values:

* `-se`, `--sensors` : The list of sensors axes as numbers from 0 to 8 included.
* `-is`, `--ignored_subjects` : The list of ignored subjects as subjects names from SA01 to SA23 and SE01 to SE15.
* `-du`, `--duration` : The duration of the sample in \[ms\] as a number between 1000 and 12000 included.
* `-fr`, `--frequencies` : The list of frequencies of the sampling \[Hz\] as numbers from 1 to 200 included and divisor of 200.
* `-pr`, `--pre_time` : The duration after the impact in \[ms\] (must be between 100 and 5000, only available with multi-class).
* `-po`, `--post_time` : The duration before the impact in [ms] (must be between 100 and 5000, only available with multi-class).
* `-cl`, `--classification` : The classification type (either binary or multi-class).
* `-mo`, `--models` : The list of machine learning algorithms to use (either knn, svm, dt, rg or gb).
* `-kf`, `--k_fold` : The number of folds to use (must be between 2 and 10).
