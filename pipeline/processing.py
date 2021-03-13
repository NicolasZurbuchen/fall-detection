import pandas as pd
import numpy as np

from datetime import datetime

from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import preprocessing
from sklearn import clone

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold


def fit_and_test_classifiers(x, y, classifiers_names, k_fold):
    """"
    Fits and tests the wanted classifiers with the previously preprocessed data.

    :param x: preprocessed data
    :param y: corresponding labels
    :param classifiers_names: wanted classifiers
    :param k_fold: number of folds in the k-fold cross-validation
    :return: results of each split
    """

    # Normalizes data
    x = preprocessing.MinMaxScaler().fit_transform(x)
    y = np.array(y)

    # Creates classifier and k-fold
    classifiers, full_names = create_classifiers(classifiers_names)
    kf = StratifiedKFold(n_splits=k_fold, random_state=None, shuffle=False)
    results = []

    # Fits and tests each classifier
    for i, classifier in enumerate(classifiers):
        for k, (train_index, test_index) in enumerate(kf.split(x, y)):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fits and times the fitting process
            clf = clone(classifier)
            start_fit = datetime.now()
            clf.fit(x_train, y_train)
            stop_fit = datetime.now()
            fit_time = stop_fit.timestamp() - start_fit.timestamp()

            # Tests and times the testing process
            start_test = datetime.now()
            y_pred = clf.predict_proba(x_test)
            stop_test = datetime.now()
            test_time = (stop_test.timestamp() - start_test.timestamp())

            # Merges results
            result = {'ksplit': k + 1, 'name': full_names[i], 'abbreviation': classifiers_names[i], 'classifier': clf, 'x_test': x_test, 'y_test': y_test, 'y_pred': y_pred, 'fit_time': fit_time, 'test_time': test_time}
            results.append(result)

    return pd.DataFrame(results)


def create_classifiers(classifiers_names):
    """
    Instantiates the classifiers and set their full names.

    :param classifiers_names: list of wanted classifiers
    :return: classifiers and their full names
    """

    classifiers = []
    full_names = []

    if 'knn' in classifiers_names:
        classifiers.append(neighbors.KNeighborsClassifier())
        full_names.append('k-Nearest Neighbour')
    if 'svm' in classifiers_names:
        classifiers.append(svm.SVC(probability=True))
        full_names.append('Support Vector Machines')
    if 'dt' in classifiers_names:
        classifiers.append(tree.DecisionTreeClassifier())
        full_names.append('Decision Tree')
    if 'rf' in classifiers_names:
        classifiers.append(RandomForestClassifier())
        full_names.append('Random Forest')
    if 'gb' in classifiers_names:
        classifiers.append(GradientBoostingClassifier())
        full_names.append('Gradient Boosting')

    return classifiers, full_names
