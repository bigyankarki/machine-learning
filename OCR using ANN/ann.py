# This file is no longer used in main program. You can safely ignore this file.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_mldata
from recognize import resize_image
from sklearn.model_selection import cross_val_predict


def run_ann():
    print("Training ANN")

    # import dataset.
    mnist = fetch_mldata('MNIST original')
    x, y = mnist['data'], mnist['target']
    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    # shuffle the training set to ensure consistent cross-validation folds.
    shuffle_index = np.random.permutation(60000)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

    # import MLPclassifier for scikit learn and train the model.
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10,), random_state=42)
    clf.fit(x_train, y_train)

    print("Training Completed.")
    # prediction
    print("Predicting the digit.")
    predicted_digit = clf.predict([resize_image()])

    # let's evaluate MLPClassifier using cross-val_score using k-fold cross-validation
    print("Finding Confidence.")
    conf = np.average(cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy"))

    print("Looking at the cocnfusion Matrix.")


    print("Returning prediction and confidence.")
    return [predicted_digit, conf]


if __name__ == "__main__":
    run_ann()