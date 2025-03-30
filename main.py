# coding: utf-8

from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_and_prepare_data
from anomaly_generator import generate_anomaly_data
from one_class_svm import train_one_class_svm, evaluate_one_class_svm, visualize_results


if __name__ == "__main__":
    # Check time
    start_time = time()

    # Define columns
    cols = ['unit', 'cycles', 'op_setting1', 'op_setting2', 'op_setting3',
            's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
            's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    # Choose between training and predict
    useTrainforSVM = True

    # Load and prepare data
    trainCopy, testCopy, combineRowData = load_and_prepare_data(cols)
    print(f"Data loading done in {time() - start_time:.3f}s")

    # Generate anomaly data
    anomalyList = generate_anomaly_data(trainCopy)

    # Combine normal data with anomaly data
    combineRowData = np.concatenate((trainCopy, anomalyList), axis=0)
    combineRowData = pd.DataFrame(combineRowData)
    combineRowData.columns = cols
    print(combineRowData.head())

    # Split data for training/testing
    features = ['s11', 's20']
    labels = ['unit']
    X = combineRowData[features]
    y = combineRowData[labels]

    # Train/evaluate One-Class SVM
    clf, X_train, X_test, y_train, y_test, resultLier = train_one_class_svm(X, y, useTrainforSVM)
    print(f"SVM training done in {time() - start_time:.3f}s")

    # Evaluate and visualize results
    evaluate_one_class_svm(X_train.values, resultLier)
    visualize_results(clf, X_train.values, resultLier, X_test, y_test)
    print(f"Total execution time: {time() - start_time:.3f}s")