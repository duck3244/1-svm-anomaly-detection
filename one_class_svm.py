# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def train_one_class_svm(X, y, useTrainforSVM=True):
    """
    Train or load One-Class SVM model
    
    Args:
        X: Feature dataframe
        y: Label dataframe
        useTrainforSVM: Whether to train a new model or load existing
        
    Returns:
        clf: Trained One-Class SVM model
        X_train, X_test, y_train, y_test: Split data
        resultLier: Boolean array of inliers/outliers
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # Convert to numpy array
    X_data = X_train.values
    
    # Visualize distribution
    plt.figure(figsize=(10, 6))
    plt.plot(X_data[:,0], linestyle="", marker="o")
    plt.plot(X_data[:,1], linestyle="", marker="x")
    plt.savefig('Distribution.png')
    plt.show()
    
    # Define outlier fraction
    OUTLIER_FRACTION = 0.01
    
    if useTrainforSVM:
        # Create One-Class SVM
        clf = svm.OneClassSVM(kernel="rbf")
        
        # Train the model
        clf.fit(X_data)
        
        # Save model
        with open('OneClassSVMModel.pickle', 'wb') as saveSVM:
            pickle.dump(clf, saveSVM)
    else:
        # Load model
        with open('OneClassSVMModel.pickle', 'rb') as loadModel:
            clf = pickle.load(loadModel)
    
    # Define threshold
    dist_to_border = clf.decision_function(X_data).ravel()
    threshold = stats.scoreatpercentile(dist_to_border, 100 * OUTLIER_FRACTION)
    
    # Get boolean values for inliers/outliers
    resultLier = dist_to_border > threshold
    
    print("One-Class SVM Process Complete")
    
    return clf, X_train, X_test, y_train, y_test, resultLier

def evaluate_one_class_svm(X, resultLier):
    """
    Evaluate One-Class SVM results
    
    Args:
        X: Feature data
        resultLier: Boolean array of inliers/outliers
    """
    # Convert boolean array to list
    booleanList = []
    for booleanValue in resultLier:
        booleanList.append(booleanValue)
    
    # Process and count outliers
    valueList = []
    csvCount = 0
    saveCount = 0
    
    for split in X:
        if booleanList[csvCount]:
            valueList.append([float(split[0]), float(split[1])])
            saveCount += 1
        else:
            valueList.append([float(split[0]), float(split[1])])
            print(f"anomaly line : {csvCount + 1}")
        
        csvCount += 1
    
    print(f"Remove Outlier : {len(X) - saveCount}")

def visualize_results(clf, X, resultLier, X_test, y_test):
    """
    Visualize One-Class SVM results and performance
    
    Args:
        clf: Trained One-Class SVM model
        X: Feature data
        resultLier: Boolean array of inliers/outliers
        X_test: Test feature data
        y_test: Test labels
    """
    # Outlier fraction
    OUTLIER_FRACTION = 0.01
    OCSVM = len(X)
    
    # Create mesh grid for contour plot
    xx, yy = np.meshgrid(np.linspace(40, 60, 1500), np.linspace(35, 50, 1500))
    n_inliers = int((1. - OUTLIER_FRACTION) * OCSVM)
    n_outliers = int(OUTLIER_FRACTION * OCSVM)
    
    # Decision function
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Calculate threshold
    dist_to_border = clf.decision_function(X).ravel()
    threshold = stats.scoreatpercentile(dist_to_border, 100 * OUTLIER_FRACTION)
    
    # Plot decision boundary
    plt.figure(figsize=(12, 8))
    plt.title("Outlier detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
    b = plt.scatter(X[resultLier == 0, 0], X[resultLier == 0, 1], c='white')
    c = plt.scatter(X[resultLier == 1, 0], X[resultLier == 1, 1], c='black')
    plt.axis('tight')
    plt.legend([a.collections[0], b, c],
               ['learned decision function', 'outliers', 'inliers'],
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlim((40, 60))
    plt.ylim((35, 50))
    plt.savefig('One-Class SVM.png')
    plt.show()
    
    print(f"Outlier value : {n_outliers + 1}")
    
    # Precision-Recall curve
    clf.fit(X_test)
    
    # The lower, the more normal
    scoring = -clf.decision_function(X_test)
    thresholds = roc_curve(y_test, scoring)
    
    precision, recall = precision_recall_curve(y_test, scoring)[:2]
    AUPR = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=1, label=f"AUPR: {AUPR:.3f}")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.title('Precision-Recall curve', fontsize=25)
    plt.legend(loc="best", prop={'size': 12})
    plt.savefig('Precision-Recall_Curve.png')
    plt.show()
