# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def load_and_prepare_data(cols):
    """
    Load data from files and prepare for processing
    
    Args:
        cols: Column names for the dataframe
        
    Returns:
        trainCopy: Subset of training data
        testCopy: Subset of test data
        combineRowData: Combined dataframe
    """
    # Load data
    trainRowData = np.loadtxt("train_FD001.txt")
    testRowData = np.loadtxt("test_FD001.txt")
    
    # Combine data
    combineRowData = np.concatenate((trainRowData, testRowData), axis=0)
    combineRowData = pd.DataFrame(combineRowData)
    combineRowData.columns = cols
    
    # Print first few rows
    print(combineRowData.head())
    
    # Copy subset of data for anomaly generation
    trainCopy = trainRowData[:191]
    testCopy = testRowData[:191]
    
    return trainCopy, testCopy, combineRowData

def visualize_data_distribution(X):
    """
    Visualize the distribution of features
    
    Args:
        X: Feature data
    """
    plt.figure(figsize=(10, 6))
    plt.plot(X[:,0], linestyle="", marker="o", label="Feature 1")
    plt.plot(X[:,1], linestyle="", marker="x", label="Feature 2")
    plt.legend()
    plt.title("Data Distribution")
    plt.savefig('Distribution.png')
    plt.show()
