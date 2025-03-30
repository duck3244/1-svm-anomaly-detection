# coding: utf-8

import random
import numpy as np


def find_min_max_values(trainCopy):
    """
    Find the min and max values for each feature in the training data
    
    Args:
        trainCopy: Training data
        
    Returns:
        maxList: List of maximum values
        minList: List of minimum values
    """
    maxList = []
    minList = []
    
    # Start from index 2 (skipping header)
    i = 2
    
    for _ in range(24):  # 26 - 2
        if i == 26:
            break
        tempMax = max(trainCopy[:,i])
        tempMin = min(trainCopy[:,i])
        maxList.append(tempMax)
        minList.append(tempMin)
        i += 1
        
    return maxList, minList

def generate_anomaly_data(trainCopy):
    """
    Generate anomaly data based on max and min values
    
    Args:
        trainCopy: Training data
        
    Returns:
        anomalyList: List of generated anomaly data
    """
    # Find max and min values
    maxList, minList = find_min_max_values(trainCopy)
    
    # Generate anomaly data
    anomalyList = []
    
    # Generate max anomalies
    anomalyMaxLabel = 1
    limit = 2
    i = 1
    
    for _ in range(200):
        if i == limit:
            break
            
        anomalyData = []
        anomalyData.append(anomalyMaxLabel)  # 0
        anomalyData.append(191 + i)  # 1
        
        # Add anomaly values for each feature
        anomalyData.append(random.uniform(maxList[0] + 3, maxList[0] + 5))  # 2
        anomalyData.append(random.uniform(maxList[1] + 1, maxList[1] + 2))  # 3
        anomalyData.append(maxList[2])  # 4
        anomalyData.append(maxList[3])  # 5
        anomalyData.append(random.uniform(maxList[4] + 5, maxList[4] + 7))  # 6
        anomalyData.append(random.uniform(maxList[5] + 8, maxList[5] + 10))  # 7
        anomalyData.append(random.uniform(maxList[6] + 7, maxList[6] + 11))  # 8
        anomalyData.append(maxList[7])  # 9
        anomalyData.append(maxList[8])  # 10
        anomalyData.append(random.uniform(maxList[9] + 5, maxList[9] + 10))  # 11
        anomalyData.append(random.uniform(maxList[10] + 5, maxList[10] + 10))  # 12
        anomalyData.append(random.uniform(maxList[11] + 5, maxList[11] + 10))  # 13
        anomalyData.append(maxList[12])  # 14
        anomalyData.append(random.uniform(maxList[13] + 5, maxList[13] + 10))  # 15
        anomalyData.append(random.uniform(maxList[14] + 5, maxList[14] + 10))  # 16
        anomalyData.append(random.uniform(maxList[15] + 5, maxList[15] + 10))  # 17
        anomalyData.append(random.uniform(maxList[16] + 15, maxList[16] + 20))  # 18
        anomalyData.append(random.uniform(maxList[17] + 5, maxList[17] + 10))  # 19
        anomalyData.append(maxList[18])  # 20
        anomalyData.append(random.uniform(maxList[19] + 5, maxList[19] + 10))  # 21
        anomalyData.append(maxList[20])  # 22
        anomalyData.append(maxList[21])  # 23
        anomalyData.append(random.uniform(maxList[22] + 5, maxList[22] + 8))  # 24
        anomalyData.append(random.uniform(maxList[23] + 8, maxList[23] + 10))  # 25
        
        anomalyList.append(anomalyData)
        i += 1
    
    # Generate min anomalies
    anomalyMinLabel = 102
    i = 1
    
    for _ in range(200):
        if i == limit:
            break
            
        anomalyData = []
        anomalyData.append(anomalyMinLabel)  # 0
        anomalyData.append(i)  # 1
        
        # Add anomaly values for each feature
        anomalyData.append(random.uniform(minList[0] - 0.1, minList[0] - 0.2))  # 2
        anomalyData.append(random.uniform(minList[1] - 0.1, minList[1] - 0.2))  # 3
        anomalyData.append(maxList[2])  # 4
        anomalyData.append(maxList[3])  # 5
        anomalyData.append(random.uniform(minList[4] - 2, minList[4] - 5))  # 6
        anomalyData.append(random.uniform(minList[5] - 5, minList[5] - 8))  # 7
        anomalyData.append(random.uniform(minList[6] - 5, minList[6] - 10))  # 8
        anomalyData.append(maxList[7])  # 9
        anomalyData.append(maxList[8])  # 10
        anomalyData.append(random.uniform(minList[9] - 3, minList[9] - 7))  # 11
        anomalyData.append(random.uniform(minList[10] - 2, minList[10] - 5))  # 12
        anomalyData.append(random.uniform(minList[11] - 3, minList[11] - 5))  # 13
        anomalyData.append(maxList[12])  # 14
        anomalyData.append(random.uniform(minList[13] - 1, minList[13] - 3))  # 15
        anomalyData.append(random.uniform(minList[14] - 1, minList[14] - 2))  # 16
        anomalyData.append(random.uniform(minList[15] - 3, minList[15] - 9))  # 17
        anomalyData.append(random.uniform(minList[16] - 10, minList[16] - 15))  # 18
        anomalyData.append(random.uniform(minList[17] - 3, minList[17] - 7))  # 19
        anomalyData.append(maxList[18])  # 20
        anomalyData.append(random.uniform(minList[19] - 1, minList[19] - 2))  # 21
        anomalyData.append(maxList[20])  # 22
        anomalyData.append(maxList[21])  # 23
        anomalyData.append(random.uniform(minList[22] - 0.3, minList[22] - 0.7))  # 24
        anomalyData.append(random.uniform(minList[23] - 0.2, minList[23] - 0.5))  # 25
        
        anomalyList.append(anomalyData)
        i += 1
    
    print("Anomaly generation complete")
    return anomalyList
