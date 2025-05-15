# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 16:31:07 2025

@author: pnapi

Program used for >this and that<
"""

import numpy as np

def correlation(data_set1, data_set2):
    
    if len(data_set1) != len(data_set2): 
        raise ValueError(
            f"Data sets lengths don't match -> {len(data_set1)}//{len(data_set2)}")
    
    mean1, mean2 = np.mean(data_set1), np.mean(data_set2)
    
    up = sum((data_set1[i] - mean1) * (data_set2[i] - mean2) for i in range(len(data_set1)))
    
    low1sum = sum((data_set1[i] - mean1)**2 for i in range(len(data_set1)))
    low2sum = sum((data_set2[i] - mean2)**2 for i in range(len(data_set2)))
    
    return up / (np.sqrt(low1sum) * np.sqrt(low2sum))

def variance(data_set):
    mean = np.mean(data_set)
    return sum((i - mean)**2 / len(data_set) for i in data_set)

def standard_deviation(data_set):
    return variance(data_set)**0.5

