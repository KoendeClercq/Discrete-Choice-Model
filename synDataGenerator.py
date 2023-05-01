#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:17:58 2021

This file generates a synthetic dataset of mode choice with 3 different modes with some basic trip and personal characteristic.

@author: gkdeclercq
"""

import csv
import numpy as np
import pandas as pd

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    Source: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out



############### Initialize parameters ###############
    

# Trip characteristics
distance = np.arange(0.5, 101, 1) # [km]

# Personal characteristics
age = np.arange(18, 91, 1) # [year]
income = np.arange(10000, 205000, 10000) # [euro/year]
gender = 1 #np.arange(-0.5, 1, 1) # [-] -0.5 = male, 0.5 = female

dataset = cartesian((distance, age, income, gender))

# Mode characteristics
U = [0, 0, 0, 0, 0, 0]
E = [0, 0, 0, 0, 0]
E_future = [0, 0, 0, 0, 0, 0]
speed = [0, 0, 0, 0, 0, 0]
time = [0, 0, 0, 0, 0, 0]
cost = [0, 0, 0, 0, 0, 0]
CHOICE = [0, 0, 0, 0, 0]
CHOICE_future = [0, 0, 0, 0, 0, 0]
data = []
data.append(["distance", "age", "income", "gender", "time_car", "cost_car", "time_carp", "cost_carp", "time_transit", "cost_transit", "time_cycle", "cost_cycle", "time_walk", "cost_walk", "time_future", "cost_future", "chance_car", "change_carp", "chance_transit", "chance_cycle", "chance_walk", "chance_car_future", "change_carp_future", "chance_transit_future", "chance_cycle_future", "chance_walk_future", "chance_future_future", "CHOICE", "CHOICE_future", "trip", "pers"])

# Parameters
B_COST = -0.2
B_TIME = -0.21

for j in range(len(dataset)):
    
    # Car
    speed[0] = 60 # [km/hour]
    time[0] = dataset[j, 0] / speed[0] # [hour]
    cost[0] = dataset[j, 0] / 2 # [euro]
    
    # Carpool
    speed[1] = 50 # [km/hour]
    time[1] = (dataset[j, 0] / speed[1]) # [hour]
    cost[1] = dataset[j, 0] / 4 # [euro]
    
    # Transit
    speed[2] = 40 # [km/hour]
    time[2] = dataset[j, 0] / speed[2] # [hour]
    cost[2] = dataset[j, 0] / 8 # [euro]
    
    # Bicycle
    speed[3] = 30 # [km/hour]
    time[3] = dataset[j, 0] / speed[3] # [hour]
    cost[3] = dataset[j, 0] / 16 # [euro]
    
    # Walk
    speed[4] = 20 # [km/hour]
    time[4] = dataset[j, 0] / speed[4] # [hour]
    cost[4] = 0 # Free
    
    # Future mode
    speed[5] = 25 # [km/hour]
    time[5] = dataset[j, 0] / speed[5] # [hour]
    cost[5] = dataset[j, 0] / 32 # [euro]
    
    
    # Utility function
    for i in range(6):
        U[i] = B_TIME * dataset[j, 1] * time[i] + B_COST * 200000 / dataset[j, 2] * cost[i]
    
    # Utilities without future mode    
    for i in range(5):
        E[i] = np.exp(U[i]) / (np.exp(U[0]) + np.exp(U[1]) + np.exp(U[2]) + np.exp(U[3]) + np.exp(U[4]))
    
    # Utilities with future mode    
    for i in range(6):
        E_future[i] = np.exp(U[i]) / (np.exp(U[0]) + np.exp(U[1]) + np.exp(U[2]) + np.exp(U[3]) + np.exp(U[4]) + np.exp(U[5]))


    # Mode choice
    mode_choice = E.index(max(E))
    
    if mode_choice == 0:
        CHOICE = [CHOICE[0] + 1, CHOICE[1], CHOICE[2], CHOICE[3], CHOICE[4]]
    if mode_choice == 1:
        CHOICE = [CHOICE[0], CHOICE[1] + 1, CHOICE[2], CHOICE[3], CHOICE[4]]
    if mode_choice == 2:
        CHOICE = [CHOICE[0], CHOICE[1], CHOICE[2] + 1, CHOICE[3], CHOICE[4]]
    if mode_choice == 3:
        CHOICE = [CHOICE[0], CHOICE[1], CHOICE[2], CHOICE[3] + 1, CHOICE[4]]
    if mode_choice == 4:
        CHOICE = [CHOICE[0], CHOICE[1], CHOICE[2], CHOICE[3], CHOICE[4] + 1]
    
    
    # Mode choice future
    mode_choice_future = E_future.index(max(E_future))
    
    if mode_choice_future == 0:
        CHOICE_future = [CHOICE_future[0] + 1, CHOICE_future[1], CHOICE_future[2], CHOICE_future[3], CHOICE_future[4], CHOICE_future[5]]
    if mode_choice_future == 1:
        CHOICE_future = [CHOICE_future[0], CHOICE_future[1] + 1, CHOICE_future[2], CHOICE_future[3], CHOICE_future[4], CHOICE_future[5]]
    if mode_choice_future == 2:
        CHOICE_future = [CHOICE_future[0], CHOICE_future[1], CHOICE_future[2] + 1, CHOICE_future[3], CHOICE_future[4], CHOICE_future[5]]
    if mode_choice_future == 3:
        CHOICE_future = [CHOICE_future[0], CHOICE_future[1], CHOICE_future[2], CHOICE_future[3] + 1, CHOICE_future[4], CHOICE_future[5]]
    if mode_choice_future == 4:
        CHOICE_future = [CHOICE_future[0], CHOICE_future[1], CHOICE_future[2], CHOICE_future[3], CHOICE_future[4] + 1, CHOICE_future[5]]
    if mode_choice_future == 5:
        CHOICE_future = [CHOICE_future[0], CHOICE_future[1], CHOICE_future[2], CHOICE_future[3], CHOICE_future[4], CHOICE_future[5] + 1]    
    
    data.append([dataset[j, 0], dataset[j, 1], dataset[j, 2], dataset[j, 3], time[0], cost[0], time[1], cost[1], time[2], cost[2], time[3], cost[3], time[4], cost[4], time[5], cost[5], E[0], E[1], E[2], E[3], E[4], E_future[0], E_future[1], E_future[2], E_future[3], E_future[4], E_future[5], mode_choice + 1, mode_choice_future + 1, 0, 0])
    
    
    print("Progress: {:.2f}".format(j/len(dataset)))

# CHOICE[:] = [x / len(dataset) for x in CHOICE]
# print(CHOICE)

CHOICE_future[:] = [x / len(dataset) for x in CHOICE_future]
print(CHOICE_future)

with open('syntheticData.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)    

# Create training and testing datasets

data_pd = pd.read_csv('syntheticData.csv')

data_pd = data_pd.dropna()
index = data_pd.index
entries = len(index)
entries_train = round(entries*4/5)
entries_test = entries_train + 1

# Shuffle data rows to ensure train and test datasets are less biased
data_pd = data_pd.sample(frac=1)

data_train_pd = data_pd[0:entries_train] # 80% of entries
data_test_pd = data_pd[entries_test:entries] # 20% of entries

data_pd.to_csv("data/FullData.csv", index=False)
data_train_pd.to_csv("data/TrainData.csv", index=False)
data_test_pd.to_csv("data/TestData.csv", index=False)

entries_train_1 = round(entries_train*1/3)
entries_train_2 = round(entries_train*2/3)

data_train_1_pd = data_pd[0:entries_train_1]
data_train_2_pd = data_pd[entries_train_1:entries_train_2]
data_train_3_pd = data_pd[entries_train_2:entries_train]

data_train_1_pd.to_csv("data/TrainData1.csv", index=False)
data_train_2_pd.to_csv("data/TrainData2.csv", index=False)
data_train_3_pd.to_csv("data/TrainData3.csv", index=False)


print("Finished creating datasets")
