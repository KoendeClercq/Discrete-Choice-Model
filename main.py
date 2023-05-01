#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 12:25:37 2021
Edited on Fri Sep 17 18:06:24 2021

@author: gkdeclercq
"""
from datetime import datetime
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pySankey.sankey import sankey

from data import adjust_data_logit
from data import kmeans_clustering
from data import estimate_choice_model
from data import estimate_choice_model_nested
from data import estimate_modal_split
from data import estimate_modal_split_nested
from data import estimate_modal_split_nested_1
from data import estimate_modal_split_nested_2
from data import interpret_result
from data import cartesian

modes = ['car', 'carp', 'transit', 'cycle', 'walk']
dt = datetime.now().strftime("%Y%m%d-%H%M%S")

# Activate parts of analysis (0 == skip, 1 == don't skip)

sharedAV = 1
electricStep = 0

prepare_data = 0
cluster_data = 0
cl_comb = 1
visualize_clusters = 0

check_correlation = 0
estimate_model = 1 #
estimate_model_nested = 0
estimate_model_no_clusters = 0
prepare_frac = 0
estimate_split = 0 #
estimate_split_nested = 0
estimate_split_nested_1 = 0
estimate_split_nested_2 = 0
interpret_results = 0

NOTHING = 0

# Use these parameters to skip parts of analysis
nr_of_clusters_trip = 1
nr_of_clusters_pers = 6
k = 0
# dt = "20220419-131056"
# nest = 3

if sharedAV == 1:
    name_future = "Shared Autonomous Car"
else:
    name_future = "Electric Step"

# Future mode chars

# cost = [0, 4, 8, 12, 16, 20]
# cost = [0.05, 0.15, 0.25] # * distance (km)
# time = [10, 20, 40, 60, 80, 100] # (distance (km) / [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]) / 60 (from hours to minutes) 
# drivingtask = [0, 1]
# skills = [0, 1]
# weatherprotection = [0, 1]
# luggage = [0, 1]
# shared = [0, 1]
# availability = [0, 0.5, 1]
# reservation = [0, 1]
# active = [0, 1]
# accessible = [0, 1]


if sharedAV:
    
    # Shared autonomous vehicle

    cost = [0.05*0.8, 0.05*0.9, 0.05, 0.05*1.1, 0.05*1.2]
    time = [60*0.8, 60*0.9, 60, 60*1.1, 60*1.2]
    drivingtask = [0]
    skills = [0]
    weatherprotection = [1]
    luggage = [1]
    shared = [0]
    availability = [0.5*0.8, 0.5*0.9, 0.5, 0.5*1.1, 0.5*1.2]
    reservation = [1]
    active = [0]
    accessible = [1]

if electricStep:

    # Electric step
    
    cost = [4*0.8, 4*0.9, 4, 4*1.1, 4*1.2]
    time = [10*0.8, 10*0.9, 10, 10*1.1, 10*1.2]
    drivingtask = [1]
    skills = [0]
    weatherprotection = [0]
    luggage = [0]
    shared = [0]
    availability = [1*0.8, 1*0.9, 1]
    reservation = [0]
    active = [1]
    accessible = [0]

# Test combinations

# cost = [2]
# time = [1, 1.5, 2]
# drivingtask = [0, 1]
# skills = [0]
# weatherprotection = [1]
# luggage = [1]
# shared = [1]
# availability = [1]
# reservation = [1]
# active = [1]
# accessible = [1]

future_mode_char = cartesian((cost, time, drivingtask, skills, weatherprotection, luggage, shared, availability, reservation, active, accessible))

## PREPARE DATA

if prepare_data:
    adjust_data_logit(modes)


## CLUSTER ANALYSIS
    
if cluster_data:
    
    # Load and select traindata
    if k == 0:
        title = 'data/TrainData.csv'
    else:
        title = 'data/TrainData' + str(k) + '.csv'
        
    with open(title, newline='') as csvfile:
        X_full_train = list(csv.reader(csvfile))
    
    idx_choice = 60 # khvm
    
    labels = np.array(X_full_train[0])
    X_train = X_full_train[1:]
    y_train = [row[idx_choice] for row in X_train]
    y_train = np.array(y_train)
    y_train = np.array(y_train,dtype=float)
    X_train = np.array(X_train,dtype=float)
    
    # Load and select testdata
    title = 'data/TestData.csv'
        
    with open(title, newline='') as csvfile:
        X_full_test = list(csv.reader(csvfile))
    
    labels = np.array(X_full_test[0])
    X_test = X_full_test[1:]
    y_test = [row[idx_choice] for row in X_test]
    y_test = np.array(y_test)
    y_test = np.array(y_test,dtype=float)
    X_test = np.array(X_test,dtype=float)
    
    
    # idx_pers = np.hstack((np.arange(54, 64), np.arange(74, 90), np.arange(116, 130), np.arange(140, 150)))
    # idx_trip = np.hstack((np.arange(64, 74), np.arange(130, 139), np.arange(151, 159)))
    idx_pers = [50, 52, 53, 54, 55, 56, 57, 58, 59, 80, 81]
    idx_trip = [49, 51, 61, 62, 63, 64, 65, 66, 73]
    idx_comb = [50, 52, 53, 54, 55, 56, 57, 58, 59, 80, 81, 49, 51, 61, 62, 63, 64, 65, 66, 73]
    if cl_comb:
        idx_pers = idx_comb

    # Cluster analysis
    X_trip_train = X_train[:, idx_trip]
    X_trip_test = X_test[:, idx_trip]
    labels_trip = labels[idx_trip]
    clusters_trip_train, clusters_trip_test, nr_of_clusters_trip = kmeans_clustering("trip characteristics", X_trip_train, X_trip_test, labels_trip)
    
    X_pers_train = X_train[:, idx_pers]
    X_pers_test = X_test[:, idx_pers]
    labels_pers = labels[idx_pers]
    clusters_pers_train, clusters_pers_test, nr_of_clusters_pers = kmeans_clustering("personal characteristics", X_pers_train, X_pers_test, labels_pers)
    
    # Save results traindata
    clusters_trip_str = np.insert(clusters_trip_train.astype(np.str), 0, 'trip_cl')
    clusters_pers_str = np.insert(clusters_pers_train.astype(np.str), 0, 'pers_cl')
    X_full_train_1 = np.insert(X_full_train, 0, clusters_trip_str, axis=1)
    X_full_train_2 = np.insert(X_full_train_1, 0, clusters_pers_str, axis=1)
    
    df = pd.DataFrame(X_full_train_2[1:], columns=X_full_train_2[0])
    if k == 0:
        df.to_csv('data/data_cl_train.csv', index=False)
    else:
        df.to_csv('data/data_cl' + str(k) + '_train.csv', index=False)
    
    # Save results testdata
    clusters_trip_str = np.insert(clusters_trip_test.astype(np.str), 0, 'trip_cl')
    clusters_pers_str = np.insert(clusters_pers_test.astype(np.str), 0, 'pers_cl')
    X_full_test_1 = np.insert(X_full_test, 0, clusters_trip_str, axis=1)
    X_full_test_2 = np.insert(X_full_test_1, 0, clusters_pers_str, axis=1)
    
    df = pd.DataFrame(X_full_test_2[1:], columns=X_full_test_2[0])
    df.to_csv('data/data_cl_test.csv', index=False)
    
    print('Finished clustering data')


if visualize_clusters:
    
    df = pd.read_csv('data/data_cl_train.csv')
    
    df = df[0:1000]
        
    personal_chars = ["ovstkaart", "d_hhchildren", "d_high_educ", "gender", "driving_license", "car_ownership", "main_car_user", "hh_highinc20", "hh_lowinc20", "AGE1E", "AGE2E"]
    trip_chars = ["sted_o", "weekday", "pur_home", "pur_work", "pur_busn", "pur_other", "departure_rain", "arrival_rain", "sted_d"]
    
    for i in range(5):
        
        df_pers = df.loc[df["pers_cl"] == i]
        df_trip = df.loc[df["trip_cl"] == i]
        
        df_pers = df_pers[personal_chars]
        df_trip = df_trip[trip_chars]
    
        hist = df_pers.hist(bins=10)
        plt.savefig('data/hist_pers_' + str(i) + '.png')
        
        hist = df_trip.hist(bins=10)
        plt.savefig('data/hist_trip_' + str(i) + '.png')
    

## CHECK CORRELATION

if check_correlation:
    
    df = pd.read_csv('data/data_cl_train.csv')
    
    # df = df[0:1000]
    
    # Estimate distance trip
    distance_estimated = []
    
    for i in range(len(df)):

        distance_trip = [0, 0, 0, 0, 0]
        distance_trip_counter = 0
        assumed_average_speed = [60, 60, 40, 15, 5] # [car, carp, transit, cycle, walk]
        
        for j in range(5):
            # Estimate distance in dataset based on travel times of all existing modes
            # average of (time for available mode) / (assumed average speed for available mode)
            if (df.iloc[i]["time_" + modes[j]] != 9999):
                distance_trip[j] = df.iloc[i]["time_" + modes[j]] / assumed_average_speed[j]
                distance_trip_counter += 1
            else:
                distance_trip[j] = 0

        # Estimate distance in dataset based on travel times of all existing modes
        # average of (time for available mode) / (assumed average speed for available mode)
        distance_estimated.append(sum(distance_trip) / distance_trip_counter)
    
    df["est_dist"] = distance_estimated
    
    # Add future mode in data 
    # ALWAYS CHECK IF INDICES ARE CORRECT!
    
    # Cost per trip
    df["cost_future"] = cost[2]
    # Cost per km
    # df["cost_future"] = cost[2] * df["est_dist"]
    
    df["time_future"] = time[2]
    df["drivingtask_future"] = drivingtask[0]
    df["skills_future"] = skills[0]
    df["weatherProtection_future"] = weatherprotection[0]
    df["luggage_future"] = luggage[0]
    df["shared_future"] = shared[0]
    df["availability_future"] = availability[2]
    df["reservation_future"] = reservation[0]
    df["active_future"] = active[0]
    df["accessible_future"] = accessible[0]
    
    
    # Normalize data where normalisation is needed (i.e. not already between 0 and 1)
    for i in ["cost", "time"]:       
        df_temp = df[[i + "_" + modes[0], i + "_" + modes[1], i + "_" + modes[2], i + "_" + modes[3], i + "_" + modes[4], i + "_" + "future"]]
        df_temp.dropna()
        
        min_df = df_temp.min().min()
        df_temp[df_temp > 4999] = min_df
        max_df = df_temp.max().max()
        diff_df = max(max_df - min_df, 1)
        df[[i + "_" + modes[0], i + "_" + modes[1], i + "_" + modes[2], i + "_" + modes[3], i + "_" + modes[4], i + "_" + "future"]] = (df[[i + "_" + modes[0], i + "_" + modes[1], i + "_" + modes[2], i + "_" + modes[3], i + "_" + modes[4], i + "_" + "future"]] - min_df ) / diff_df
        
    all_modes = ["car", "carp", "transit", "cycle", "walk", "future"]
    
    # for i in range(len(df)):
    distance = np.zeros((6, 6))
    
    for j in range(len(all_modes)):
        for k in range(len(all_modes)):
            # Exception when one of the values is higher than 4999 for cost and time (i.e. when those modes are not available in those entries)
            cost_mtx = np.power(df["cost_" + all_modes[j]] - df["cost_" + all_modes[k]], 2)
            cost_mtx[cost_mtx > 1] = 1
            time_mtx = np.power(df["time_" + all_modes[j]] - df["time_" + all_modes[k]], 2)
            time_mtx[time_mtx > 1] = 1
            
            drivtask_mtx = np.abs(df["drivingtask_" + all_modes[j]] - df["drivingtask_" + all_modes[k]])
            skills_mtx = np.abs(df["skills_" + all_modes[j]] - df["skills_" + all_modes[k]])
            wp_mtx = np.abs(df["weatherProtection_" + all_modes[j]] - df["weatherProtection_" + all_modes[k]])
            luggage_mtx = np.abs(df["luggage_" + all_modes[j]] - df["luggage_" + all_modes[k]])
            shared_mtx = np.abs(df["shared_" + all_modes[j]] - df["shared_" + all_modes[k]])
            av_mtx = np.abs(df["availability_" + all_modes[j]] - df["availability_" + all_modes[k]])
            res_mtx = np.abs(df["reservation_" + all_modes[j]] - df["reservation_" + all_modes[k]])
            act_mtx = np.abs(df["active_" + all_modes[j]] - df["active_" + all_modes[k]])
            acc_mtx = np.abs(df["accessible_" + all_modes[j]] - df["accessible_" + all_modes[k]])    
            
            distance[j][k] = np.average(cost_mtx + time_mtx + drivtask_mtx + skills_mtx + wp_mtx + luggage_mtx + shared_mtx + av_mtx + res_mtx + act_mtx + acc_mtx) / 11
    
    similarity_matrix = 1 - distance
    
    with open("data/similarity_matrix_" + dt + ".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(similarity_matrix)
    
    # Determine most similar existing mode
    max_sim = similarity_matrix[5, 0:5].max()
    temp_arr = similarity_matrix[5, 0:5]
    nest = temp_arr.argmax()
    
    print('Finished calculating correlation indices')


## ESTIMATE CHOICE MODEL
 
if estimate_model:
        
    for act_asc in range(1): # 0 == INACTIVE || 1 == ACTIVE (for both: range(2))    
        for i in range(1): # range(nr_of_clusters_trip):
                for j in range(nr_of_clusters_pers):
                    dt = estimate_choice_model(modes, dt, k, i, j, act_asc, 1)
                
    print('Finished estimating model')


if estimate_model_nested:
        
    for act_asc in range(1): # 0 == INACTIVE || 1 == ACTIVE (for both: range(2))    
        for i in range(1): # range(nr_of_clusters_trip):
                for j in range(nr_of_clusters_pers):
                    dt = estimate_choice_model_nested(modes, dt, k, i, j, act_asc, 1)
                
    print('Finished estimating model')
   

if estimate_model_no_clusters:
    
    dt = estimate_choice_model(modes, dt)
    
    print('Finished estimating model without clusters')
    
    
## ESTIMATE MODAL SPLIT
    
if prepare_frac:
    
    # Reduce test size estimation due to overload of computer and computational issues
    final = pd.DataFrame()
    for i in range(nr_of_clusters_trip):
        for j in range(nr_of_clusters_pers):
            df = pd.read_csv('data/data_cl_test.csv')
            df_temp = df[df["trip_cl"] == i]
            df_selection = df_temp[df_temp["pers_cl"] == j]
            # Shuffle
            df_selection = df_selection.sample(frac=0.1)
            frac_list = [1] * len(df_selection)
            for k in range(round(len(df_selection)/3), round(len(df_selection)*2/3)):
                frac_list[k] = 2
            for k in range(round(len(df_selection)*2/3), len(df_selection)):
                frac_list[k] = 3
            df_selection['frac'] = frac_list
            final = pd.concat([final, df_selection])
    final.to_csv('data/data_cl_test_frac.csv', index=False)
    
    print('Finished preparing fractionizing test data')
    
if estimate_split:
    
    frac = 1 # TODO: implement loop to automatically do all 3 fractions of testdata + add exception for when sample size of cluster is 0... (or 1?)
    
    for subset in ['Test']: #, 'Train']:
        for act_asc in range(1): # 0 == INACTIVE || 1 == ACTIVE (for both: range(2))
            
            results = pd.DataFrame()
            results_est = pd.DataFrame()
            results_future = pd.DataFrame(columns=['idx_future_mode_char','trip','pers','size','car','carp','transit','cycle','walk','future'])
            
            results_stats = pd.DataFrame(columns=['InitLog', 'FinalLog', 'R2', 'R2bar', 'BIC', 'trip', 'pers', 'R2-est'])
        
        
            for i in range(1): # range(nr_of_clusters_trip):
                for j in range(nr_of_clusters_pers):
                    # try:
                    res_cl, res_pd, gs, modal_split_future, R_square, res_future = estimate_modal_split(modes, k, i, j, frac, act_asc, dt, subset, future_mode_char)   
                    
                    res_cl_formatted = pd.DataFrame(0, index=['1', '2', '3', '4', '5', '6', 'accuracy', 'macro avg', 'weighted avg'], columns=['precision', 'recall', 'f1-score', 'support', 'trip', 'pers', 'modal_split_estimated', 'modal_split_actual', 'modal_split_future'])
                    try: 
                        res_cl_formatted.loc['1'] = res_cl.loc['1']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['2'] = res_cl.loc['2']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['3'] = res_cl.loc['3']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['4'] = res_cl.loc['4']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['5'] = res_cl.loc['5']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['6'] = res_cl.loc['6']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['1'] = res_cl.loc['1.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['2'] = res_cl.loc['2.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['3'] = res_cl.loc['3.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['4'] = res_cl.loc['4.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['5'] = res_cl.loc['5.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['6'] = res_cl.loc['6.0']
                    except:
                        pass
                        
                    res_cl_formatted.loc['6'] = ['0', '0', '0', '0', '0', '0', '0', '0', modal_split_future]
                                        
                    res_cl_formatted.loc['accuracy'] = res_cl.loc['accuracy']
                    res_cl_formatted.loc['macro avg'] = res_cl.loc['macro avg']
                    res_cl_formatted.loc['weighted avg'] = res_cl.loc['weighted avg']
                    
                    results = pd.concat([results, res_cl_formatted])
                    results_est = pd.concat([results_est, res_pd])
                    results_stats.loc['{0}{1}'.format(i, j)] = pd.Series({'InitLog':gs['Init log likelihood'][0],
                                                        'FinalLog':gs['Final log likelihood'][0],
                                                        'R2':gs['Rho-square for the init. model'][0],
                                                        'R2bar':gs['Rho-square-bar for the init. model'][0],
                                                        'BIC':gs['Bayesian Information Criterion'][0],
                                                        'trip': i,
                                                        'pers': j,
                                                        'R2-est': R_square})
                    
                    res_future_temp = pd.DataFrame(res_future, columns=['idx_future_mode_char','trip','pers','size','car','carp','transit','cycle','walk','future'])
                    # Save results per cluster
                    res_future_temp.to_csv("data/" + subset + "_results_future_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_trip' + str(i) + '_pers' + str(j) + '_' + dt + ".csv")
                    results_future = pd.concat([results_future, res_future_temp])
                    # except:
                    #     print("Estimating future modal split for trip cluster", i, "and personal cluster", j, "failed")
        
            results.to_csv("data/" + subset + "_results_performance_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_' + dt + ".csv")
            results_est.to_csv("data/" + subset + "_results_parameters_estimation_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_' + dt + ".csv")
            results_stats.to_csv("data/" + subset + "_results_stats_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_' + dt + ".csv")
            results_future.to_csv("data/" + subset + "_results_future_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_' + dt + ".csv")
            
            results_stats = pd.DataFrame()
        
    print('Finished estimating modal split')
    


## ESTIMATE SPLIT NESTED LOGIT    

if estimate_split_nested:
    
    frac = 1 # TODO: implement loop to automatically do all 3 fractions of testdata + add exception for when sample size of cluster is 0... (or 1?)
    # nest = 3
    # max_sim = 0.8824182705342121
    mu_b = 10 
    
    for subset in ['Test']: #, 'Train']:
        for act_asc in range(1): # 0 == INACTIVE || 1 == ACTIVE (for both: range(2))
            
            results = pd.DataFrame()
            results_est = pd.DataFrame()
            results_future = pd.DataFrame(columns=['idx_future_mode_char','trip','pers','size','car','carp','transit','cycle','walk','future'])
            
            results_stats = pd.DataFrame(columns=['InitLog', 'FinalLog', 'R2', 'R2bar', 'BIC', 'trip', 'pers', 'R2-est'])
        
        
            for i in range(1): # range(nr_of_clusters_trip):
                for j in range(nr_of_clusters_pers):
                    # try:
                    if estimate_split_nested_1:
                        res_cl, res_pd, gs, modal_split_future, R_square, res_future = estimate_modal_split_nested_1(modes, k, i, j, frac, act_asc, dt, subset, future_mode_char, nest, max_sim, mu_b)
                    elif estimate_split_nested_2:
                        res_cl, res_pd, gs, modal_split_future, R_square, res_future = estimate_modal_split_nested_2(modes, k, i, j, frac, act_asc, dt, subset, future_mode_char, nest, max_sim, mu_b)
                    else:
                        res_cl, res_pd, gs, modal_split_future, R_square, res_future = estimate_modal_split_nested(modes, k, i, j, frac, act_asc, dt, subset, future_mode_char, nest, max_sim)
                    
                    
                    res_cl_formatted = pd.DataFrame(0, index=['1', '2', '3', '4', '5', '6', 'accuracy', 'macro avg', 'weighted avg'], columns=['precision', 'recall', 'f1-score', 'support', 'trip', 'pers', 'modal_split_estimated', 'modal_split_actual', 'modal_split_future'])
                    try: 
                        res_cl_formatted.loc['1'] = res_cl.loc['1']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['2'] = res_cl.loc['2']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['3'] = res_cl.loc['3']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['4'] = res_cl.loc['4']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['5'] = res_cl.loc['5']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['6'] = res_cl.loc['6']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['1'] = res_cl.loc['1.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['2'] = res_cl.loc['2.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['3'] = res_cl.loc['3.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['4'] = res_cl.loc['4.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['5'] = res_cl.loc['5.0']
                    except:
                        pass
                    
                    try: 
                        res_cl_formatted.loc['6'] = res_cl.loc['6.0']
                    except:
                        pass
                    
                    res_cl_formatted.loc['6'] = ['0', '0', '0', '0', '0', '0', '0', '0', modal_split_future]
                                        
                    res_cl_formatted.loc['accuracy'] = res_cl.loc['accuracy']
                    res_cl_formatted.loc['macro avg'] = res_cl.loc['macro avg']
                    res_cl_formatted.loc['weighted avg'] = res_cl.loc['weighted avg']
                    
                    results = pd.concat([results, res_cl_formatted])
                    results_est = pd.concat([results_est, res_pd])
                    results_stats.loc['{0}{1}'.format(i, j)] = pd.Series({'InitLog':gs['Init log likelihood'][0],
                                                        'FinalLog':gs['Final log likelihood'][0],
                                                        'R2':gs['Rho-square for the init. model'][0],
                                                        'R2bar':gs['Rho-square-bar for the init. model'][0],
                                                        'BIC':gs['Bayesian Information Criterion'][0],
                                                        'trip': i,
                                                        'pers': j,
                                                        'R2-est': R_square})
                    
                    res_future_temp = pd.DataFrame(res_future, columns=['idx_future_mode_char','trip','pers','size','car','carp','transit','cycle','walk','future'])
                    # Save results per cluster
                    res_future_temp.to_csv("data/" + subset + "_results_future_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_trip' + str(i) + '_pers' + str(j) + '_' + dt + ".csv")
                    results_future = pd.concat([results_future, res_future_temp])
                    # except:
                    #     print("Estimating future modal split for trip cluster", i, "and personal cluster", j, "failed")
        
            results.to_csv("data/" + subset + "_results_performance_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_' + dt + ".csv")
            results_est.to_csv("data/" + subset + "_results_parameters_estimation_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_' + dt + ".csv")
            results_stats.to_csv("data/" + subset + "_results_stats_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_' + dt + ".csv")
            results_future.to_csv("data/" + subset + "_results_future_par" + str(gs['Number of estimated parameters'][0]) + "_asc" + str(act_asc) + '_' + dt + ".csv")
            
            results_stats = pd.DataFrame()
        
    print('Finished estimating modal split')

    
## INTERPRET RESULTS
    
if interpret_results:
    
    nrOfClusters = 1 * nr_of_clusters_pers
    
    # ADJUST FOR EACH UTILITY FUNCTION
    for subset in['Test']: #, 'Train']:
        for j in range(2):
            for i in range(150):
                try:
                    interpret_result(i, j, nrOfClusters, dt, subset, future_mode_char)
                except:
                    pass
    
    # Combine choice datasets
    small_dfs = []
    for i in range(1):
        for j in range(nr_of_clusters_pers):
            small_dfs.append(pd.read_csv("data/CHOICE" + str(i) + str(j) + ".csv"))

    combined_csv = pd.concat(small_dfs, ignore_index=True)    
    combined_csv.to_csv("data/combined_choice.csv", index=False, encoding='utf-8-sig')
    
    text = open("data/combined_choice.csv", "r")
    text = ''.join([i for i in text]) \
        .replace("1.0", "Car")
    text = ''.join([i for i in text]) \
        .replace("2.0", "Carpool")
    text = ''.join([i for i in text]) \
        .replace("3.0", "Transit")
    text = ''.join([i for i in text]) \
        .replace("4.0", "Cycle")
    text = ''.join([i for i in text]) \
        .replace("5.0", "Walk")
    text = ''.join([i for i in text]) \
        .replace("6.0", name_future)
    x = open("data/combined_choice_edited.csv","w")
    x.writelines(text)
    x.close()
    
    df = pd.read_csv("data/combined_choice_edited.csv", sep=",")

    # Create Sankey diagram
    sankey(
        df["Actual"], df["Future"], 
        aspect=20, fontsize=16
    )
    
    # Get current figure
    fig = plt.gcf()
    
    # Set size in inches
    fig.set_size_inches(6, 6)
    
    # Set the color of the background to white
    fig.set_facecolor("w")
    
    # Save the figure
    fig.savefig("Sankey.png", bbox_inches="tight", dpi=150)
        
    print('Finished summarising results')
        
        
    
    
