import os
import re
import csv
import pandas as pd
import numpy as np
import math
import random
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.results as res
from biogeme.expressions import Beta
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import r2_score
from pySankey.sankey import sankey


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



def adjust_data_logit(modes):
    
    data = pd.read_csv("data/data_original_corrected.csv")
    
    # Replace all -1 with 0, since in a utility function it would not have any effect
    data.replace(-1, 0, inplace=True)
    
    data.insert(0, "cost_car", 0)
    data.insert(0, "cost_carp", 0)
    data.insert(0, "cost_walk", 0)
    data.insert(0, "cost_cycle", 0)
    
    data["cost_car"] = data["tc_car"]
    data["cost_carp"] = data["tc_car"] / 2
    
    data.rename({'t_car': 'time_car', 't_carp': 'time_carp', 't_transit': 'time_transit', 't_cycle': 'time_cycle', 't_walk': 'time_walk', 'c_transit': 'cost_transit'}, axis='columns', inplace=True)
    
    
    for i in range(5):
        
        # Add characteristic columns (exluded for now: waitingtime, delay, emissions & noise)
        data.insert(0, "drivingtask_" + modes[i], 0) # active driving task
        data.insert(0, "skills_" + modes[i], 0) # need for (digital) skills
        data.insert(0, "weatherProtection_" + modes[i], 0)
        data.insert(0, "luggage_" + modes[i], 0) # space for luggage
        data.insert(0, "shared_" + modes[i], 0)
        data.insert(0, "availability_" + modes[i], 1)
        data.insert(0, "reservation_" + modes[i], 1) # DONE, 1 if reservation is available or not needed (practical availability of 100%)
        data.insert(0, "active_" + modes[i], 0)
        data.insert(0, "accessible_" + modes[i], 0)
        
        
        # Add info of characteristics
        # use value of i (mode choice) to determine non-zero values; 0: car, 1:carp, 2:transit, 3:cycle, 4:walk        
        if i == 0: 
            data["drivingtask_car"] = 1 # DONE
            data["skills_car"] = 1 # DONE
            data["weatherProtection_car"] = 1 # DONE
            data["luggage_car"] = 1 # DONE
        
        if i == 1:
            data["availability_carp"] = 0.1 # DONE, estimated due to dependency on other driver
            data["weatherProtection_carp"] = 1 # DONE
            data["luggage_carp"] = 1 # DONE
            data["shared_carp"] = 1 # DONE
            data["accessible_carp"] = 1 # DONE
        
        if i == 2:
            data["availability_transit"] = data["sted_o"] * data["sted_d"] / 25 #DONE, estimated based on urban density of origin and destination
            data["weatherProtection_transit"] = 1 # DONE
            data["luggage_transit"] = 0.5 # DONE, estimated
            data["shared_transit"] = 1 # DONE
            data["reservation_transit"] = 0 # DONE
            data["accessible_transit"] = 1 # DONE
            
        if i == 3:
            data["drivingtask_cycle"] = 1 # DONE
            data["active_cycle"] = 1 # DONE
        
        if i == 4:
            data["active_walk"] = 1 # DONE  
    
    
    data['departure_rain'] = pd.to_numeric(data['departure_rain'],errors='coerce')
    data['arrival_rain'] = pd.to_numeric(data['arrival_rain'],errors='coerce')
    
    data.loc[data['departure_rain'] > 1, 'departure_rain'] = 1
    data.loc[data['arrival_rain'] > 1, 'arrival_rain'] = 1
    
    data = data.dropna()
    index = data.index
    entries = len(index)
    entries_train = round(entries*4/5)
    entries_test = entries_train + 1
    
    # Shuffle data rows to ensure train and test datasets are less biased
    data = data.sample(frac=1)
    
    data_train = data[0:entries_train] # 80% of entries
    data_test = data[entries_test:entries] # 20% of entries
    
    data.to_csv("data/FullData.csv", index=False)
    data_train.to_csv("data/TrainData.csv", index=False)
    data_test.to_csv("data/TestData.csv", index=False)
    
    entries_train_1 = round(entries_train*1/3)
    entries_train_2 = round(entries_train*2/3)
    
    data_train_1 = data[0:entries_train_1]
    data_train_2 = data[entries_train_1:entries_train_2]
    data_train_3 = data[entries_train_2:entries_train]

    data_train_1.to_csv("data/TrainData1.csv", index=False)
    data_train_2.to_csv("data/TrainData2.csv", index=False)
    data_train_3.to_csv("data/TrainData3.csv", index=False)
    
    print("Finished preparing data")
    


def kmeans_clustering(char_cat, data, data_test, labels):

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_test_scaled = scaler.fit_transform(data_test)
    
    
    # Elbow technique
    
    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        # define the model
        model = KMeans(
            init="random",
            n_clusters=k,
            n_init=10,
            max_iter=300,
            random_state=42)
        # fit the model
        model.fit(data_scaled)
        sse.append(model.inertia_)
    
    # number of clusters where curvature of line is the largest
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.title("Clusters " + char_cat + ", elbow found at " + str(kl.elbow))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig('figures/Clusters ' + char_cat + ", elbow found at " + str(kl.elbow) + '.png',bbox_inches='tight', dpi=300)
    plt.show()
        
    # Run model with kl.elbow clusters
    
    model = KMeans(
        init="random",
        n_clusters=kl.elbow,
        n_init=10,
        max_iter=300,
        random_state=42)
    # fit the model
    model.fit(data_scaled)
    sse.append(model.inertia_)
    # assign a cluster to each example
    yhat = model.predict(data_scaled)
    yhat_test = model.predict(data_test_scaled)
    
    return yhat, yhat_test, kl.elbow



def estimate_choice_model(modes, dt, k=0, trip_cl=99, pers_cl=99, act_asc=0, clusters=0):
    
    if k == 0:
        df = pd.read_csv('data/data_cl_train.csv')
    else:
        df = pd.read_csv('data/data_cl' + str(k) + '_train.csv')
    
    # df = pd.read_csv('data/data_original.csv')
    # df = pd.read_csv('data/TrainData.csv')
    
    if clusters:
        # df_temp = df[df["trip_cl"] == trip_cl] # When doing double clusters, trip & pers
        df_temp = df # When analyzing only pers clusters
        df_selection = df_temp[df_temp["pers_cl"] == pers_cl]
    else:
        df_selection = df
    # df_selection = df
    
    database = db.Database('traindataAI', df_selection)
    globals().update(database.variables)
    
    
    ############### ADJUST BELOW THIS LINE ###############

    B = [0] * 110
    for i in range(110):
        B[i] = Beta('B_{}'.format(i), ((random.randint(0, 100) - 50 )/ 100), None, None, 0)

    # Definition of the utility functions
    V_temp = [0] * 5
    
    
    for i in range(5):
          # Added all characteristics (R2 = , no clusters: R2 = )
          V_temp[i] = B[0] * globals()["cost_" + modes[i]] + B[1] * globals()["time_" + modes[i]] + \
              B[2] * globals()["driving_license"] + B[3] * globals()["drivingtask_" + modes[i]] + \
              B[4] * globals()["skills_" + modes[i]] + B[5] * globals()["weatherProtection_" + modes[i]] + \
              B[6] * globals()["luggage_" + modes[i]] + B[7] * globals()["shared_" + modes[i]] + \
              B[8] * globals()["availability_" + modes[i]] + B[9] * globals()["reservation_" + modes[i]] + \
              B[10] * globals()["active_" + modes[i]] + B[11] * globals()["accessible_" + modes[i]]
            
            
          # Synthetic data utility function (no clusters: R2 = 0.999)
          # V_temp[i] = B_TIME * globals()["age"] * globals()["time_" + modes[i]] + B_COST * 200000 / globals()["income"] * globals()["cost_" + modes[i]] # Exactly the same as in data generation
    
    ############### ADJUST ABOVE THIS LINE ###############
    
    # # Associate utility functions with the numbering of alternatives
    V = {1: V_temp[0],
          2: V_temp[1],
          3: V_temp[2],
          4: V_temp[3],
          5: V_temp[4]}
    
    
    # Associate the availability conditions with the alternatives (change all of these values to 1?)
    av = {1: 1,
          2: 1,
          3: 1,
          4: 1,
          5: 1}
    
    
    # Definition of the model. This is the contribution of each
    # observation to the log likelihood function.
    # logprob = models.loglogit(V, av, modal_choice)
    logprob = models.loglogit(V, av, CHOICE)
    
    # Create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = 'logit_extended_pers' + str(pers_cl) + '_trip' + str(trip_cl) + '_' + dt
    
    # Estimate the parameters
    results = biogeme.estimate()
    
    print('Finished estimating model for trip cluster ' + str(trip_cl) + ' and pers cluster ' + str(pers_cl))
    
    return dt



def estimate_choice_model_nested(modes, dt, k=0, trip_cl=99, pers_cl=99, act_asc=0, clusters=0):
    
    if k == 0:
        df = pd.read_csv('data/data_cl_train.csv')
    else:
        df = pd.read_csv('data/data_cl' + str(k) + '_train.csv')
    
    # df = pd.read_csv('data/data_original.csv')
    # df = pd.read_csv('data/TrainData.csv')
    
    if clusters:
        df_temp = df[df["trip_cl"] == trip_cl]
        df_selection = df_temp[df_temp["pers_cl"] == pers_cl]
    else:
        df_selection = df
    # df_selection = df
    
    database = db.Database('traindataAI', df_selection)
    globals().update(database.variables)
    
    
    ############### ADJUST BELOW THIS LINE ###############

    B = [0] * 110
    for i in range(110):
        B[i] = Beta('B_{}'.format(i), ((random.randint(0, 100) - 50 )/ 100), None, None, 0)
    
    MU_1 = Beta('MU_1', 1, 1, 10, 0)
    MU_2 = Beta('MU_2', 1, 1, 10, 0)
    
    # Definition of the utility functions
    V_temp = [0] * 5
    
    for i in range(5):      
          V_temp[i] = B[0] * globals()["cost_" + modes[i]] + B[1] * globals()["time_" + modes[i]] + \
              B[2] * globals()["driving_license"] + B[3] * globals()["drivingtask_" + modes[i]] + \
              B[4] * globals()["skills_" + modes[i]] + B[5] * globals()["weatherProtection_" + modes[i]] + \
              B[6] * globals()["luggage_" + modes[i]] + B[7] * globals()["shared_" + modes[i]] + \
              B[8] * globals()["availability_" + modes[i]] + B[9] * globals()["reservation_" + modes[i]] + \
              B[10] * globals()["active_" + modes[i]] + B[11] * globals()["accessible_" + modes[i]]
                
    ############### ADJUST ABOVE THIS LINE ###############
    
    # # Associate utility functions with the numbering of alternatives
    V = {1: V_temp[0],
          2: V_temp[1],
          3: V_temp[2],
          4: V_temp[3],
          5: V_temp[4]}
    
    
    # Associate the availability conditions with the alternatives (change all of these values to 1?)
    av = {1: 1,
          2: 1,
          3: 1,
          4: 1,
          5: 1}
    
    # Definition of nests
    
    # Cut-off similarity at 0.80 (1)
    # car = 1.0, [1]
    # carp = 1.0, [2]
    # transit = 1.0, [3]
    # active = MU_1, [4, 5]
    # nests = car, carp, transit, active 
    
    
    # Cut-off similarity at 0.70 (2)
    car = 1.0, [1]
    shared = MU_1, [2, 3]
    active = MU_2, [4, 5]
    nests = car, shared, active
    
    
    # Definition of the model. This is the contribution of each
    # observation to the log likelihood function.
    # The choice model is a nested logit, with availability conditions
    logprob = models.lognested(V, av, nests, CHOICE)

    
    # Create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = 'logit_extended_pers' + str(pers_cl) + '_trip' + str(trip_cl) + '_' + dt
    
    # Estimate the parameters
    results = biogeme.estimate()
    
    print('Finished estimating model for trip cluster ' + str(trip_cl) + ' and pers cluster ' + str(pers_cl))
    
    return dt



    
def estimate_modal_split(modes, k, trip_cl, pers_cl, frac_cl, act_asc, dt, subset, future_mode_char):
        
    # Simulate modal split with testset
    
    # Load testdata
    df = pd.read_csv('data/data_cl_test.csv')
    # df_temp = df[df["trip_cl"] == trip_cl]
    df_temp = df
    df_selection = df_temp[df_temp["pers_cl"] == pers_cl]
    # df_selection = df_selection[df_selection["frac"] == frac_cl]
    
    database = db.Database('traindataAI', df_selection)
    globals().update(database.variables)
    
    # Load results estimation
    results = res.bioResults(pickleFile = 'logit_extended_pers' + str(pers_cl) + '_trip' + str(trip_cl) + '_' + dt + '.pickle')
    pandasResults = results.getEstimatedParameters()
    gs = results.getGeneralStatistics()
    
    sc_f = 1 # / -np.min(pandasResults['Value']['B_COST'], pandasResults['Value']['B_TIME'])
    
    ############### ADJUST BELOW THIS LINE ###############
    
    modes = ['car', 'carp', 'transit', 'cycle', 'walk', 'future']
    
    P = np.empty((0, 6))
    P_future = np.empty((0, 7))
    modal_split_actual = [0, 0, 0, 0, 0]
    modal_split_future = [0, 0, 0, 0, 0, 0]
    CHOICE_M = np.empty((0, 2))
    
    
    print('Number of combinations future mode tested', len(future_mode_char))
    
    # Create empty matrix to store results
    w, h, d = 6, len(future_mode_char), len(df_selection)
    P_temp = [[[0 for x in range(w)] for y in range(h)] for z in range(d)]
    
    for i in range(len(df_selection)):
        
        
        
        V = [0, 0, 0, 0, 0, 0]
        distance_trip = [0, 0, 0, 0, 0]
        distance_trip_counter = 0
        assumed_average_speed = [60, 60, 40, 15, 5] # [car, carp, transit, cycle, walk]
        
        # try:
        for j in range(5):
            
            if df_selection.iloc[i]["CHOICE"] == j + 1:
                modal_split_actual[j] += 1
            
            
            # V[j] = sc_f * (float(pandasResults['Value']['B_COST']) * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["time_" + modes[j]])
            
            V[j] = sc_f * (float(pandasResults['Value']['B_0']) * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_1']) * df_selection.iloc[i]["time_" + modes[j]] + \
              float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * df_selection.iloc[i]["drivingtask_" + modes[j]] + \
              float(pandasResults['Value']['B_4']) * df_selection.iloc[i]["skills_" + modes[j]] + float(pandasResults['Value']['B_5']) * df_selection.iloc[i]["weatherProtection_" + modes[j]] + \
              float(pandasResults['Value']['B_6']) * df_selection.iloc[i]["luggage_" + modes[j]] + float(pandasResults['Value']['B_7']) * df_selection.iloc[i]["shared_" + modes[j]] + \
              float(pandasResults['Value']['B_8']) * df_selection.iloc[i]["availability_" + modes[j]] + float(pandasResults['Value']['B_9']) * df_selection.iloc[i]["reservation_" + modes[j]] + \
              float(pandasResults['Value']['B_10']) * df_selection.iloc[i]["active_" + modes[j]] + float(pandasResults['Value']['B_11']) * df_selection.iloc[i]["accessible_" + modes[j]])

            # Utility synthetic data
            # V[j] = float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["age"] * df_selection.iloc[i]["time_" + modes[j]] * sc_f + float(pandasResults['Value']['B_COST']) * 200000 / df_selection.iloc[i]["income"] * df_selection.iloc[i]["cost_" + modes[j]] * sc_f # Exactly the same as in data generation
        
        
            # Estimate distance in dataset based on travel times of all existing modes
            # average of (time for available mode) / (assumed average speed for available mode)
            if (df_selection.iloc[i]["time_" + modes[j]] != 9999):
                distance_trip[j] = df_selection.iloc[i]["time_" + modes[j]] / assumed_average_speed[j]
                distance_trip_counter += 1
            else:
                distance_trip[j] = 0
                
        E = [0, 0, 0, 0, 0]
        for j in range(5):
            E[j] = np.exp(V[j]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V[3]) + np.exp(V[4]))
        
        choice = E.index(max(E)) + 1
        P = np.append(P, np.array([[E[0], E[1], E[2], E[3], E[4], choice]]), axis=0) # nrOfModes in modal split
            
        
        # Add utility function of new mobility system here.
        # V[5] = -100 # sc_f * (float(pandasResults['Value']['B_COST']) * df_selection.iloc[i]["c_" + modes[j]] + float(pandasResults['Value']['B_DUR1']) * df_selection.iloc[i]["t_" + modes[j]])
 
        # Estimate distance in dataset based on travel times of all existing modes
        # average of (time for available mode) / (assumed average speed for available mode)
        distance_estimated = sum(distance_trip) / distance_trip_counter
        
        
        for j in range(len(future_mode_char)):
            try:
                # Variable costs (per km)
                V[5] = sc_f * (float(pandasResults['Value']['B_0']) * future_mode_char[j, 0] * distance_estimated + float(pandasResults['Value']['B_1']) * (distance_estimated / future_mode_char[j, 1]) / 60 + \
                      float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * future_mode_char[j, 2] + \
                      float(pandasResults['Value']['B_4']) * future_mode_char[j, 3] + float(pandasResults['Value']['B_5']) * future_mode_char[j, 4] + \
                      float(pandasResults['Value']['B_6']) * future_mode_char[j, 5] + float(pandasResults['Value']['B_7']) * future_mode_char[j, 6] + \
                      float(pandasResults['Value']['B_8']) * future_mode_char[j, 7] + float(pandasResults['Value']['B_9']) * future_mode_char[j, 8] + \
                      float(pandasResults['Value']['B_10']) * future_mode_char[j, 9] + float(pandasResults['Value']['B_11']) * future_mode_char[j, 10])
                
                # Fixed costs
                # V[5] = sc_f * (float(pandasResults['Value']['B_0']) * future_mode_char[j, 0] + float(pandasResults['Value']['B_1']) * (distance_estimated / future_mode_char[j, 1]) / 60 + \
                #       float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * future_mode_char[j, 2] + \
                #       float(pandasResults['Value']['B_4']) * future_mode_char[j, 3] + float(pandasResults['Value']['B_5']) * future_mode_char[j, 4] + \
                #       float(pandasResults['Value']['B_6']) * future_mode_char[j, 5] + float(pandasResults['Value']['B_7']) * future_mode_char[j, 6] + \
                #       float(pandasResults['Value']['B_8']) * future_mode_char[j, 7] + float(pandasResults['Value']['B_9']) * future_mode_char[j, 8] + \
                #       float(pandasResults['Value']['B_10']) * future_mode_char[j, 9] + float(pandasResults['Value']['B_11']) * future_mode_char[j, 10])
            except:
                V[5] = -100
                print('Could not calculate V[5], retrieving pandasResults failed for', trip_cl, pers_cl)
            # if(j % 100==0):
            #     print('Progress estimate future modal split', j/len(future_mode_char))
        # Utility synthetic data
        # V[5] = sc_f * (float(pandasResults['Value']['B_COST']) * 200000 / df_selection.iloc[i]["income"] * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["age"] * df_selection.iloc[i]["time_" + modes[j]])

            E_future = [0, 0, 0, 0, 0, 0]
            for k in range(6):
                E_future[k] = np.exp(V[k]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V[3]) + np.exp(V[4]) + np.exp(V[5]))
                
                try:
                    P_temp[i][j][k] = E_future[k]
                except:
                    print(E_future)
            
        choice_future = E_future.index(max(E_future)) + 1
        
        # Write change in choice in dataset
        CHOICE_M = np.append(CHOICE_M, np.array([[df_selection.iloc[i]["CHOICE"], choice_future]]), axis=0)
            
        P_future = np.append(P_future, np.array([[E_future[0], E_future[1], E_future[2], E_future[3], E_future[4], E_future[5], choice_future]]), axis=0)
        print('Progress estimate modal split', (i/len(df_selection)))
        
    modal_split = np.sum(P[:,0:5], axis=0) / np.sum(np.sum(P[:,0:5], axis=0)) # nrOfModes in modal split
    
    # Save change modal choice
    CHOICE_M_TEMP = pd.DataFrame(CHOICE_M, columns=["Actual", "Future"])
    CHOICE_M_TEMP.to_csv("data/CHOICE" + str(trip_cl) + str(pers_cl) + ".csv")
    
    res_future = np.empty((0, 10))
    
    for j in range(len(future_mode_char)):
        res_temp = [[0 for x in range(6)] for y in range(len(df_selection))]
        for i in range(len(df_selection)):
            res_temp[i][:] = P_temp[i][j][:]
        
        
        modal_split_future = np.sum(res_temp[:][:], axis=0) / np.sum(np.sum(res_temp[:][:], axis=0))
        res_future = np.append(res_future, np.array([[j, trip_cl, pers_cl, len(df_selection), modal_split_future[0], modal_split_future[1], modal_split_future[2], modal_split_future[3], modal_split_future[4], modal_split_future[5]]]), axis=0)
        

    
    modal_split_future = np.sum(P_future[:,0:6], axis=0) / np.sum(np.sum(P_future[:,0:6], axis=0))
    
    R_square = r2_score(P[:, 5], df_selection["CHOICE"])
    print('Coefficient of Determination', R_square) 
    
    report = metrics.classification_report(df_selection['CHOICE'], P[:,5], digits=3, output_dict=True) # nrOfModes in modal split
    results = pd.DataFrame(report).transpose()
    results['trip'] = trip_cl
    results['pers'] = pers_cl
    
    results['modal_split_estimated'] = 0
    results['modal_split_actual'] = 0
    results['modal_split_future'] = 0
    for i in np.arange(1, 6): # nrOfModes in modal split
        results.loc['{0}'.format(i), 'modal_split_estimated'] = modal_split[i-1]
        results.loc['{0}'.format(i), 'modal_split_actual'] = (modal_split_actual[i-1] / np.sum(modal_split_actual))
        results.loc['{0}'.format(i), 'modal_split_future'] = modal_split_future[i-1] # Note: mode 6 is not included here
    
    results.dropna(subset = ["precision"], inplace=True)
    
    ############### ADJUST ABOVE THIS LINE ###############
    
    pandasResults['trip'] = trip_cl
    pandasResults['pers'] = pers_cl

    print('Finished estimating modal split for trip cluster ' + str(trip_cl) + ' and pers cluster ' + str(pers_cl))
    
    return results, pandasResults, gs, modal_split_future[5], R_square, res_future



def estimate_modal_split_nested(modes, k, trip_cl, pers_cl, frac_cl, act_asc, dt, subset, future_mode_char, nest, max_sim):
        
    # Simulate modal split with testset
    
    # Load testdata
    df = pd.read_csv('data/data_cl_test.csv')
    # df_temp = df[df["trip_cl"] == trip_cl]
    df_temp = df
    df_selection = df_temp[df_temp["pers_cl"] == pers_cl]
    # df_selection = df_selection[df_selection["frac"] == frac_cl]
    
    database = db.Database('traindataAI', df_selection)
    globals().update(database.variables)
    
    # Load results estimation
    results = res.bioResults(pickleFile = 'logit_extended_pers' + str(pers_cl) + '_trip' + str(trip_cl) + '_' + dt + '.pickle')
    pandasResults = results.getEstimatedParameters()
    gs = results.getGeneralStatistics()
    
    sc_f = 1 # / -np.min(pandasResults['Value']['B_COST'], pandasResults['Value']['B_TIME'])
    
    ############### ADJUST BELOW THIS LINE ###############
    
    modes = ['car', 'carp', 'transit', 'cycle', 'walk', 'future']
    
    P = np.empty((0, 6))
    P_future = np.empty((0, 7))
    modal_split_actual = [0, 0, 0, 0, 0]
    modal_split_future = [0, 0, 0, 0, 0, 0]
    CHOICE_M = np.empty((0, 2))
    
    
    print('Number of combinations future mode tested', len(future_mode_char))
    
    # Create empty matrix to store results
    w, h, d = 6, len(future_mode_char), len(df_selection)
    P_temp = [[[0 for x in range(w)] for y in range(h)] for z in range(d)]
    
    for i in range(len(df_selection)):
        
        V = [0, 0, 0, 0, 0, 0]
        distance_trip = [0, 0, 0, 0, 0]
        distance_trip_counter = 0
        assumed_average_speed = [60, 60, 40, 15, 5] # [car, carp, transit, cycle, walk]
        
        # try:
        for j in range(5):
            
            if df_selection.iloc[i]["CHOICE"] == j + 1:
                modal_split_actual[j] += 1
            
            
            # V[j] = sc_f * (float(pandasResults['Value']['B_COST']) * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["time_" + modes[j]])
            
            V[j] = sc_f * (float(pandasResults['Value']['B_0']) * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_1']) * df_selection.iloc[i]["time_" + modes[j]] + \
              float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * df_selection.iloc[i]["drivingtask_" + modes[j]] + \
              float(pandasResults['Value']['B_4']) * df_selection.iloc[i]["skills_" + modes[j]] + float(pandasResults['Value']['B_5']) * df_selection.iloc[i]["weatherProtection_" + modes[j]] + \
              float(pandasResults['Value']['B_6']) * df_selection.iloc[i]["luggage_" + modes[j]] + float(pandasResults['Value']['B_7']) * df_selection.iloc[i]["shared_" + modes[j]] + \
              float(pandasResults['Value']['B_8']) * df_selection.iloc[i]["availability_" + modes[j]] + float(pandasResults['Value']['B_9']) * df_selection.iloc[i]["reservation_" + modes[j]] + \
              float(pandasResults['Value']['B_10']) * df_selection.iloc[i]["active_" + modes[j]] + float(pandasResults['Value']['B_11']) * df_selection.iloc[i]["accessible_" + modes[j]])

            # Utility synthetic data
            # V[j] = float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["age"] * df_selection.iloc[i]["time_" + modes[j]] * sc_f + float(pandasResults['Value']['B_COST']) * 200000 / df_selection.iloc[i]["income"] * df_selection.iloc[i]["cost_" + modes[j]] * sc_f # Exactly the same as in data generation
        
        
            # Estimate distance in dataset based on travel times of all existing modes
            # average of (time for available mode) / (assumed average speed for available mode)
            if (df_selection.iloc[i]["time_" + modes[j]] != 9999):
                distance_trip[j] = df_selection.iloc[i]["time_" + modes[j]] / assumed_average_speed[j]
                distance_trip_counter += 1
            else:
                distance_trip[j] = 0
                
        E = [0, 0, 0, 0, 0]
        for j in range(5):
            E[j] = np.exp(V[j]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V[3]) + np.exp(V[4]))
        
        choice = E.index(max(E)) + 1
        P = np.append(P, np.array([[E[0], E[1], E[2], E[3], E[4], choice]]), axis=0) # nrOfModes in modal split
            
        
        # Add utility function of new mobility system here.
        # V[5] = -100 # sc_f * (float(pandasResults['Value']['B_COST']) * df_selection.iloc[i]["c_" + modes[j]] + float(pandasResults['Value']['B_DUR1']) * df_selection.iloc[i]["t_" + modes[j]])
 
        # Estimate distance in dataset based on travel times of all existing modes
        # average of (time for available mode) / (assumed average speed for available mode)
        distance_estimated = sum(distance_trip) / distance_trip_counter
        
        
        for j in range(len(future_mode_char)):
            try:
                # Variable costs (per km)
                V[5] = sc_f * (float(pandasResults['Value']['B_0']) * future_mode_char[j, 0] * distance_estimated + float(pandasResults['Value']['B_1']) * (distance_estimated / future_mode_char[j, 1]) / 60 + \
                      float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * future_mode_char[j, 2] + \
                      float(pandasResults['Value']['B_4']) * future_mode_char[j, 3] + float(pandasResults['Value']['B_5']) * future_mode_char[j, 4] + \
                      float(pandasResults['Value']['B_6']) * future_mode_char[j, 5] + float(pandasResults['Value']['B_7']) * future_mode_char[j, 6] + \
                      float(pandasResults['Value']['B_8']) * future_mode_char[j, 7] + float(pandasResults['Value']['B_9']) * future_mode_char[j, 8] + \
                      float(pandasResults['Value']['B_10']) * future_mode_char[j, 9] + float(pandasResults['Value']['B_11']) * future_mode_char[j, 10])
                
                # Fixed costs
                # V[5] = sc_f * (float(pandasResults['Value']['B_0']) * future_mode_char[j, 0] + float(pandasResults['Value']['B_1']) * (distance_estimated / future_mode_char[j, 1]) / 60 + \
                #       float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * future_mode_char[j, 2] + \
                #       float(pandasResults['Value']['B_4']) * future_mode_char[j, 3] + float(pandasResults['Value']['B_5']) * future_mode_char[j, 4] + \
                #       float(pandasResults['Value']['B_6']) * future_mode_char[j, 5] + float(pandasResults['Value']['B_7']) * future_mode_char[j, 6] + \
                #       float(pandasResults['Value']['B_8']) * future_mode_char[j, 7] + float(pandasResults['Value']['B_9']) * future_mode_char[j, 8] + \
                #       float(pandasResults['Value']['B_10']) * future_mode_char[j, 9] + float(pandasResults['Value']['B_11']) * future_mode_char[j, 10])
            except:
                V[5] = -100
                print('Could not calculate V[5], retrieving pandasResults failed for', trip_cl, pers_cl)
            # if(j % 100==0):
            #     print('Progress estimate future modal split', j/len(future_mode_char))
        # Utility synthetic data
        # V[5] = sc_f * (float(pandasResults['Value']['B_COST']) * 200000 / df_selection.iloc[i]["income"] * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["age"] * df_selection.iloc[i]["time_" + modes[j]])

            E_future = [0, 0, 0, 0, 0, 0]
            
            # Which existing mode will be nested with the future mode?
            # nest = 1
            mu_b = sc_f / max_sim
            
            for k in range(6):
                
                
                if nest == 5:
                    E_future[k] = np.exp(V[k]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V[3]) + np.exp(V[4]) + np.exp(V[5]))
                else:
                    
                    V_nest = (sc_f / mu_b) * np.log(np.exp(mu_b * V[nest]) + np.exp(mu_b * V[5]))
                    idx_mode = [0, 1, 2, 3, 4]
                    idx_mode = np.delete(idx_mode, nest)
                    if (k != nest and k != 5):
                        E_future[k] = np.exp(V[k]) / (np.exp(np.take(V, idx_mode[0])) + np.exp(np.take(V, idx_mode[1])) + np.exp(np.take(V, idx_mode[2])) + np.exp(np.take(V, idx_mode[3])) + np.exp(V_nest))
                    else:
                        E_future[k] = np.exp(mu_b * V[k]) / (np.exp(np.take(V, idx_mode[0])) + np.exp(np.take(V, idx_mode[1])) + np.exp(np.take(V, idx_mode[2])) + np.exp(np.take(V, idx_mode[3])) + np.exp(V_nest))
                
                try:
                    P_temp[i][j][k] = E_future[k]
                except:
                    print(E_future)
            
        choice_future = E_future.index(max(E_future)) + 1
        
        # Write change in choice in dataset
        CHOICE_M = np.append(CHOICE_M, np.array([[df_selection.iloc[i]["CHOICE"], choice_future]]), axis=0)
        
        P_future = np.append(P_future, np.array([[E_future[0], E_future[1], E_future[2], E_future[3], E_future[4], E_future[5], choice_future]]), axis=0)
        print('Progress estimate modal split', (i/len(df_selection)))
        
    modal_split = np.sum(P[:,0:5], axis=0) / np.sum(np.sum(P[:,0:5], axis=0)) # nrOfModes in modal split
    
    # Save change modal choice
    CHOICE_M_TEMP = pd.DataFrame(CHOICE_M, columns=["Actual", "Future"])
    CHOICE_M_TEMP.to_csv("data/CHOICE" + str(trip_cl) + str(pers_cl) + ".csv")
    
    res_future = np.empty((0, 10))
    
    for j in range(len(future_mode_char)):
        res_temp = [[0 for x in range(6)] for y in range(len(df_selection))]
        for i in range(len(df_selection)):
            res_temp[i][:] = P_temp[i][j][:]
        
        
        modal_split_future = np.sum(res_temp[:][:], axis=0) / np.sum(np.sum(res_temp[:][:], axis=0))
        res_future = np.append(res_future, np.array([[j, trip_cl, pers_cl, len(df_selection), modal_split_future[0], modal_split_future[1], modal_split_future[2], modal_split_future[3], modal_split_future[4], modal_split_future[5]]]), axis=0)
        

    
    modal_split_future = np.sum(P_future[:,0:6], axis=0) / np.sum(np.sum(P_future[:,0:6], axis=0))
    
    R_square = r2_score(P[:, 5], df_selection["CHOICE"])
    print('Coefficient of Determination', R_square) 
    
    report = metrics.classification_report(df_selection['CHOICE'], P[:,5], digits=3, output_dict=True) # nrOfModes in modal split
    results = pd.DataFrame(report).transpose()
    results['trip'] = trip_cl
    results['pers'] = pers_cl
    
    results['modal_split_estimated'] = 0
    results['modal_split_actual'] = 0
    results['modal_split_future'] = 0
    for i in np.arange(1, 6): # nrOfModes in modal split
        results.loc['{0}'.format(i), 'modal_split_estimated'] = modal_split[i-1]
        results.loc['{0}'.format(i), 'modal_split_actual'] = (modal_split_actual[i-1] / np.sum(modal_split_actual))
        results.loc['{0}'.format(i), 'modal_split_future'] = modal_split_future[i-1] # Note: mode 6 is not included here
    
    results.dropna(subset = ["precision"], inplace=True)
    
    ############### ADJUST ABOVE THIS LINE ###############
    
    pandasResults['trip'] = trip_cl
    pandasResults['pers'] = pers_cl

    print('Finished estimating modal split for trip cluster ' + str(trip_cl) + ' and pers cluster ' + str(pers_cl))
    
    return results, pandasResults, gs, modal_split_future[5], R_square, res_future


def estimate_modal_split_nested_1(modes, k, trip_cl, pers_cl, frac_cl, act_asc, dt, subset, future_mode_char, nest, max_sim, mu_b):
        
    # Simulate modal split with testset for threshold 0.8
    
    # Load testdata
    df = pd.read_csv('data/data_cl_test.csv')
    # df_temp = df[df["trip_cl"] == trip_cl]
    df_temp = df
    df_selection = df_temp[df_temp["pers_cl"] == pers_cl]
    # df_selection = df_selection[df_selection["frac"] == frac_cl]
    
    database = db.Database('traindataAI', df_selection)
    globals().update(database.variables)
    
    # Load results estimation
    results = res.bioResults(pickleFile = 'logit_extended_pers' + str(pers_cl) + '_trip' + str(trip_cl) + '_' + dt + '.pickle')
    pandasResults = results.getEstimatedParameters()
    gs = results.getGeneralStatistics()
    
    sc_f = 1 # / -np.min(pandasResults['Value']['B_COST'], pandasResults['Value']['B_TIME'])
    
    ############### ADJUST BELOW THIS LINE ###############
    
    modes = ['car', 'carp', 'transit', 'cycle', 'walk', 'future']
    
    P = np.empty((0, 6))
    P_future = np.empty((0, 7))
    modal_split_actual = [0, 0, 0, 0, 0]
    modal_split_future = [0, 0, 0, 0, 0, 0]
    
    
    print('Number of combinations future mode tested', len(future_mode_char))
    
    # Create empty matrix to store results
    w, h, d = 6, len(future_mode_char), len(df_selection)
    P_temp = [[[0 for x in range(w)] for y in range(h)] for z in range(d)]
    
    for i in range(len(df_selection)):
        
        V = [0, 0, 0, 0, 0, 0]
        distance_trip = [0, 0, 0, 0, 0]
        distance_trip_counter = 0
        assumed_average_speed = [60, 60, 40, 15, 5] # [car, carp, transit, cycle, walk]
        
        # try:
        for j in range(5):
            
            if df_selection.iloc[i]["CHOICE"] == j + 1:
                modal_split_actual[j] += 1
            
            
            # V[j] = sc_f * (float(pandasResults['Value']['B_COST']) * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["time_" + modes[j]])
            
            V[j] = sc_f * (float(pandasResults['Value']['B_0']) * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_1']) * df_selection.iloc[i]["time_" + modes[j]] + \
              float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * df_selection.iloc[i]["drivingtask_" + modes[j]] + \
              float(pandasResults['Value']['B_4']) * df_selection.iloc[i]["skills_" + modes[j]] + float(pandasResults['Value']['B_5']) * df_selection.iloc[i]["weatherProtection_" + modes[j]] + \
              float(pandasResults['Value']['B_6']) * df_selection.iloc[i]["luggage_" + modes[j]] + float(pandasResults['Value']['B_7']) * df_selection.iloc[i]["shared_" + modes[j]] + \
              float(pandasResults['Value']['B_8']) * df_selection.iloc[i]["availability_" + modes[j]] + float(pandasResults['Value']['B_9']) * df_selection.iloc[i]["reservation_" + modes[j]] + \
              float(pandasResults['Value']['B_10']) * df_selection.iloc[i]["active_" + modes[j]] + float(pandasResults['Value']['B_11']) * df_selection.iloc[i]["accessible_" + modes[j]])

            # Utility synthetic data
            # V[j] = float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["age"] * df_selection.iloc[i]["time_" + modes[j]] * sc_f + float(pandasResults['Value']['B_COST']) * 200000 / df_selection.iloc[i]["income"] * df_selection.iloc[i]["cost_" + modes[j]] * sc_f # Exactly the same as in data generation
        
        
            # Estimate distance in dataset based on travel times of all existing modes
            # average of (time for available mode) / (assumed average speed for available mode)
            if (df_selection.iloc[i]["time_" + modes[j]] != 9999):
                distance_trip[j] = df_selection.iloc[i]["time_" + modes[j]] / assumed_average_speed[j]
                distance_trip_counter += 1
            else:
                distance_trip[j] = 0
                
        E = [0, 0, 0, 0, 0]
        V_nest = (sc_f / float(pandasResults['Value']['MU_1'])) * np.log(np.exp(float(pandasResults['Value']['MU_1']) * V[3]) + np.exp(float(pandasResults['Value']['MU_1']) * V[4]))
        for j in range(5):
            if (j < 3):
                E[j] = np.exp(V[j]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V_nest))
            else:
                E[j] = np.exp(float(pandasResults['Value']['MU_1']) * V[j]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V_nest))
                
        
        choice = E.index(max(E)) + 1
        P = np.append(P, np.array([[E[0], E[1], E[2], E[3], E[4], choice]]), axis=0) # nrOfModes in modal split
            
        
        # Add utility function of new mobility system here.
        # V[5] = -100 # sc_f * (float(pandasResults['Value']['B_COST']) * df_selection.iloc[i]["c_" + modes[j]] + float(pandasResults['Value']['B_DUR1']) * df_selection.iloc[i]["t_" + modes[j]])
 
        # Estimate distance in dataset based on travel times of all existing modes
        # average of (time for available mode) / (assumed average speed for available mode)
        distance_estimated = sum(distance_trip) / distance_trip_counter
        
        
        for j in range(len(future_mode_char)):
            try:
                # Variable costs (per km)
                V[5] = sc_f * (float(pandasResults['Value']['B_0']) * future_mode_char[j, 0] * distance_estimated + float(pandasResults['Value']['B_1']) * (distance_estimated / future_mode_char[j, 1]) / 60 + \
                      float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * future_mode_char[j, 2] + \
                      float(pandasResults['Value']['B_4']) * future_mode_char[j, 3] + float(pandasResults['Value']['B_5']) * future_mode_char[j, 4] + \
                      float(pandasResults['Value']['B_6']) * future_mode_char[j, 5] + float(pandasResults['Value']['B_7']) * future_mode_char[j, 6] + \
                      float(pandasResults['Value']['B_8']) * future_mode_char[j, 7] + float(pandasResults['Value']['B_9']) * future_mode_char[j, 8] + \
                      float(pandasResults['Value']['B_10']) * future_mode_char[j, 9] + float(pandasResults['Value']['B_11']) * future_mode_char[j, 10])
                
                # Fixed costs
                # V[5] = sc_f * (float(pandasResults['Value']['B_0']) * future_mode_char[j, 0] + float(pandasResults['Value']['B_1']) * (distance_estimated / future_mode_char[j, 1]) / 60 + \
                #       float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * future_mode_char[j, 2] + \
                #       float(pandasResults['Value']['B_4']) * future_mode_char[j, 3] + float(pandasResults['Value']['B_5']) * future_mode_char[j, 4] + \
                #       float(pandasResults['Value']['B_6']) * future_mode_char[j, 5] + float(pandasResults['Value']['B_7']) * future_mode_char[j, 6] + \
                #       float(pandasResults['Value']['B_8']) * future_mode_char[j, 7] + float(pandasResults['Value']['B_9']) * future_mode_char[j, 8] + \
                #       float(pandasResults['Value']['B_10']) * future_mode_char[j, 9] + float(pandasResults['Value']['B_11']) * future_mode_char[j, 10])
            except:
                V[5] = -100
                print('Could not calculate V[5], retrieving pandasResults failed for', trip_cl, pers_cl)
            # if(j % 100==0):
            #     print('Progress estimate future modal split', j/len(future_mode_char))
        # Utility synthetic data
        # V[5] = sc_f * (float(pandasResults['Value']['B_COST']) * 200000 / df_selection.iloc[i]["income"] * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["age"] * df_selection.iloc[i]["time_" + modes[j]])

            E_future = [0, 0, 0, 0, 0, 0]
            
            # Which existing mode will be nested with the future mode?
            # nest = 3
            # mu_b = sc_f / max_sim
            
            for k in range(6):
                    
                if nest < 3:
                    V_nest = (sc_f / float(pandasResults['Value']['MU_1'])) * np.log(np.exp(float(pandasResults['Value']['MU_1']) * V[3]) + np.exp(float(pandasResults['Value']['MU_1']) * V[4]))    
                    V_nest_2 = (sc_f / mu_b) * np.log(np.exp(mu_b * V[nest]) + np.exp(mu_b * V[5]))
                    idx_mode = [0, 1, 2]
                    idx_mode = np.delete(idx_mode, nest)
                    
                    if (k < 3) or (k == 5):
                        if (k != nest and k != 5):
                            E_future[k] = np.exp(V[k]) / (np.exp(np.take(V, idx_mode[0])) + np.exp(np.take(V, idx_mode[1])) + np.exp(np.take(V, idx_mode[2])) + np.exp(V_nest) + np.exp(V_nest_2))
                        else:
                            E_future[k] = np.exp(mu_b * V[k]) / (np.exp(np.take(V, idx_mode[0])) + np.exp(np.take(V, idx_mode[1])) + np.exp(np.take(V, idx_mode[2])) + np.exp(V_nest) + np.exp(V_nest_2))
                    else:
                        E_future[k] = np.exp(float(pandasResults['Value']['MU_1']) * V[k]) / (np.exp(np.take(V, idx_mode[0])) + np.exp(np.take(V, idx_mode[1])) + np.exp(np.take(V, idx_mode[2])) + np.exp(V_nest) + np.exp(V_nest_2))
                
                elif nest == 3 or nest == 4:
                    V_nest = (sc_f / mu_b) * np.log(np.exp(mu_b * V[3]) + np.exp(mu_b * V[4]) + np.exp(mu_b * V[5]))
                    
                    if (k < 3):
                        E_future[k] = np.exp(V[k]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V_nest))
                    else:
                        E_future[k] = np.exp(mu_b * V[k]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V_nest))
                
                else:
                    V_nest = (sc_f / float(pandasResults['Value']['MU_1'])) * np.log(np.exp(float(pandasResults['Value']['MU_1']) * V[3]) + np.exp(float(pandasResults['Value']['MU_1']) * V[4]))    
                
                    if (k < 3) or (k == 5):
                        E_future[k] = np.exp(V[k]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V_nest) + np.exp(V[5]))
                    else:
                        E_future[k] = np.exp(float(pandasResults['Value']['MU_1']) * V[k]) / (np.exp(V[0]) + np.exp(V[1]) + np.exp(V[2]) + np.exp(V_nest) + np.exp(V[5]))
              
                try:
                    P_temp[i][j][k] = E_future[k]
                except:
                    print(E_future)
            
        choice_future = E_future.index(max(E_future)) + 1
            
        P_future = np.append(P_future, np.array([[E_future[0], E_future[1], E_future[2], E_future[3], E_future[4], E_future[5], choice_future]]), axis=0)
        print('Progress estimate modal split', (i/len(df_selection)))
        
    modal_split = np.sum(P[:,0:5], axis=0) / np.sum(np.sum(P[:,0:5], axis=0)) # nrOfModes in modal split
    
    
    res_future = np.empty((0, 10))
    
    for j in range(len(future_mode_char)):
        res_temp = [[0 for x in range(6)] for y in range(len(df_selection))]
        for i in range(len(df_selection)):
            res_temp[i][:] = P_temp[i][j][:]
        
        
        modal_split_future = np.sum(res_temp[:][:], axis=0) / np.sum(np.sum(res_temp[:][:], axis=0))
        res_future = np.append(res_future, np.array([[j, trip_cl, pers_cl, len(df_selection), modal_split_future[0], modal_split_future[1], modal_split_future[2], modal_split_future[3], modal_split_future[4], modal_split_future[5]]]), axis=0)
        

    
    modal_split_future = np.sum(P_future[:,0:6], axis=0) / np.sum(np.sum(P_future[:,0:6], axis=0))
    
    R_square = r2_score(P[:, 5], df_selection["CHOICE"])
    print('Coefficient of Determination', R_square) 
    
    report = metrics.classification_report(df_selection['CHOICE'], P[:,5], digits=3, output_dict=True) # nrOfModes in modal split
    results = pd.DataFrame(report).transpose()
    results['trip'] = trip_cl
    results['pers'] = pers_cl
    
    results['modal_split_estimated'] = 0
    results['modal_split_actual'] = 0
    results['modal_split_future'] = 0
    for i in np.arange(1, 6): # nrOfModes in modal split
        results.loc['{0}'.format(i), 'modal_split_estimated'] = modal_split[i-1]
        results.loc['{0}'.format(i), 'modal_split_actual'] = (modal_split_actual[i-1] / np.sum(modal_split_actual))
        results.loc['{0}'.format(i), 'modal_split_future'] = modal_split_future[i-1] # Note: mode 6 is not included here
    
    results.dropna(subset = ["precision"], inplace=True)
    
    ############### ADJUST ABOVE THIS LINE ###############
    
    pandasResults['trip'] = trip_cl
    pandasResults['pers'] = pers_cl

    print('Finished estimating modal split for trip cluster ' + str(trip_cl) + ' and pers cluster ' + str(pers_cl))
    
    return results, pandasResults, gs, modal_split_future[5], R_square, res_future


def estimate_modal_split_nested_2(modes, k, trip_cl, pers_cl, frac_cl, act_asc, dt, subset, future_mode_char, nest, max_sim, mu_b):
        
    # Simulate modal split with testset for threshold 0.8
    
    # Load testdata
    df = pd.read_csv('data/data_cl_test.csv')
    # df_temp = df[df["trip_cl"] == trip_cl]
    df_temp = df
    df_selection = df_temp[df_temp["pers_cl"] == pers_cl]
    # df_selection = df_selection[df_selection["frac"] == frac_cl]
    
    database = db.Database('traindataAI', df_selection)
    globals().update(database.variables)
    
    # Load results estimation
    results = res.bioResults(pickleFile = 'logit_extended_pers' + str(pers_cl) + '_trip' + str(trip_cl) + '_' + dt + '.pickle')
    pandasResults = results.getEstimatedParameters()
    gs = results.getGeneralStatistics()
    
    sc_f = 1 # / -np.min(pandasResults['Value']['B_COST'], pandasResults['Value']['B_TIME'])
    
    ############### ADJUST BELOW THIS LINE ###############
    
    modes = ['car', 'carp', 'transit', 'cycle', 'walk', 'future']
    
    P = np.empty((0, 6))
    P_future = np.empty((0, 7))
    modal_split_actual = [0, 0, 0, 0, 0]
    modal_split_future = [0, 0, 0, 0, 0, 0]
    
    
    print('Number of combinations future mode tested', len(future_mode_char))
    
    # Create empty matrix to store results
    w, h, d = 6, len(future_mode_char), len(df_selection)
    P_temp = [[[0 for x in range(w)] for y in range(h)] for z in range(d)]
    
    for i in range(len(df_selection)):
        
        V = [0, 0, 0, 0, 0, 0]
        distance_trip = [0, 0, 0, 0, 0]
        distance_trip_counter = 0
        assumed_average_speed = [60, 60, 40, 15, 5] # [car, carp, transit, cycle, walk]
        
        # try:
        for j in range(5):
            
            if df_selection.iloc[i]["CHOICE"] == j + 1:
                modal_split_actual[j] += 1
            
            
            # V[j] = sc_f * (float(pandasResults['Value']['B_COST']) * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["time_" + modes[j]])
            
            V[j] = sc_f * (float(pandasResults['Value']['B_0']) * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_1']) * df_selection.iloc[i]["time_" + modes[j]] + \
              float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * df_selection.iloc[i]["drivingtask_" + modes[j]] + \
              float(pandasResults['Value']['B_4']) * df_selection.iloc[i]["skills_" + modes[j]] + float(pandasResults['Value']['B_5']) * df_selection.iloc[i]["weatherProtection_" + modes[j]] + \
              float(pandasResults['Value']['B_6']) * df_selection.iloc[i]["luggage_" + modes[j]] + float(pandasResults['Value']['B_7']) * df_selection.iloc[i]["shared_" + modes[j]] + \
              float(pandasResults['Value']['B_8']) * df_selection.iloc[i]["availability_" + modes[j]] + float(pandasResults['Value']['B_9']) * df_selection.iloc[i]["reservation_" + modes[j]] + \
              float(pandasResults['Value']['B_10']) * df_selection.iloc[i]["active_" + modes[j]] + float(pandasResults['Value']['B_11']) * df_selection.iloc[i]["accessible_" + modes[j]])

            # Utility synthetic data
            # V[j] = float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["age"] * df_selection.iloc[i]["time_" + modes[j]] * sc_f + float(pandasResults['Value']['B_COST']) * 200000 / df_selection.iloc[i]["income"] * df_selection.iloc[i]["cost_" + modes[j]] * sc_f # Exactly the same as in data generation
        
        
            # Estimate distance in dataset based on travel times of all existing modes
            # average of (time for available mode) / (assumed average speed for available mode)
            if (df_selection.iloc[i]["time_" + modes[j]] != 9999):
                distance_trip[j] = df_selection.iloc[i]["time_" + modes[j]] / assumed_average_speed[j]
                distance_trip_counter += 1
            else:
                distance_trip[j] = 0
                
        E = [0, 0, 0, 0, 0]
        V_nest = (sc_f / float(pandasResults['Value']['MU_1'])) * np.log(np.exp(float(pandasResults['Value']['MU_1']) * V[3]) + np.exp(float(pandasResults['Value']['MU_1']) * V[4]))
        V_nest_1 = (sc_f / float(pandasResults['Value']['MU_2'])) * np.log(np.exp(float(pandasResults['Value']['MU_2']) * V[1]) + np.exp(float(pandasResults['Value']['MU_2']) * V[2]))
      
        for j in range(5):
            if (j == 0):
                E[j] = np.exp(V[j]) / (np.exp(V[0]) + np.exp(V_nest_1) + np.exp(V_nest))
            elif (j == 1) or (j == 2):
                E[j] = np.exp(float(pandasResults['Value']['MU_2']) * V[j]) / (np.exp(V[0]) + np.exp(V_nest_1) + np.exp(V_nest))
            else:
                E[j] = np.exp(float(pandasResults['Value']['MU_1']) * V[j]) / (np.exp(V[0]) + np.exp(V_nest_1) + np.exp(V_nest))
        
        choice = E.index(max(E)) + 1
        P = np.append(P, np.array([[E[0], E[1], E[2], E[3], E[4], choice]]), axis=0) # nrOfModes in modal split
            
        
        # Add utility function of new mobility system here.
        # V[5] = -100 # sc_f * (float(pandasResults['Value']['B_COST']) * df_selection.iloc[i]["c_" + modes[j]] + float(pandasResults['Value']['B_DUR1']) * df_selection.iloc[i]["t_" + modes[j]])
 
        # Estimate distance in dataset based on travel times of all existing modes
        # average of (time for available mode) / (assumed average speed for available mode)
        distance_estimated = sum(distance_trip) / distance_trip_counter
        
        
        for j in range(len(future_mode_char)):
            try:
                # Variable costs (per km)
                # V[5] = sc_f * (float(pandasResults['Value']['B_0']) * future_mode_char[j, 0] * distance_estimated + float(pandasResults['Value']['B_1']) * (distance_estimated / future_mode_char[j, 1]) / 60 + \
                #       float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * future_mode_char[j, 2] + \
                #       float(pandasResults['Value']['B_4']) * future_mode_char[j, 3] + float(pandasResults['Value']['B_5']) * future_mode_char[j, 4] + \
                #       float(pandasResults['Value']['B_6']) * future_mode_char[j, 5] + float(pandasResults['Value']['B_7']) * future_mode_char[j, 6] + \
                #       float(pandasResults['Value']['B_8']) * future_mode_char[j, 7] + float(pandasResults['Value']['B_9']) * future_mode_char[j, 8] + \
                #       float(pandasResults['Value']['B_10']) * future_mode_char[j, 9] + float(pandasResults['Value']['B_11']) * future_mode_char[j, 10])
                
                # Fixed costs
                V[5] = sc_f * (float(pandasResults['Value']['B_0']) * future_mode_char[j, 0] + float(pandasResults['Value']['B_1']) * (distance_estimated / future_mode_char[j, 1]) / 60 + \
                      float(pandasResults['Value']['B_2']) * df_selection.iloc[i]["driving_license"] + float(pandasResults['Value']['B_3']) * future_mode_char[j, 2] + \
                      float(pandasResults['Value']['B_4']) * future_mode_char[j, 3] + float(pandasResults['Value']['B_5']) * future_mode_char[j, 4] + \
                      float(pandasResults['Value']['B_6']) * future_mode_char[j, 5] + float(pandasResults['Value']['B_7']) * future_mode_char[j, 6] + \
                      float(pandasResults['Value']['B_8']) * future_mode_char[j, 7] + float(pandasResults['Value']['B_9']) * future_mode_char[j, 8] + \
                      float(pandasResults['Value']['B_10']) * future_mode_char[j, 9] + float(pandasResults['Value']['B_11']) * future_mode_char[j, 10])
            except:
                V[5] = -100
                print('Could not calculate V[5], retrieving pandasResults failed for', trip_cl, pers_cl)
            # if(j % 100==0):
            #     print('Progress estimate future modal split', j/len(future_mode_char))
        # Utility synthetic data
        # V[5] = sc_f * (float(pandasResults['Value']['B_COST']) * 200000 / df_selection.iloc[i]["income"] * df_selection.iloc[i]["cost_" + modes[j]] + float(pandasResults['Value']['B_TIME']) * df_selection.iloc[i]["age"] * df_selection.iloc[i]["time_" + modes[j]])

            E_future = [0, 0, 0, 0, 0, 0]
            
            # Which existing mode will be nested with the future mode?
            # nest = 3
            # mu_b =  sc_f / max_sim
            
            for k in range(6):
                    
                if nest == 0:
                    V_nest = (sc_f / float(pandasResults['Value']['MU_1'])) * np.log(np.exp(float(pandasResults['Value']['MU_1']) * V[3]) + np.exp(float(pandasResults['Value']['MU_1']) * V[4]))    
                    V_nest_1 = (sc_f / float(pandasResults['Value']['MU_2'])) * np.log(np.exp(float(pandasResults['Value']['MU_2']) * V[1]) + np.exp(float(pandasResults['Value']['MU_2']) * V[2]))
                    V_nest_2 = (sc_f / mu_b) * np.log(np.exp(mu_b * V[nest]) + np.exp(mu_b * V[5]))

                    if (k == 1) or (k == 2):
                        E_future[k] = np.exp(float(pandasResults['Value']['MU_1']) * V[k]) / (np.exp(V_nest) + np.exp(V_nest_1) + np.exp(V_nest_2))
                    elif (k == 3) or (k == 4):
                        E_future[k] = np.exp(float(pandasResults['Value']['MU_2']) * V[k]) / (np.exp(V_nest) + np.exp(V_nest_1) + np.exp(V_nest_2))
                    else:
                        E_future[k] = np.exp(mu_b * V[k]) / (np.exp(V_nest) + np.exp(V_nest_1) + np.exp(V_nest_2))
                    
                elif (nest == 1) or (nest == 2):
                    V_nest = (sc_f / float(pandasResults['Value']['MU_1'])) * np.log(np.exp(float(pandasResults['Value']['MU_1']) * V[3]) + np.exp(float(pandasResults['Value']['MU_1']) * V[4]))    
                    V_nest_1 = (sc_f / float(pandasResults['Value']['MU_2'])) * np.log(np.exp(float(pandasResults['Value']['MU_2']) * V[1]) + np.exp(float(pandasResults['Value']['MU_2']) * V[2]) + np.exp(float(pandasResults['Value']['MU_2']) * V[5]))
                
                    if (k == 1) or (k == 2) or (k == 5):
                        E_future[k] = np.exp(float(pandasResults['Value']['MU_2']) * V[k]) / (np.exp(V_nest) + np.exp(V_nest_1) + np.exp(V[0]))
                    elif (k == 3) or (k == 4):
                        E_future[k] = np.exp(float(pandasResults['Value']['MU_1']) * V[k]) / (np.exp(V_nest) + np.exp(V_nest_1) + np.exp(V[0]))
                    else:
                        E_future[k] = np.exp(V[k]) / (np.exp(V_nest) + np.exp(V_nest_1) + np.exp(V[0]))
                    
                else:
                    V_nest_1 = (sc_f / float(pandasResults['Value']['MU_2'])) * np.log(np.exp(float(pandasResults['Value']['MU_2']) * V[1]) + np.exp(float(pandasResults['Value']['MU_2']) * V[2]))    
                    V_nest = (sc_f / float(pandasResults['Value']['MU_1'])) * np.log(np.exp(float(pandasResults['Value']['MU_1']) * V[3]) + np.exp(float(pandasResults['Value']['MU_1']) * V[4]) + np.exp(float(pandasResults['Value']['MU_1']) * V[5]))
                
                    if (k == 3) or (k == 4) or (k == 5):
                        E_future[k] = np.exp(float(pandasResults['Value']['MU_1']) * V[k]) / (np.exp(V_nest) + np.exp(V_nest_1) + np.exp(V[0]))
                    elif (k == 1) or (k == 2):
                        E_future[k] = np.exp(float(pandasResults['Value']['MU_2']) * V[k]) / (np.exp(V_nest) + np.exp(V_nest_1) + np.exp(V[0]))
                    else:
                        E_future[k] = np.exp(V[k]) / (np.exp(V_nest) + np.exp(V_nest_1) + np.exp(V[0]))
                    
                    
                try:
                    P_temp[i][j][k] = E_future[k]
                except:
                    print(E_future)
            
        choice_future = E_future.index(max(E_future)) + 1
            
        P_future = np.append(P_future, np.array([[E_future[0], E_future[1], E_future[2], E_future[3], E_future[4], E_future[5], choice_future]]), axis=0)
        print('Progress estimate modal split', (i/len(df_selection)))
        
    modal_split = np.sum(P[:,0:5], axis=0) / np.sum(np.sum(P[:,0:5], axis=0)) # nrOfModes in modal split
    
    
    res_future = np.empty((0, 10))
    
    for j in range(len(future_mode_char)):
        res_temp = [[0 for x in range(6)] for y in range(len(df_selection))]
        for i in range(len(df_selection)):
            res_temp[i][:] = P_temp[i][j][:]
        
        
        modal_split_future = np.sum(res_temp[:][:], axis=0) / np.sum(np.sum(res_temp[:][:], axis=0))
        res_future = np.append(res_future, np.array([[j, trip_cl, pers_cl, len(df_selection), modal_split_future[0], modal_split_future[1], modal_split_future[2], modal_split_future[3], modal_split_future[4], modal_split_future[5]]]), axis=0)
        

    
    modal_split_future = np.sum(P_future[:,0:6], axis=0) / np.sum(np.sum(P_future[:,0:6], axis=0))
    
    R_square = r2_score(P[:, 5], df_selection["CHOICE"])
    print('Coefficient of Determination', R_square) 
    
    report = metrics.classification_report(df_selection['CHOICE'], P[:,5], digits=3, output_dict=True) # nrOfModes in modal split
    results = pd.DataFrame(report).transpose()
    results['trip'] = trip_cl
    results['pers'] = pers_cl
    
    results['modal_split_estimated'] = 0
    results['modal_split_actual'] = 0
    results['modal_split_future'] = 0
    for i in np.arange(1, 6): # nrOfModes in modal split
        results.loc['{0}'.format(i), 'modal_split_estimated'] = modal_split[i-1]
        results.loc['{0}'.format(i), 'modal_split_actual'] = (modal_split_actual[i-1] / np.sum(modal_split_actual))
        results.loc['{0}'.format(i), 'modal_split_future'] = modal_split_future[i-1] # Note: mode 6 is not included here
    
    results.dropna(subset = ["precision"], inplace=True)
    
    ############### ADJUST ABOVE THIS LINE ###############
    
    pandasResults['trip'] = trip_cl
    pandasResults['pers'] = pers_cl

    print('Finished estimating modal split for trip cluster ' + str(trip_cl) + ' and pers cluster ' + str(pers_cl))
    
    return results, pandasResults, gs, modal_split_future[5], R_square, res_future


def interpret_result(nrOfPar, act_asc, nrOfClusters, dt, subset, future_mode_char):
    
    np.seterr(divide='ignore', invalid='ignore')
    
    # Accuracy, recall, precision, modal split
    
    df_perf = pd.read_csv("data/" + subset + "_results_performance_par" + str(nrOfPar) + "_asc" + str(act_asc) + '_' + dt + ".csv")
    df_stats = pd.read_csv("data/" + subset + "_results_stats_par" + str(nrOfPar) + "_asc" + str(act_asc) + '_' + dt + ".csv")

    idx_1 = np.arange(0, 9*nrOfClusters, 9)
    idx_2 = [x+1 for x in idx_1]
    idx_3 = [x+2 for x in idx_1]
    idx_4 = [x+3 for x in idx_1]
    idx_5 = [x+4 for x in idx_1]
    idx_6 = [x+5 for x in idx_1]
    idx_acc = [x+6 for x in idx_1]
    idx_macro = [x+7 for x in idx_1]
    idx_wghtd = [x+8 for x in idx_1]
    
    idx = [idx_1, idx_2, idx_3, idx_4, idx_5]
    
    with open("data/" + subset + "_results_summary_par" + str(nrOfPar) + "_asc" + str(act_asc) + '_' + dt + '.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        
        writer.writerow(['mode', 'precision', 'recall', 'f1-score', 'modal split', 'modal split actual', 'modal split future'])
        
        for count, value in enumerate(idx):
        
            prec = (df_perf['support'][value].values * df_perf['precision'][value]).sum() / df_perf['support'][value].values.sum()
            recall = (df_perf['support'][value].values * df_perf['recall'][value]).sum() / df_perf['support'][value].values.sum()
            f1 = (df_perf['support'][value].values * df_perf['f1-score'][value]).sum() / df_perf['support'][value].values.sum()

            split = (df_perf['support'][idx_macro].values * df_perf['modal_split_estimated'][value].values).sum() / df_perf['support'][idx_macro].sum()
            split_actual = (df_perf['support'][idx_macro].values * df_perf['modal_split_actual'][value].values).sum() / df_perf['support'][idx_macro].sum()
            split_future = (df_perf['support'][idx_macro].values * df_perf['modal_split_future'][value].values).sum() / df_perf['support'][idx_macro].sum()
            row = [count + 1, prec, recall, f1, split, split_actual, split_future]
            writer.writerow(row)
        
        # Modal split future mode
        split_future = (df_perf['support'][idx_macro].values * df_perf['modal_split_future'][idx_6].values).sum() / df_perf['support'][idx_macro].sum()
        row = ['6', '0', '0', '0', '0', '0', split_future]
        writer.writerow(row)
        
        # Accuracy, macro average, weighted average resp.
        row = ['accuracy', (df_perf['support'][idx_macro].values * df_perf['precision'][idx_acc].values).sum() / df_perf['support'][idx_macro].sum()]
        writer.writerow(row)
        
        prec = (df_perf['support'][idx_macro] * df_perf['precision'][idx_macro]).sum() / df_perf['support'][idx_macro].sum()
        recall = (df_perf['support'][idx_macro] * df_perf['recall'][idx_macro]).sum() / df_perf['support'][idx_macro].sum()
        f1 = (df_perf['support'][idx_macro] * df_perf['f1-score'][idx_macro]).sum() / df_perf['support'][idx_macro].sum()
        row = ['macro average', prec, recall, f1]    
        writer.writerow(row)
        
        prec = (df_perf['support'][idx_wghtd] * df_perf['precision'][idx_wghtd]).sum() / df_perf['support'][idx_wghtd].sum()
        recall = (df_perf['support'][idx_wghtd] * df_perf['recall'][idx_wghtd]).sum() / df_perf['support'][idx_wghtd].sum()
        f1 = (df_perf['support'][idx_wghtd] * df_perf['f1-score'][idx_wghtd]).sum() / df_perf['support'][idx_wghtd].sum()
        row = ['weighted average', prec, recall, f1]    
        writer.writerow(row)

        # Init. & final log, R2, R2bar, BIC
        row = ['InitLog', df_stats['InitLog'].values.sum()]
        writer.writerow(row)     
        
        row = ['FinalLog', df_stats['FinalLog'].values.sum()]
        writer.writerow(row)  
        
        row = ['R2', (df_perf['support'][idx_macro].values * df_stats['R2'].values).sum() / df_perf['support'][idx_macro].sum()]
        writer.writerow(row) 
        
        row = ['R2bar', (df_perf['support'][idx_macro].values * df_stats['R2bar'].values).sum() / df_perf['support'][idx_macro].sum()]
        writer.writerow(row) 
        
        row = ['BIC', (nrOfPar * np.log(df_perf['support'][idx_macro].sum()) - 2 * np.log(np.absolute(df_stats['FinalLog'].values.sum())))]
        writer.writerow(row)
        
        row = ['R2-est', (df_perf['support'][idx_macro].values * df_stats['R2-est'].values).sum() / df_perf['support'][idx_macro].sum()]
        writer.writerow(row) 
 
        
    df_future = pd.read_csv("data/" + subset + "_results_future_par" + str(nrOfPar) + "_asc" + str(act_asc) + '_' + dt + ".csv")
    
    with open("data/" + subset + "_modal_split_future_par" + str(nrOfPar) + "_asc" + str(act_asc) + '_' + dt + ".csv", 'w') as f:
        # create the csv writer
        writer = csv.writer(f)  
        
        row = ['cost', 'time', 'drivingtask', 'skills', 'weatherprotection', 'luggage', 'shared', 'availability', 'reservation', 'active', 'accessible', 'modal_split_car', 'modal_split_carp', 'modal_split_transit', 'modal_split_cycle', 'modal_split_walk', 'modal_split_future', 'support_incl_nan', 'support_excl_nan_used_in_calcs']
        writer.writerow(row)
    
        split_future = [0, 0, 0, 0, 0, 0]
        
        for i in range(len(future_mode_char)):
            idx_full = df_future.index[df_future['idx_future_mode_char'] == i].tolist()
            df_future_new = df_future.dropna()
            idx = df_future_new.index[df_future_new['idx_future_mode_char'] == i].tolist()
            
            for c, j in enumerate(['car', 'carp', 'transit', 'cycle', 'walk', 'future']):
                split_future[c] = (df_future_new['size'][idx].values * df_future_new[j][idx].values).sum() / df_future_new['size'][idx].sum()
            
            chars = future_mode_char[i, :]
            row = np.concatenate((chars, split_future, [df_future['size'][idx_full].sum(), df_future_new['size'][idx].sum()]), axis=None)
            writer.writerow(row)
            



