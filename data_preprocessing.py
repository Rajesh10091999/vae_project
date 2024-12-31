import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

# NOTE: Standard arrangment of data [n_samples, n_features], compatible with train_test_split
## NOTE: probably, we need to employ physical scaling.
## Question: Why shift the prior in test set?

rnd_state = 42
def data_preprocessing(data,train_test_percent=0.2,test_valid_percent=0.5, noise_frac=0.00, n_poi = 1):
    input_dim = data.shape[1] - n_poi
    n_sample = data.shape[0]  
    data_train,data_res = train_test_split(data,test_size=train_test_percent,random_state=rnd_state)
    data_test,data_valid = train_test_split(data_res,test_size=test_valid_percent, random_state=rnd_state)
    
    # Split the data into input and output
    
    u_train = data_train[:,:-n_poi]
    poi_train = data_train[:,-n_poi:]
    #u_train_noisy = u_train + np.transpose(np.random.normal(0.0,u_train.std(axis=1),np.transpose(u_train).shape))*noise_frac
    u_train_noisy = np.zeros_like(u_train)
    # Generate noise for each column
    sigmas = u_train.std(axis=1)
    for i in np.arange(u_train.shape[0]):
        u_train_noisy[i,:] = (u_train[i,:] + np.random.normal(0.0,sigmas[i],input_dim)*noise_frac).reshape(1,-1)
    data_noisy = np.column_stack([u_train_noisy,poi_train])

    # Create a StandardScaler instance
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(data_noisy)
    test_data_scaled = scaler.fit_transform(data_test)
    valid_data_scaled = scaler.fit_transform(data_valid)

    # shifting the orginal values by 25% fro prior mean
    poi_train_prior = train_data_scaled[:,-n_poi:]+0.25*train_data_scaled[:,-n_poi:]
    poi_test_prior = test_data_scaled[:,-n_poi:]+0.25*test_data_scaled[:,-n_poi:]
    poi_valid_prior = valid_data_scaled[:,-n_poi:]+0.25*valid_data_scaled[:,-n_poi:]
    
    return train_data_scaled, test_data_scaled, valid_data_scaled, poi_train_prior, poi_test_prior, poi_valid_prior 
