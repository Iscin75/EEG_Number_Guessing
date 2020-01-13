# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mne

import keras
from keras.callbacks import TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import LSTM

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)

DATA_PATH = 'data/'

def get_all_ids():
    
    
    ids = []
    
    for folder in os.listdir('./data/'):
         if "Experiment" in folder:
             ids.append(folder.split('_')[1])
    return ids

def get_all_thought_numbers():
    
    numbers = []
    
    for folder in os.listdir('./data/'):
         if "Experiment" in folder:
             for file in os.listdir('./data/' + folder + '/'):
                 if file.endswith('.txt'):
                     f = open('./data/' + folder + '/' + file, "r")
                     for line in f:
                      if("thought" in line):
                          numbers.append((line.split(' ')[-1]).rstrip("\n\r"))
    return numbers

def fill_csv(ids, numbers):
    
    with open('experiments.csv', 'w') as f:

        fnames = ['id_experiment', 'number_thought']
        writer = csv.DictWriter(f, fieldnames=fnames) 
        writer.writeheader()
        
        for i in range(len(ids)):
            writer.writerow({'id_experiment' : ids[i], 'number_thought': numbers[i]})
        

def get_eids_from_directory():

    all_ids = []
    for _, dirnames, files in os.walk(DATA_PATH):
        if dirnames == []:
            all_ids.append(int(_.split('_')[1]))
    return all_ids

def get_eeg_path(eeg_id):
 
    eeg_folder = "Experiment_" + str(eeg_id) + "_P3_Numbers/"
    
    for file in os.listdir(DATA_PATH + eeg_folder):
        if file.endswith(".vhdr"):
            return os.path.join(DATA_PATH + eeg_folder, file)

def data_from_eeg(eeg_id):

    filename = get_eeg_path(eeg_id)
    raw = mne.io.read_raw_brainvision(filename)
    print(raw)
    print(raw.info)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=True,
                       exclude='bads')

    t_idx = raw.time_as_index([5., 10.])  
    data, times = raw[picks, t_idx[0]:t_idx[1]] 
    
    return data

def create_dataset_array(df):
    numbers = []
    count = 0
    X_eeg = []
    
    for index, row in df.iterrows():
        try:
            
            count += 1
            eeg_id = int(row['id_experiment'])
            number_thought = str(row['number_thought'])
            data = data_from_eeg(eeg_id)

 
            X_eeg.append([data])
            numbers.append(number_thought)
            if count % 100 == 0:
                print("Iteration number: ", count, '/', len(df))
        except Exception as e:
            print(e)
            print("Couldn't process: ", count)
            continue
    y_arr = np.array(numbers)
    return X_eeg, y_arr

def create_neural_network(X_arr):
    
    model = Sequential()
    model.add(LSTM(input_shape=(None, X_arr.input_shape[0] * X_arr.input_shape[1]), return_sequences=True))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(LSTM(return_sequences=True))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(LSTM(return_sequences=False))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(9))
    model.add(Activation("softmax"))
    
    
    return model

def display_graphes(history):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
      
     
    plt.semilogy(history.history['loss'])
    plt.semilogy(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def train_neural_network(model, X_train, y_train, X_val, y_val, X_test, y_test):
    
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     
     history = model.fit(X_train, y_train,
              batch_size=2,
              epochs=20,
              validation_data=(X_val, y_val),
              shuffle=True,
              verbose=True,
              )
    
     result = model.evaluate(X_test, y_test, verbose=0)
     display_graphes(history)
    
     model.save('number_guession_model.h5')
    
if __name__ == "__main__":
    
    
    filepath = 'experiments.csv'
    eegs = pd.read_csv(filepath)
    print(eegs)
    X_eeg, y_arr = create_dataset_array(eegs)

    X_train, X_test, y_train, y_test = train_test_split(X_eeg, y_arr, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    
    #model = create_neural_network(X_eeg)
    #train_neural_network(model, X_train, y_train, X_val, y_val, X_test, y_test)
