# -*- coding: utf-8 -*-
from tkinter import *
from tkinter import filedialog

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import mne

from mne import pick_channels, concatenate_epochs
from mne import pick_types, viz, io, Epochs, create_info
from mne import channels, find_events, concatenate_raws

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

import matplotlib.pyplot as plt


#----------Global Variables ----------#

root = Tk()
DATA_PATH = 'data/'
CSV_PATH = 'experiments.csv'
PLAY_PATH = ""
HAS_BEEN_TRAINED = False
GUESSED_NUMBER = StringVar(root)
X_arr = []
y_arr = []
raw_data = []
model = tf.keras.Sequential()

#---------- Data preparation methods ----------#

def get_all_ids():
    ''' Get all the experiments ids using their folder nomenclature '''    
    ids = []
    
    for folder in os.listdir('./data/'):
         if "Experiment" in folder:
             ids.append(folder.split('_')[1])
    return ids

def get_all_thought_numbers():
    ''' Get all the experiments guessed number reading the experiment .txt file '''
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

def get_eeg_path(eeg_id):
    ''' Return the path of the currrent processed EEG file, using it ID '''
    
    eeg_folder = "Experiment_" + str(eeg_id) + "_P3_Numbers/"
    
    for file in os.listdir(DATA_PATH + eeg_folder):
        if file.endswith(".vhdr"):
            return os.path.join(DATA_PATH + eeg_folder, file)

def load_eeg(filename, plot_sensors=True, plot_raw=True,
  plot_raw_psd=True ):
    ''' load the current selected EEG file data using it .vhdr (header) file '''
 
    raw = mne.io.read_raw_brainvision(filename, preload=True)

    sfreq = raw.info['sfreq']

    if plot_sensors:
        raw.plot_sensors(show_names='True')


    if plot_raw:
        raw.plot(n_channels=16, block=True)


    if plot_raw_psd:
        raw.plot_psd(fmin=.1, fmax=100 )

    return raw, sfreq

def load_all_eeg(raw_data):
  '''' Return two arrays containing all the EEG data and the associated number thought value '''
  
  global GUESSED_NUMBER
  
  X_arr = []
  y_arr = []
    
  for index, row in raw_data.iterrows():
      raw, freq = load_eeg(get_eeg_path(row['id_experiment']),plot_sensors=False,plot_raw=False,
              plot_raw_psd=False)
      X_arr.append(raw) # Didn't find the right solution to format the data for NN
      y_arr.append(row['number_thought'])
        
  GUESSED_NUMBER.set("Data loaded successfullly, let's train the model!")
  return X_arr, y_arr

#---------- Neural Network methods ----------#

def create_model(X_arr):
    ''' Define the Neural Network LSTM architecture with all its layers '''
    global model
    
    units=[256,256,256,256]
    nunits = len(units)

    
    model.add(layers.LSTM(input_shape=(4,273267), # Need to convert the X_arr to right format to access array.shape
               units=units[0], return_sequences=True))

    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(.25))


    for unit in units[1:-1]:
        model.add(layers.LSTM(units=unit,return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(.25))

    model.add(layers.LSTM(units=units[-1],return_sequences=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(.25))

    model.add(layers.Dense(units=9))
    model.add(layers.Activation("softmax"))

    model.compile(loss='binary_crossentropy',
          optimizer=tf.keras.optimizers.Adam(0.01),
          metrics=['accuracy'])



    model.summary()
    
    global GUESSED_NUMBER
    GUESSED_NUMBER.set("Model created successfully, training model...")

    return model

def train_test_model():
    ''' Train the model created in create_model method using the EEG data splitted in train, validation and test dataset 
    and display model acc and loss'''
  
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    
    history = model.fit(X_train, y_train,
    batch_size=2,
    epochs=20,
    validation_data=(X_val, y_val),
    shuffle=True,
    verbose=True,
    )

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



    score, acc = model.evaluate(X_test, y_test,
                batch_size=2)
    print('Test loss:', score)
    print('Test accuracy:', acc)

    data = {}
    data['score'] = score
    data['acc'] = acc
    
    global GUESSED_NUMBER
    GUESSED_NUMBER.set("Model training done, you can start playing!")
    
    return model, data

def guess_eeg_number():
    ''' Guess the thought number associated to the EEG file located in PLAY_PATH '''
    
    
    X_play, freq = load_eeg(get_eeg_path(PLAY_PATH ,plot_sensors=False,plot_raw=False, plot_raw_psd=False))
    
    y_play = model.predict(X_play)
    return y_play

#---------- UI's methods ----------#

def load_data(event):
    """Load the project data from the data/ folder"""
    print("Load")
    global X_arr
    global y_arr
    global raw_data
    
    raw_data = pd.read_csv(CSV_PATH)
    X_arr, y_arr = load_all_eeg(raw_data)
    
    print(DATA_PATH)
    
def train_model(event):
    """ Launch the training of the Neural Network with the data at the DATA_PATH location when clicking on Train button"""
    print("training")
    global HAS_BEEN_TRAINED
    global model
    if(DATA_PATH != ""):
        print("OK")
        model = create_model(X_arr)
        train_test_model()
        
        HAS_BEEN_TRAINED = True
    else:
        print("Pas de Path")
        
    
def play_sample(event):
    """ Try to predict the sample EEG number thought through our Neural Network when clicking on Play button"""
    print("play")
    global PLAY_PATH
    global GUESSED_NUMBER
    
    if(HAS_BEEN_TRAINED):
        PLAY_PATH = filedialog.askopenfilename(initialdir = "/",title = "Select playing EEG",filetypes = ((".vhdr")))
        if(PLAY_PATH != ""):
            GUESSED_NUMBER.set(str(guess_eeg_number()[0]))
    else:
        print("Model need to be trained before starting guessing")
    print(PLAY_PATH)

def create_interface(root):
    """ Build the Tkinter windows interface with it labels and buttons"""
    root.geometry("500x250")
    root.title('EEG Number Guessing')
    
    global GUESSED_NUMBER
    
    Label(text = "Number Guessed :", bg = "grey", width = "55", height = "1", font = ("Calibri", 13)).grid(row=1, column=0, columnspan=3)
    Label(textvariable = GUESSED_NUMBER, bg = "grey", width = "55", height = "1", font = ("Calibri", 13)).grid(row=2, column=0, columnspan=3)
    bt_load = Button(root,text = 'LOAD',width = 15, height = 5, )
    bt_load.grid(padx=25, pady=10, row=3, column=0)     
    bt_train = Button(root,text = 'TRAIN',width = 15, height = 5, )
    bt_train.grid(padx=25, pady=20, row=3, column=1)    
    bt_play = Button(root,text = 'PLAY',width = 15, height = 5, )
    bt_play.grid(padx=25, pady=20, row=3, column=2) 
    
    bt_load.bind('<Button-1>', load_data)
    bt_train.bind('<Button-1>', train_model)
    bt_play.bind('<Button-1>', play_sample)
    
    
if __name__ == "__main__":
    
    create_interface(root)
    root.mainloop()

