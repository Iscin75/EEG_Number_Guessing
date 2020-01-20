# -*- coding: utf-8 -*-
from tkinter import *
from tkinter import filedialog


root = Tk()
DATA_PATH = ""
PLAY_PATH = ""
HAS_BEEN_TRAINED = False
GUESSED_NUMBER = StringVar(root)

def load_data(event):
    print("Load")
    global DATA_PATH
    DATA_PATH = filedialog.askopenfilename(initialdir = "/",title = "Select data file")
    print(DATA_PATH)
    
def train_model(event):
    print("training")
    global HAS_BEEN_TRAINED
    if(DATA_PATH != ""):
        print("OK")
        HAS_BEEN_TRAINED = True
    else:
        print("Pas de Path")
        
    
def play_sample(event):
    print("play")
    global PLAY_PATH
    global GUESSED_NUMBER
    
    if(HAS_BEEN_TRAINED):
        PLAY_PATH = filedialog.askopenfilename(initialdir = "/",title = "Select playing EEG")
        GUESSED_NUMBER.set('1')
    else:
        print("Model need to be trained before starting guessing")
    print(PLAY_PATH)

def create_interface(root):
    
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

