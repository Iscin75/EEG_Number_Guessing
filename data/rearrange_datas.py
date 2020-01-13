

import os
import shutil
import zipfile

def clear_useless_data():
    for x in os.listdir("./"):
        if os.path.isdir("./" + x):
            shutil.rmtree('./' + x + '/License', ignore_errors=True)
            shutil.rmtree('./' + x + '/Scenario', ignore_errors=True)
            if os.path.exists('./' + x + '/metadata.xml'):
                os.remove('./' + x + '/metadata.xml')

           
def extract_zip_data():
    for x in os.listdir("./"):
        if os.path.isdir("./" + x):
            for y in os.listdir("./" + x + '/Data/'):
             if('.zip' in y):
                 with zipfile.ZipFile("./" + x + '/Data/' + y) as Z :
                     for elem in Z.namelist() :
                         Z.extract(elem, "./" + x)
      
            
                            
def move_txt_data():
    for x in os.listdir("./"):
        if os.path.isdir("./" + x):
            for y in os.listdir("./" + x + '/Data/'):
             if('.txt' in y):
                 shutil.move(os.path.join("./" + x + '/Data/', y), os.path.join("./" + x, y))
                 
                 
def delete_data_folder():
    for x in os.listdir("./"):
        if os.path.isdir("./" + x):
            shutil.rmtree("./" + x + "/Data")
            
