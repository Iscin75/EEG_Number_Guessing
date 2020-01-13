import csv
import os

def write_data_in_csv():
    header = ["id_experiment", "number_thought"]

    with open('experiments.csv', 'wt', newline ='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(i for i in header)
        for x in os.listdir("./data/"):
            if os.path.isdir("./data/" + x):
                for y in os.listdir("./data/" + x):
                    if(".txt" in y):
                        with open("./data/" + x + "/" + y, "r") as f :
                            line = f.readline()
                            while line:
                                if("the number thought:" in line):
                                    print(x.split('_')[1] + " - " +line.split(' ')[-1].rstrip('\n'))
                                    writer.writerow([x.split('_')[1] ,line.split(' ')[-1].rstrip('\n')])
                                    break;
                                else:
                                    line = f.readline()
                            f.close()
    file.close()
                
            
            
write_data_in_csv()