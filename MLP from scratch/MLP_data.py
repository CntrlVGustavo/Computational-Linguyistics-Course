#import torch
import csv
import numpy as np
#from torch import nn


def make_batches(batch_size):
    #initializing variables 
    batches = []
    batch = []
    iter = 0

    #opening the csv file
    with open('fashion-mnist_test.csv', 'r') as file:
        #making a readable object
        reader = csv.reader(file)

        next(reader) #skipping the first row of the csv file

        #looping through each row of the csv file and making the bacthes
        for line in reader:
            if (iter / batch_size) != 1: 
                batch.append(line)
                iter += 1
            else:
                batches.append(batch)
                batch = []
                batch.append(line)
                iter = 1
        batches.append(batch)
    return np.array(batches)


#b = make_batches(250)
#d = 785
#print(b)