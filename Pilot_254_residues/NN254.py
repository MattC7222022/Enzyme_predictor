#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:17:59 2024

@author: mattc
This version adds the length of the protein to the input

fixed the backpropagation so that the softmax outputs should work now. They don't train in the previous script'

This version also is changed to use ReLU as an activation function

adjusts normalization of amino acids to have a larger range

makes test code that checks if the setup even has potential

this version only uses the 254 residue data, and includes functions to really loop through and generate 
models with statistics
"""

import numpy as np
import pandas as pd
import time

Model_type = "Regular feed forward NN with only ReLu until the output layer. First input data is number of amino acids"
np.random.seed(int((time.time())*10000)%10000)


amino_acids_pI = {'D': 2.98, 'E': 3.08, 'C': 5.15, 'N': 5.43, 'T': 5.60, 'Y': 5.63, 'Q': 5.65, 'S': 5.70, 'M': 5.71, 'F': 5.76, 'W': 5.88, 'V': 6.00, 'I': 6.03, 'L': 6.04, 'G': 6.06, 'A': 6.11, 'P': 6.30, 'H': 7.64, 'K': 9.47, 'R': 10.76, 'Z':4.37, 'X':6.87, 'J':6.04, 'U':3.9, 'B':4.2} 
pI_normalized = {x:(10*(amino_acids_pI[x]-2.98)/(10.76-2.98))  for x in amino_acids_pI}
del amino_acids_pI
#This is a dictionary of Amino acid single letter symbols (used in sequence files) to their PIs.
#for Z, I used the mean pIs of Glutamic acid and Glutamine (0.286245353 + 0.525092937)/2 =  0.405669145
#for X, I used the mean of the lowest pH, Aspartic acid, and the highest, Arginine, (0.276951673 + 1)/2 = 0.6384758365
#for J, I used the means of Leucine and isoleucine, 0.560808922
#for U, I used selenocysteine 3.9/10.76 = 0.3624535
#for B, I used the mean of aspartic acid and asparagine, (0.276951673 + 0.50464684)/2 =0.3907992565

def prepare_data(path, lim = 250):
    #this function takes a file of training data, and turns the sequence into pI values and the EC value into a list of 1s and 0s
    training_data = pd.read_excel(path, sheet_name = "Sheet0")
    training_data = training_data[training_data["Length"] <= lim]
    entries = []
    EC_list = []
    seq_list = []
    for x in range(len(training_data)):
        entries.append(training_data.iloc[x,1])
        EC = training_data.iloc[x,4]
        if not isinstance(training_data.iloc[x,4], str):
            EC = 87*[.005]
        else:
            EC = EC.split(";")
            EC = EC[0]
            EC = EC.split(".")
            #some of the 4th level EC numbers are unknown and notated as -
            EC = [".005" if ("n" in x or "-" in x) else x for x in EC] 
            EC = [float(x) for x in EC]
            EC = format_data(EC)
        EC_list.append(EC)
        seq = training_data.iloc[x,6]
        pI_list = []
        for y in range(len(seq)):
            pI_list.append(pI_normalized[seq[y]])
        if len(seq) < lim:
            pI_list = pI_list + (lim-len(seq))*[0]
        pI_list.insert(0, training_data.iloc[x,5]/1000)
        seq_list.append(pI_list)
       
    df = pd.DataFrame({"Uniprot":entries, "EC": EC_list, "Values":seq_list})
    return(df)

def format_data(EC):
    #This function takes a list of four EC numbers, and turns it into an 87 element binary list.
    #the first element is 1 for enzyme, 0 for not, then the next seven are 0 or 1 for each enzyme class
    #and the next 79 are enzyme subclass. It essentially throws out the 3rd and 4th number
    #the first number is a sigmoid, the next 7 (enzyme classification level 1) are a softmax, the next 79 (enzyme classification level 2) are another softmax
    if EC[0] < 1:
        return 87*[.005]
    else:
        output = 87*[.005]
        output[0] = .995
        output[round(EC[0])] = .995
        
        #Some of the enzyme classes have codes that skip numbers to 98 or 99 for their subclass
        #the embedded if statements make it so that all subclasses are sequentially numbered
        if EC[0] == 1:
            if EC[1] > 23:
                EC[1] =  EC[1] - 73
            output[round(EC[1]+7)] = .995
        elif EC[0] == 2:
            output[round(EC[1]+33)] = .995
        elif EC[0] == 3:
            output[round(EC[1]+43)] = .995
        elif EC[0] == 4:
            if EC[1] > 10:
                EC[1] = EC[1] - 89
            output[round(EC[1]+56)] = .995
        elif EC[0] == 5:
            if EC[1] > 7:
                EC[1] = 7
            output[round(EC[1]+66)] = .995
        elif EC[0] == 6:
            output[round(EC[1]+73)] = .995
        elif EC[0] == 7:
            output[round(EC[1]+80)] = .995
        
        return output
    


def initialize_parameters(input_size = 251, hidden_size = [100, 20], output_size = 87, initial = .1):
    #creates a model. The layer number is the key, the actual values are a numpy array. Weight matrices are whole number keys, bias vectors are multiples of .5
    np.random.seed(int((time.time())*10000)%10000)
    parameters = {}
    for x in range(1,hidden_size[1]+1):
        if x == 1:
            W = np.random.randn(hidden_size[0], input_size)
            b = np.zeros((hidden_size[0], 1))
        elif x == hidden_size[1]:
            W = np.random.randn(output_size, hidden_size[0])
            b = np.zeros((output_size, 1))
        else:
            W = np.random.randn(hidden_size[0], hidden_size[0])
            b = np.zeros((hidden_size[0], 1))
        parameters[x] = W*initial
        parameters[(x+.5)] = b*initial
    return parameters

def sigmoid(x):
    if isinstance(x, np.ndarray):
        x[x > 12] = 12
        #prevent overflows
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# relu activation function
def relu(Z):
    return np.maximum(Z,.01*Z)

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


def softmax(vec):
    e = 2.71828183
    final = [e**x for x in vec]
    total = sum(final)
    final = [x/total for x in final]
    return final

def softmax_derivative(vec):
    #should create a numpy array where each row is the input vector, each column in the softmax output vector
    # and each x,y value corresponds to the derivative of each softmax output with respect to each input
    answer = np.zeros((len(vec), len(vec)))
    for x in range(len(vec)):
        for y in range(len(vec)):
            if x == y:
                answer[x,y] = vec[x]*(1-vec[x])
            else:
                answer[x,y] = (-1)*vec[x]*vec[y]
    return(answer)

# forward propagation
def forward_propagation(X, parameters):
    data = np.asarray(X)
    data = data.reshape(-1,1)
    cache = {}
    count = 1
    temp = 0
    for m in parameters:
        if m == 1:
            temp = np.dot(parameters[1], data)
        elif (int(m) != m-.5):
            #executes if weights
            temp = np.dot(parameters[m], cache[count-1])
        else:
            #executes if bias vector
            temp = temp + parameters[m]
            cache[count-.5] = temp
            if int(m) == len(parameters)/2:
                #executes if this is the last bias addition
                output = [0]
                output[0] = sigmoid(temp[0])
                output= output + softmax(temp[1:8])+ softmax(temp[8::])
                output = [x[0] for x in output]
                cache[count] = np.array(output).reshape(1,-1)
            else:
                cache[count] = relu(temp)
                count += 1 
    return cache

def backward_propagation(parameters, cache, X, y):
    predicted = cache[(len(cache)/2)]
    dL_dy = 2*(predicted-y)
    #this next line reduces saturation by telling the model not to worry about correcting
    #when error is less than .05
    dL_dy[abs(dL_dy) < .045] = 0
    dL_dInput_sigmoid = np.multiply(dL_dy[0,0],sigmoid_derivative(cache[(len(cache)/2)-.5].T[0,0]))
    dL_dInput_sigmoid = np.array(dL_dInput_sigmoid).reshape(-1,1)
    dL_dInput_softmax1 = np.dot(dL_dy[0,1:8],softmax_derivative(predicted[0,1:8]))
    dL_dInput_softmax2 = np.dot(dL_dy[0,8::],softmax_derivative(predicted[0,8::]))
    dL_dInput_softmax1 = dL_dInput_softmax1.reshape(1,-1)
    dL_dInput_softmax2 = dL_dInput_softmax2.reshape(1,-1)
    dL_dInput = np.concatenate((dL_dInput_sigmoid, dL_dInput_softmax1, dL_dInput_softmax2), axis = None)
    dL_dInput = dL_dInput.reshape(1,-1)
    gradients = {int(len(cache)/2): np.dot(cache[int(len(cache)/2)-1], dL_dInput).T, (int(len(cache)/2)+.5):dL_dInput.T}
    for i in list(range(1,int(len(cache)/2)))[::-1]:
        temp = np.dot(gradients[i+1.5].T, parameters[i+1]).T
        gradients[i+.5] = np.multiply(temp,leaky_relu_derivative(cache[i-.5]))
        if i == 1:
            X2 = np.array(X)
            X2 = X2.reshape(-1,1)
            gradients[i] = np.dot(X2, gradients[i+.5].T).T
        else:
            gradients[i] = np.dot(cache[i-1], gradients[i+.5].T)
    return(gradients)
    

def update_parameters(parameters, gradients, learning_rate=.1):
    for x in range(1,int(len(parameters)/2)):
        parameters[x+.5] = parameters[x+.5] - gradients[x+.5]*learning_rate
        parameters[x] = parameters[x] - gradients[x]*learning_rate
    return parameters

def save_parameters(params, path):
    params = {str(x):params[x] for x in params}
    np.savez(path, **params)
    
def load_parameters(path):
    params = np.load(path)
    params = {float(x):params[x] for x in params}
    return(params)


#naming conventions:
#each set of weights and biases arising from a new parameters is a new model
#each additionally trained set of that model is a version

#test code
#25 in the excel sheet is an enzyme, 7 is not
#I made this block of code below to just test that model creation, forward propagation, and backpropagation work
#p = prepare_data("/home/mattc/Documents/Fun/Training_data_EC_predictor/proteins254.xlsx", lim=254)
#params = initialize_parameters(input_size = 255, initial = .1)

# cache = forward_propagation(p["Values"][25], params)
# print(cache[20][0,0:9])
# gradients = backward_propagation(params, cache, p["Values"][25], p["EC"][25])
# params = update_parameters(params, gradients, learning_rate=.2)

# cache = forward_propagation(p["Values"][7], params)
# print(cache[20][0,0:9])
# gradients = backward_propagation(params, cache, p["Values"][7], p["EC"][7])
# params = update_parameters(params, gradients, learning_rate=.2)


 
import matplotlib.pyplot as plt 
np.random.seed(int((time.time())*10000)%10000)



def generate_model(training, epochs = 20):
    #training is a pandas dataframe of training data, from prepare_data
    #a random number of nodes 100-200 in the hidden layers
    nodes = np.random.randint(250,350)
    #a random number of hidden layers, an even number from 18 to 50
    nlayers = 14+ 2*np.random.randint(1,15)
    #a random initialization between .0 and .1
    initf = np.random.random()*.08
    params = initialize_parameters(input_size = 255,initial = initf, hidden_size=[nodes, nlayers])
    for y in range(1, epochs):
        alpha = abs(.05-(.003*(y-1)))
        #learning rate slightly decreases with each epoch, starts at .05
        for x in range(len(training)):
            if (training["EC"][x][0] < .1):
                if (np.random.randint(1,11) > 7):
                    #This code makes it so that there is a 30% chance non-enzymes will be skipped in each loop,
                    #which adds weight to enzymes
                    continue
            cache = forward_propagation(training["Values"][x], params)
            gradients = backward_propagation(params, cache, training["Values"][x], training["EC"][x])
            params = update_parameters(params, gradients, learning_rate=alpha)
        #this code stops trying to train a NN if it converges to a single value
        temp = []
        for x in range(10):
            seq = list(np.random.randint(11,size=255))
            seq[0] = .254
            cache = forward_propagation(seq, params)
            temp.append(cache[nlayers][0,0])
        if (max(temp)-min(temp)) < .05:
            print("NN converges, starting new NN")
            return(0)
    return(params)
        
    
def test_model(training, params, show=True, modeln =1, version = 1, threshhold  = 1):
    nlayers = int(max(params.keys()))
    error = []
    EC_values = []
    enz_error = []
    for x in range(len(training)):
        cache = forward_propagation(training["Values"][x], params)
        error.append(abs(cache[nlayers][0,0]-training["EC"][x][0]))
        EC_values.append(training["EC"][x][0])
        if (p["EC"][x][0] > .1):
            enz_error.append(abs(cache[nlayers][0,0]-.995))
            #only plot last 500 of last training set
    #score is an arbitrary metric I made up, it is sum of mean enzyme error and all protein mean error
    score = np.mean(enz_error)+np.mean(error)
    if show==True:
        plt.plot(list(range(len(error)))[-500::], error[-500::], label = "Error") 
        plt.plot(list(range(len(EC_values)))[-500::], EC_values[-500::], 'o', label = "EC_value", markersize=2) 
        plt.legend() 
        plt.show()
        nodes = params[2].shape[1]
        print("Model: "+ str(modeln)+"." + str(version))
        print("number of layers is " +str(nlayers) + " and number of nodes is " +str(nodes))
        print("Mean, median error is: "+ str(np.mean(error)) + ", " + str(np.median(error)))
        print("Mean, median error for enzymes is: "+ str(np.mean(enz_error)) + ", " + str(np.median(enz_error)))
        print("Score is : " + str(score))
        if score < threshhold:
            file = open("/home/mattc/Documents/Fun/models.csv", 'a')
            file.write("\n" + str(modeln)+"." + str(version) +", "+str(nlayers)+", "+ str(nodes)+", "+str(np.mean(error))+", "+str(np.mean(enz_error))+", "+ str(score))
            file.close()
    return(score)

def train_further(training, params1, epochs = 5, alpha = 1):
    params = params1.copy()
    score  = test_model(training, params1, show=False)
    for y in range(1, epochs):
        for x in range(len(training)):
            n = 1
            if (training["EC"][x][0] < .1):
                if (np.random.randint(1,11) > 7):
                    #This code makes it so that there is a 30% chance non-enzymes will be skipped in each loop,
                    #which adds weight to enzymes
                    continue
            else: 
                n = 1.8
            cache = forward_propagation(training["Values"][x], params)
            gradients = backward_propagation(params, cache, training["Values"][x], training["EC"][x])
            params = update_parameters(params, gradients, learning_rate=.004*alpha*n)
        #this code stops trying to train a NN if it converges to a single value
        temp = []
        nlayers = int(max(params))
        for x in range(10):
            seq = list(np.random.randint(11,size=255))
            seq[0] = .254
            cache = forward_propagation(seq, params)
            temp.append(cache[nlayers][0,0])
        if (max(temp)-min(temp)) < .05:
	#this code stops trying to train a NN if it converges to a single value
            return(params1)
    if test_model(training, params, show=False) < score:
        return(params)
    else:
        return(params1)
    
# file = open("/home/mattc/Documents/Fun/models.csv", 'w')
# file.write("model number, number layers, number nodes, mean error, mean enzyme error, score")
# file.close()

import time
p = prepare_data("/home/mattc/Documents/Fun/Training_data_EC_predictor/proteins254.xlsx", lim=254)
counter = 1
while True:
    b = generate_model(p)
    if isinstance(b, dict):
        c= test_model(p, b, show=False)
        print("Score is " + str(c))
        time.sleep(30)
        if c <.8:
            c= test_model(p, b, show=True, modeln = counter)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(counter)+"v1")
            counter += 1
del c

p = prepare_data("/home/mattc/Documents/Fun/Training_data_EC_predictor/proteins254.xlsx", lim=254)
for x in range(1,33): 
    b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v1.npz")
    c= test_model(p, b, show=False)
    for y in range(0,4):
        b = train_further(p, b, alpha = .9**(y))
        #give processor a short rest
        time.sleep(30)
    c2 = test_model(p, b, show = False)
    if (c2 < c):
        print("trained a model further! new score is: " + str(c2))
        save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v2")
del c, c2, b, x, y
    
for x in range(1,33): 
    try:
        b = load_parameters("/home/mattc/Docume3nts/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v2.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,4):
            b = train_further(p, b, alpha = .9**(y))
            #give processor a short rest
            time.sleep(30)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 3)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v3")
    except:
        continue
del b,c,x, c2, y


for x in range(1,33): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v3.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,4):
            b = train_further(p, b, alpha = .9**(y))
            #give processor a short rest
            time.sleep(15)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 4)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v4")
    except:
        continue
del b,c,x, c2, y

for x in range(1,33): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v4.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,7):
            b = train_further(p, b, alpha = .9**(y))
            #give processor a short rest
            time.sleep(30)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 5)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v5")
    except:
        continue
del b,c,x, c2, y


for x in range(1,33): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v5.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,8):
            b = train_further(p, b, alpha = .9**(y-1))
            #give processor a short rest
            time.sleep(30)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 6)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v6")
    except:
        continue
del b,c,x, c2, y


for x in range(1,33): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v6.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,8):
            b = train_further(p, b, alpha = .9**(y-1))
            #give processor a short rest
            time.sleep(30)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 7)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v7")
    except:
        continue
del b,c,x, c2, y


for x in range(1,33): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v7.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,8):
            b = train_further(p, b, alpha = .9**(y-1))
            #give processor a short rest
            time.sleep(30)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 8)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v8")
    except:
        continue
del b,c,x, c2, y

for x in range(1,33): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v8.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,8):
            b = train_further(p, b, alpha = .9**(y-1))
            #give processor a short rest
            time.sleep(30)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 9)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v9")
    except:
        continue
del b,c,x, c2, y



counter = 70
while counter < 85:
    b = generate_model(p)
    if isinstance(b, dict):
        c= test_model(p, b, show=False)
        time.sleep(30)
        if c <.68:
            c= test_model(p, b, show=True, modeln = counter)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(counter)+"v1")
            counter += 1
del c, b, counter

for x in range(70,85): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v1.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,10):
            b = train_further(p, b, alpha = np.random.random()*.4)
            #give processor a short rest
            time.sleep(20)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 2)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v2")
        #give processor a short rest
        time.sleep(20)
    except:
        continue
del b,c,x, c2, y

for x in range(70,85): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v2.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,10):
            b = train_further(p, b, alpha = np.random.random()*.4)
            #give processor a short rest
            time.sleep(20)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 3)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v3")
        #give processor a short rest
        time.sleep(20)
    except:
        continue
del b,c,x, c2, y


for x in range(70,85): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v3.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,10):
            b = train_further(p, b, alpha = np.random.random()*.4)
            #give processor a short rest
            time.sleep(20)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 4)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v4")
        #give processor a short rest
        time.sleep(20)
    except:
        continue
del b,c,x, c2, y

for x in range(70,85): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v4.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,10):
            b = train_further(p, b, alpha = np.random.random()*.4)
            #give processor a short rest
            time.sleep(20)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 5)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v5")
        #give processor a short rest
        time.sleep(20)
    except:
        continue
del b,c,x, c2, y

for x in range(70,85): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v5.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,20):
            b = train_further(p, b, alpha = np.random.random()*.06)
            #give processor a short rest
            time.sleep(20)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 6)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v6")
        #give processor a short rest
        time.sleep(20)
    except:
        continue
del b,c,x, c2, y

def train_further2(training, params1, epochs = 5, alpha = 1):
    nlayers = int(max(params1.keys()))
    params = params1.copy()
    score  = test_model(training, params1, show=False)
    for y in range(1, epochs):
        for x in range(len(training)):
            if (training["EC"][x][0] < .1):
                if (np.random.randint(1,11) > 7):
                    #This code makes it so that there is a 30% chance non-enzymes will be skipped in each loop,
                    #which adds weight to enzymes
                    continue
            cache = forward_propagation(training["Values"][x], params)
            gradients = backward_propagation(params, cache, training["Values"][x], training["EC"][x])
            n = 5*abs(cache[nlayers][0,0]-training["EC"][x][0])
            params = update_parameters(params, gradients, learning_rate=.004*alpha*n)
        #this code stops trying to train a NN if it converges to a single value
        temp = []
        nlayers = int(max(params))
        for x in range(10):
            seq = list(np.random.randint(11,size=255))
            seq[0] = .254
            cache = forward_propagation(seq, params)
            temp.append(cache[nlayers][0,0])
        if (max(temp)-min(temp)) < .05:
            print("recursive! 1")
            return(params1)
    if test_model(training, params, show=False) < score:
        return(params)
    else:
        return(params1)
    
for x in range(74,76): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v6.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,200):
            b = train_further(p, b, alpha = np.random.random()*.04)
            #give processor a short rest
            time.sleep(20)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 7)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v7")
        #give processor a short rest
        time.sleep(20)
    except:
        continue
del b,c,x, c2, y

for x in range(74,76): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v7.npz")
        c= test_model(p, b, show=False) 
        for y in range(0,200):
            b = train_further2(p, b, alpha = np.random.random()*.05)
            #give processor a short rest
            time.sleep(20)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 8)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v8")
        #give processor a short rest
        time.sleep(20)
    except:
        continue
del b,c,x, c2, y


counter = 85
while counter < 124:
    b = generate_model(p)
    if isinstance(b, dict):
        c= test_model(p, b, show=False)
        time.sleep(30)
        if c <.75:
            c= test_model(p, b, show=True, modeln = counter)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(counter)+"v1")
            counter += 1
del c, b, counter

for x in range(85,124): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v1.npz")
        c= test_model(p, b, show=False) 
        for y in range(1,11):
            b = train_further(p, b, alpha = max(abs(np.sin(y)), .1))
            #give processor a short rest
            time.sleep(15)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 2)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v2")
    except:
        continue
del b,c,x, c2, y


for x in range(85,124): 
    try:
        b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v2.npz")
        c= test_model(p, b, show=False) 
        for y in range(1,20):
            b = train_further(p, b, alpha = max(abs(np.sin(y)), .1))
            #give processor a short rest
            time.sleep(15)
        c2 = test_model(p, b, show = False)
        if (c2 < c):
            test_model(p, b, show = True, modeln = x, version = 3)
            save_parameters(b, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/model"+str(x)+"v3")
    except:
        continue
del b,c,x, c2, y

percent_enzyme = 0
for x in range(len(p)):
   if p["EC"][x][0] > .1:
       percent_enzyme += 1
percent_enzyme = percent_enzyme/len(p)
