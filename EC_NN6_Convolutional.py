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

this version is convolutional 
"""

import numpy as np
import pandas as pd
import time

np.random.seed(int((time.time())*10000)%10000)
filepath = "/home/mattc/Documents/Fun/Training_data_EC_predictor/"
Training_sets = {x:filepath+"mixed_TD2000_"+str(x)+".xlsx" for x in range(1,17)}


amino_acids_pI = {'D': 2.98, 'E': 3.08, 'C': 5.15, 'N': 5.43, 'T': 5.60, 'Y': 5.63, 'Q': 5.65, 'S': 5.70, 'M': 5.71, 'F': 5.76, 'W': 5.88, 'V': 6.00, 'I': 6.03, 'L': 6.04, 'G': 6.06, 'A': 6.11, 'P': 6.30, 'H': 7.64, 'K': 9.47, 'R': 10.76, 'Z':4.37, 'X':6.87, 'J':6.04, 'U':3.9, 'B':4.2} 
pI_normalized = {x:(10*(amino_acids_pI[x]-2.98)/(10.76-2.98))  for x in amino_acids_pI}
del amino_acids_pI

def prepare_data(path, lim = 2000, convolution = 200):
    #this function takes a file of training data, and turns the sequence into pI values and the EC value into a list of 1s and 0s
    training_data = pd.read_excel(path, sheet_name = "Sheet0")
    training_data = training_data[training_data["Length"] <= lim]
    entries = []
    EC_list = []
    seq_list = []
    #for Z, I used the mean pIs of Glutamic acid and Glutamine (0.286245353 + 0.525092937)/2 =  0.405669145
    #for X, I used the mean of the lowest pH, Aspartic acid, and the highest, Arginine, (0.276951673 + 1)/2 = 0.6384758365
    #for J, I used the means of Leucine and isoleucine, 0.560808922
    #for U, I used selenocysteine 3.9/10.76 = 0.3624535
    #for B, I used the mean of aspartic acid and asparagine, (0.276951673 + 0.50464684)/2 =0.3907992565
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
        if len(seq) % convolution != 0:
            pI_list = pI_list + (convolution-(len(seq) % convolution))*[-5]
        pI_list.insert(0, training_data.iloc[x,5]/1000)
        seq_list.append(pI_list)
       
    df = pd.DataFrame({"Uniprot":entries, "EC": EC_list, "Values":seq_list})
    return(df)

def find_percent_enzyme(training):
    #This function finds the percentage of proteins in a data set that are enzymes
    total = 0
    for x in range(len(training)):
        if training["EC"][x][0] == .995:
            total += 1
    return(total/len(training))

def format_data(EC):
    #This function takes a list of four EC numbers, and turns it into an 87 element binary list.
    #the first element is 1 for enzyme, 0 for not, then the next seven are 0 or 1 for each enzyme class
    #and the next 79 are enzyme subclass. It essentially throws out the 3rd and 4th number
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
    


def initialize_parameters(input_size = 200, pooling_size = [100, 12], conv_size = [50, 8], output_size = 87, initial = .1):
   """
    

    Parameters
    ----------
    input_size : TYPE, optional
        length of input layer for each convolutional window. The default is 125.
    pooling_size : TYPE, optional
        number of nodes in each pooling layer and the number of pooling layers. The default is [100, 12].
    conv_size : TYPE, optional
        number of nodes in the convolutional window and the number of layers. The default is [50, 8].
    output_size : TYPE, optional
        Size of output vector. The default is 87.
    initial : TYPE, optional
        Initialization constant for weights. The default is .1.

    Returns
    -------
    parameters : TYPE
        DESCRIPTION.

    """
   np.random.seed(int((time.time())*10000)%10000)
   parameters = {}
   parameters["hyperparameters"] = {"Model number":None, "Version":1, "Number of convolution layers":conv_size[1], "Number of convolution nodes":conv_size[0], 
                                    "Number of pooling layers":pooling_size[1], "Number of pooling nodes":pooling_size[0], "Input convolution size":input_size,
                                    "Initialization factor":initial, "Mean error": None, "Mean enzyme error": None, "Score": None, "Epochs trained":0}

   if 2000 % input_size != 0:
       raise Exception("Because this is a weight sharing NN, the input size must be an even number multiple of the convolution network size") 
        
   for x in range(1,(conv_size[1])+1):
        
            
        if x == 1:
            W = np.random.randn(conv_size[0], input_size)
            b = np.zeros((conv_size[0], 1))
        elif x == conv_size[1]:
            W = np.random.randn(pooling_size[0]-1, conv_size[0])
            b = np.zeros((pooling_size[0]-1, 1))
        else:
            W = np.random.randn(conv_size[0],conv_size[0])
            b = np.zeros((conv_size[0], 1))
        parameters[x] = W*initial
        parameters[(x+.5)] = b*initial
        
   for x in range((conv_size[1])+1,conv_size[1] +pooling_size[1]+1):
        if x == (conv_size[1]+pooling_size[1]):
            W = np.random.randn(output_size, pooling_size[0])
            b = np.zeros((output_size, 1))
        else:
            W = np.random.randn(pooling_size[0], pooling_size[0])
            b = np.zeros((pooling_size[0], 1))
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

def ceil(x):
    return int(x+1) - int(1-(x-int(x)))
# forward propagation
def forward_propagation(X, parameters):
    cache = {}
    pooling_size = parameters["hyperparameters"]["Number of pooling nodes"]
    conv_size = parameters["hyperparameters"]["Number of convolution layers"]
    input_size = parameters["hyperparameters"]["Input convolution size"]
    X2 = X[1::].copy()
    n_convs = 2*ceil(len(X2)/input_size)-1
    #temp2 is input for pooling layer
    temp2 = np.zeros((1,pooling_size-1))
    for i in range(1, n_convs+1):
        subcache = {}
        index1 = int(((i-1)*input_size*.5))
        index2 = int((((i-1)*input_size*.5) + input_size))
        X_fragment = X2[index1:index2]
        del index1, index2
        X_fragment = np.asarray(X_fragment)
        X_fragment = X_fragment.reshape(-1,1)
        count = 1
        temp = 0
        for m in parameters:
            if isinstance(m, str):
                continue
            elif m > (conv_size+.5):
                break
            else:
                if m == 1:
                    temp = np.dot(parameters[1], X_fragment)
                elif (int(m) != m-.5):
                    #executes if weights
                    temp = np.dot(parameters[m], subcache[count-1])
                else:
                    #executes if bias vector
                    temp = temp + parameters[m]
                    subcache[count-.5] = temp
                    subcache[count] = relu(temp)
                    if int(m) == conv_size:
                        #executes if this is the last bias addition of each convolutional subnetwork
                        temp2 = np.add(temp2, subcache[count].T)
                    count += 1 
        cache["Subcache" + str(i)] = subcache
    temp2 = np.insert(temp2, 0, X[0])
    temp2 = temp2.reshape(-1,1)
    count = conv_size+1
    for m in parameters:
        if (isinstance(m, str)) or (m < (conv_size+1)):
            continue
        if m == (conv_size+1):
            temp = np.dot(parameters[m], temp2)
        elif (int(m) != m-.5):
            #executes if weights
            temp = np.dot(parameters[m], cache[count-1])
        else:
            #executes if bias vector
            temp = temp + parameters[m]
            cache[count-.5] = temp
            if int(m) == (parameters["hyperparameters"]["Number of pooling layers"] + conv_size):
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
    keys = list(parameters.keys())
    keys = [x for x in keys if isinstance(x, int)]
    keys.reverse()
    
    subcaches = list(cache.keys())
    subcaches = [x for x in subcaches if isinstance(x, str)]
    
    predicted = cache[keys[0]]
    dL_dy = 2*(predicted-y)
    #this next line reduces saturation by telling the model not to worry about correcting
    #when error is less than .05
    dL_dy[abs(dL_dy) < .045] = 0
    dL_dInput_sigmoid = np.multiply(dL_dy[0,0],sigmoid_derivative(cache[keys[0]-.5].T[0,0]))
    dL_dInput_sigmoid = np.array(dL_dInput_sigmoid).reshape(-1,1)
    dL_dInput_softmax1 = np.dot(dL_dy[0,1:8],softmax_derivative(predicted[0,1:8]))
    dL_dInput_softmax2 = np.dot(dL_dy[0,8::],softmax_derivative(predicted[0,8::]))
    dL_dInput_softmax1 = dL_dInput_softmax1.reshape(1,-1)
    dL_dInput_softmax2 = dL_dInput_softmax2.reshape(1,-1)
    dL_dInput = np.concatenate((dL_dInput_sigmoid, dL_dInput_softmax1, dL_dInput_softmax2), axis = None)
    dL_dInput = dL_dInput.reshape(1,-1)
    gradients = {keys[0]: np.dot(cache[keys[1]], dL_dInput).T, keys[0]+.5:dL_dInput.T}
    keys = keys[1:(parameters["hyperparameters"]["Number of pooling layers"]-1)]
    for i in keys:
        temp = np.dot(gradients[i+1.5].T, parameters[i+1]).T
        gradients[i+.5] = np.multiply(temp,leaky_relu_derivative(cache[i-.5]))
        gradients[i] = np.dot(cache[i-1], gradients[i+.5].T)
    n = parameters["hyperparameters"]["Number of convolution layers"]
    gradients[n+1.5] = np.multiply(np.dot(gradients[i+2.5].T, parameters[i+2]).T, leaky_relu_derivative(cache[n+.5]))
    pool_input = np.zeros(np.shape(cache["Subcache1"][n]))
    for i in subcaches:
        pool_input = np.add(pool_input, cache[i][n])
    pool_input = np.insert(pool_input, 0, X[0])
    pool_input = pool_input.reshape(-1,1)
    gradients[n+1] = np.dot(pool_input, gradients[n+1.5].T)
    del pool_input
    X2 = X[1::]
    for j in range(len(subcaches)):
        gradients[subcaches[j]] = {}
        index1 = int((j*parameters["hyperparameters"]["Input convolution size"]*.5))
        index2 = int(((j*parameters["hyperparameters"]["Input convolution size"]*.5) + parameters["hyperparameters"]["Input convolution size"]))
        X_fragment = X2[index1:index2]
        X_fragment = np.array(X_fragment)
        X_fragment = X_fragment.reshape(-1,1)
        del index1, index2
        j = subcaches[j]
        for i in list(range(1,n+1))[::-1]:
            if i == n:
                temp = np.dot(gradients[i+1.5].T, parameters[i+1]).T
                gradients[j][i+.5] = np.multiply(temp[1::],leaky_relu_derivative(cache[j][i-.5]))
                gradients[j][i] = np.dot(cache[j][i-1], gradients[j][i+.5].T).T
            else:
               temp = np.dot(gradients[j][i+1.5].T, parameters[i+1]).T
               gradients[j][i+.5] = np.multiply(temp,leaky_relu_derivative(cache[j][i-.5]))
               if i == 1:
                   gradients[j][i] = np.dot(X_fragment, gradients[j][i+.5].T).T
               else:
                   gradients[j][i] = np.dot(cache[j][i-1], gradients[j][i+.5].T)
    del n
    for x in list(gradients["Subcache1"].keys()):
        gradients[x] = np.zeros(np.shape(gradients["Subcache1"][x]))
        for j in subcaches:
            gradients[x] = np.add(gradients[x], gradients[j][x])
    for j in subcaches:
        del gradients[j]
    return(gradients)
    

def update_parameters(parameters, gradients, learning_rate=.1):
    keys = list(parameters.keys())
    keys = [x for x in keys if isinstance(x, int)]
    for x in keys:
        parameters[x+.5] = parameters[x+.5] - gradients[x+.5]*learning_rate
        parameters[x] = parameters[x] - gradients[x]*learning_rate
    return parameters

def save_parameters(params, path):
    params = {str(x):params[x] for x in params}
    np.savez(path, **params)
    
def load_parameters(path):
    params = np.load(path, allow_pickle=True)
    params = {str(x):params[x] for x in params}
    params2 = {}
    for x in params:
        if x[0].isdigit():
            if "." in x:
                params2[float(x)] = params[x]
            else:
                params2[int(x)] = params[x]
        else:
            params2[x] = params[x].tolist()
    return(params2)

def print_hyperparameters(params, path="/home/mattc/Documents/Fun/models.csv"):
    try:
        file = open(path, 'r')
        file.close()
        file = open(path, 'a')
        text = list(params["hyperparameters"].values())
        text = ["NA" if x is None else str(x) for x in text]
        text = ", ".join(text)
        text = "\n" + text
        file.write(text)
        file.close()
    except:
        file = open(path, 'w')
        text = list(params["hyperparameters"].values())
        text = ["NA" if x is None else str(x) for x in text]
        text = ", ".join(text)
        file.write(", ".join(params["hyperparameters"]) + "\n" + text)
        file.close()




#naming conventions:
#each set of weights and biases arising from a new parameters is a new model
#each additionally trained set of that model is a version

#test code
#p = prepare_data(Training_sets[1], convolution = 200)
#params = initialize_parameters(input_size = 200, pooling_size = [100, 12], conv_size = [50, 8], initial = .1)
# a = .3*abs(np.sin(int((time.time())*10000)%10000))
# cache = forward_propagation(p["Values"][3], params)
# print(cache[20][0,0:9])
# gradients = backward_propagation(params, cache, p["Values"][3], p["EC"][3])
# params = update_parameters(params, gradients, learning_rate=a)

# cache = forward_propagation(p["Values"][2], params)
# print(cache[20][0, 0:9])
# gradients = backward_propagation(params, cache, p["Values"][2], p["EC"][2])
# params = update_parameters(params, gradients, learning_rate=a)

# cache = forward_propagation(p["Values"][12], params)
# print(cache[20][0, 0:9])
# gradients = backward_propagation(params, cache, p["Values"][12], p["EC"][12])
# params = update_parameters(params, gradients, learning_rate=a)

# cache = forward_propagation(p["Values"][13], params)
# print(cache[20][0, 0:9])
# gradients = backward_propagation(params, cache, p["Values"][13], p["EC"][13])
# params = update_parameters(params, gradients, learning_rate=a)
# del a

# save_parameters(params, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/modeltestsave")
# b = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/modeltestsave.npz")

#3 is 228 aa long, 2 is 113 aa long, 12 is 1052 aa long, 13 is 297 aa long

# a = .3*abs(np.sin(int((time.time())*10000)%10000))
# cache = forward_propagation(p["Values"][3], b)
# print(cache[20][0,0:9])
# gradients = backward_propagation(b, cache, p["Values"][3], p["EC"][3])
# b = update_parameters(b, gradients, learning_rate=a)
# del a

import matplotlib.pyplot as plt 
np.random.seed(int((time.time())*10000)%10000)



def generate_model(L =200, max_conv = 12, max_pool = 18):
    np.random.seed(int((time.time())*10000)%10000)
    #a random number of nodes 100-200 in the convolutional layers
    nodes_c = np.random.randint(100,250)
    #a random number of nodes 100-200 in the pooling layers
    nodes_p = np.random.randint(200,350)
    #a random number of convolutional layers, an even number from 18 to 50
    nlayers_c = 4+ 2*np.random.randint(1,(((max_conv-4)/2)+1))
    #a random number of pooling layers, an even number from 18 to 50
    nlayers_p = 8+ 2*np.random.randint(1,(((max_pool-8)/2)+1))
    #a random initialization between .0 and .1
    initf = np.random.random()*.16
    params = initialize_parameters(input_size = L, pooling_size = [nodes_p, nlayers_p], conv_size = [nodes_c, nlayers_c], initial = initf)
    return(params)
        
    
def test_model(training, params, show=True):
    nlayers_c = params["hyperparameters"]["Number of convolution layers"]
    nlayers_p = params["hyperparameters"]["Number of pooling layers"]
    nlayers = nlayers_c + nlayers_p
    error = []
    EC_values = []
    enz_error = []
    #scores based on first 1000 genes, if they exist
    for x in range(min(len(training),1000)):
        cache = forward_propagation(training["Values"][x], params)
        error.append(abs(cache[nlayers][0,0]-training["EC"][x][0]))
        EC_values.append(training["EC"][x][0])
        if (training["EC"][x][0] > .1):
            enz_error.append(abs(cache[nlayers][0,0]-.995))
            #only plot last 500 of last training set
    score = np.mean(enz_error)+np.mean(error)
    if show==True:
        plt.plot(list(range(len(error)))[-500::], error[-500::], label = "Error") 
        plt.plot(list(range(len(EC_values)))[-500::], EC_values[-500::], 'o', label = "EC_value", markersize=2) 
        plt.legend() 
        plt.show()
        print("number of convolutional layers is " +str(nlayers_c) + " and number of convolutional nodes is " +str(params["hyperparameters"]["Number of convolution nodes"]))
        print("number of pooling layers is " +str(nlayers_p) + " and number of pooling nodes is " +str(params["hyperparameters"]["Number of pooling nodes"]))
        print("Mean, median error is: "+ str(np.mean(error)) + ", " + str(np.median(error)))
        print("Mean, median error for enzymes is: "+ str(np.mean(enz_error)) + ", " + str(np.median(enz_error)))
        print("Score is : " + str(score))
    return(score, np.mean(enz_error))

def train_further(training, params1, alpha = 1, adjust = .25, retest = False, show_graph=False):
    params = params1.copy()
    L = params1["hyperparameters"]["Input convolution size"]
    nlayers = params1["hyperparameters"]["Number of pooling layers"] + params1["hyperparameters"]["Number of convolution layers"]
    alpha2 = alpha*.0003
    for x in range(len(training)):
        if (training["EC"][x][0] < .1):
            if (np.random.random() > (adjust/(1-adjust))):
                #adjust is the percentage of enzymes in the training data
                #This code makes it so that enough non-enzymes will be skipped so that the number
                #of enzymes and non-enzymes trained on is equal
                continue
        cache = forward_propagation(training["Values"][x], params)
        gradients = backward_propagation(params, cache, training["Values"][x], training["EC"][x])
        params = update_parameters(params, gradients, learning_rate=alpha2)
    #this code stops trying to train a NN if it converges to a single value. 
    temp = []
    for x in range(10):
        seq = list(np.random.randint(11,size=np.random.randint(100,1000)))
        seq[0] = len(seq)/1000
        if len(seq)%L != 1:
            seq = seq + (L-(len(seq)%L)+1)*[-5]
        cache = forward_propagation(seq, params)
        temp.append(cache[nlayers][0,0])
    if (max(temp)-min(temp)) < .05:
        return(params1)
    if any(np.isnan(temp)):
        return(params1)
    if retest == True:
        score  = params1["hyperparameters"]["Score"]
        scores = test_model(training, params, show=show_graph)
        if scores[0] < score:
            params["hyperparameters"]["Score"] = scores[0]
            params["hyperparameters"]["Mean enzyme error"] = scores[1]
            params["hyperparameters"]["Mean error"] = scores[0]- scores[1]
            params["hyperparameters"]["Version"] = params1["hyperparameters"]["Version"] +1
            return(params)
        else:
            return(params1)
    else:
        return(params)

from statistics import mode

def test_converge(params, cutoff = .07):
    #returns false if a model converges
    L = params["hyperparameters"]["Input convolution size"]
    nlayers = params["hyperparameters"]["Number of pooling layers"] + params["hyperparameters"]["Number of convolution layers"]
    temp = []
    for x in range(20):
        seq = list(np.random.randint(11,size=np.random.randint(100,1000)))
        seq[0] = len(seq)/1000
        if len(seq)%L != 1:
            seq = seq + (L-(len(seq)%L)+1)*[-5]
        cache = forward_propagation(seq, params)
        temp.append(cache[nlayers][0,0])
    if (max(temp)-min(temp)) < cutoff:
        return(False)
    elif(any(x==0 for x in temp)):
        #because I'm using a sigmoid function, its impossible for me to get a 0 output naturally
        #the only way a 0 can occur is if there is a runtime overflow
        return(False)
    elif(temp.count(mode(temp)) >=3):
        return(False)
    elif any(np.isnan(temp)):
        return(False)
    else:
        return(True)
    
def check_overflow(params):
    #returns false if a model converges
    L = params["hyperparameters"]["Input convolution size"]
    nlayers = params["hyperparameters"]["Number of pooling layers"] + params["hyperparameters"]["Number of convolution layers"]
    temp = []
    for x in range(20):
        seq = list(np.random.randint(11,size=np.random.randint(100,1000)))
        seq[0] = len(seq)/1000
        if len(seq)%L != 1:
            seq = seq + (L-(len(seq)%L)+1)*[-5]
        cache = forward_propagation(seq, params)
        temp.append(cache[nlayers][0,0])
    if(any(x==0 for x in temp)):
        #because I'm using a sigmoid function, its impossible for me to get a 0 output naturally
        #the only way a 0 can occur is if there is a runtime overflow
        return(False)
    elif any(np.isnan(temp)):
        return(False)
    else:
        return(True)
    
x = 1
tries = 0
p = prepare_data(Training_sets[1], convolution = 100)
ratio = find_percent_enzyme(p)
p2 = prepare_data(Training_sets[2], convolution = 100)
ratio2 = find_percent_enzyme(p2)
p3 = prepare_data(Training_sets[3], convolution = 100)
ratio3 = find_percent_enzyme(p3)
p4 = prepare_data(Training_sets[4], convolution = 100)
ratio4 = find_percent_enzyme(p4)
while x < 200:
    m = generate_model(L=100)
    m = train_further(p, m, adjust =ratio, alpha=.5)
    if check_overflow(m) == False:
        print("overflow/converged")
        continue
    m = train_further(p2, m, adjust =ratio2, alpha=.5)
    if check_overflow(m) == False:
        print("overflow/converged")
        continue
    time.sleep(20)
    m = train_further(p3, m, adjust =ratio3, alpha=.5)
    if test_converge(m) == False:
        tries+=1
        print("Model converged")
        print("Initialization factor: "+str(m["hyperparameters"]["Initialization factor"]))
        print("Unsuccessful tries:" +str(tries))
        continue    
    m = train_further(p4, m, adjust =ratio4, alpha=.5)
    time.sleep(10)
    if test_converge(m) == False:
        tries+=1
        print("Model converged")
        print("Initialization factor: "+str(m["hyperparameters"]["Initialization factor"]))
        print("Unsuccessful tries:" +str(tries))
        continue
    scores2 = test_model(p, m, show=True)
    m["hyperparameters"]["Score"] = scores2[0]
    m["hyperparameters"]["Epochs trained"] = .25
    m["hyperparameters"]["Model number"] = x
    m["hyperparameters"]["Mean enzyme error"] = scores2[1]
    m["hyperparameters"]["Mean error"] = scores2[0]- scores2[1]
    save_parameters(m, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/cmodel"+str(x))
    print_hyperparameters(m)
    x += 1
del scores2, x, tries, p, ratio, p2, ratio2, p3, ratio3, p4, ratio4, m   


p = prepare_data(Training_sets[1], convolution = 100)
for x in range(1,19):
    try:
        converged = False
        m = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/cmodel"+str(x)+".npz")
        for y in range(5,17):
            train = prepare_data(Training_sets[y], convolution = 100)
            m = train_further(train, m, adjust =find_percent_enzyme(train), alpha=.5)
            time.sleep(20)
            if test_converge(m) == False:
                converged = True
                break
        if converged == True:
            print("converged")
            continue
        for y in range(1,17):
            train = prepare_data(Training_sets[y], convolution = 100)
            m = train_further(train, m, adjust =find_percent_enzyme(train), alpha=.5)
            time.sleep(20)
            if test_converge(m) == False:
                converged = True
                break
        if converged == True:
            print("converged")
            continue
        scores = test_model(p, m, show=True)
        m["hyperparameters"]["Score"] = scores[0]
        m["hyperparameters"]["Epochs trained"] = 2
        m["hyperparameters"]["Version"] = 2
        m["hyperparameters"]["Mean enzyme error"] = scores[1]
        m["hyperparameters"]["Mean error"] = scores[0]- scores[1]
        save_parameters(m, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/cmodel"+str(x)+"v2")
        print_hyperparameters(m)
    except:
        continue
del scores, y,x,m, p, train, converged

p = prepare_data(Training_sets[1], convolution = 100)
for x in range(1,19):
    try:
        converged = False
        m = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/cmodel"+str(x)+"v2.npz")
        for y in range(1,17):
            train = prepare_data(Training_sets[y], convolution = 100)
            m = train_further(train, m, adjust =find_percent_enzyme(train), alpha=.5*abs(np.sin(int((time.time())*10000)%10000)))
            time.sleep(20)
            if test_converge(m) == False:
                converged = True
                break
        if converged == True:
            print("converged")
            continue
        for y in range(1,17):
            train = prepare_data(Training_sets[y], convolution = 100)
            m = train_further(train, m, adjust =find_percent_enzyme(train), alpha=.5*abs(np.sin(int((time.time())*10000)%10000)))
            time.sleep(20)
            if test_converge(m) == False:
                converged = True
                break
        if converged == True:
            print("converged")
            continue
        scores = test_model(p, m, show=True)
        m["hyperparameters"]["Score"] = scores[0]
        m["hyperparameters"]["Epochs trained"] = 4
        m["hyperparameters"]["Version"] = 3
        m["hyperparameters"]["Mean enzyme error"] = scores[1]
        m["hyperparameters"]["Mean error"] = scores[0]- scores[1]
        save_parameters(m, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/cmodel"+str(x)+"v3")
        print_hyperparameters(m)
    except:
        continue
del scores, y,x,m, p, train, converged


p = prepare_data(Training_sets[1], convolution = 100)
for x in range(1,19):
    try:
        converged = False
        m = load_parameters("/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/cmodel"+str(x)+"v3.npz")
        for y in range(1,17):
            train = prepare_data(Training_sets[y], convolution = 100)
            m = train_further(train, m, adjust =find_percent_enzyme(train), alpha=.5*abs(np.sin(int((time.time())*10000)%10000)))
            time.sleep(20)
            if test_converge(m) == False:
                converged = True
                break
        if converged == True:
            print("converged")
            continue
        for y in range(1,17):
            train = prepare_data(Training_sets[y], convolution = 100)
            m = train_further(train, m, adjust =find_percent_enzyme(train), alpha=.5*abs(np.sin(int((time.time())*10000)%10000)))
            time.sleep(20)
            if test_converge(m) == False:
                converged = True
                break
        if converged == True:
            print("converged")
            continue
        scores = test_model(p, m, show=True)
        m["hyperparameters"]["Score"] = scores[0]
        m["hyperparameters"]["Epochs trained"] = 6
        m["hyperparameters"]["Version"] = 4
        m["hyperparameters"]["Mean enzyme error"] = scores[1]
        m["hyperparameters"]["Mean error"] = scores[0]- scores[1]
        save_parameters(m, "/home/mattc/Documents/Fun/Training_data_EC_predictor/Models/cmodel"+str(x)+"v4")
        print_hyperparameters(m)
    except:
        continue
del scores, y,x,m, p, train, converged
