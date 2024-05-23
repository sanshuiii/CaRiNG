import os
import glob
import tqdm
import torch
import scipy
import random
import numpy as np
from torch import nn
from torch.nn import init
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ortho_group
import pdb

import sys

# Setting the seed
seed = int(sys.argv[1])
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# If you are using any other libraries that use randomness, set the seed for those too

# Enforce determinism in the PyTorch (may slow down the computations)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

root_dir = "../caring/datasets"

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

def sigmoidAct(x):
    return 1. / (1 + np.exp(-1 * x))

def generateUniformMat(Ncomp, condT):
    
    
    
    
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())


    return A



def stationary_local_uninvertable_z_tulda():
    ### inilization of parameters
    lags = 1
    lags_indicator = 0
    Nlayer = 3
    length = 9
    condList = []
    negSlope = 0.2
    latent_size = 3
    obs_size = 2
    transitions = []
    noise_scale = 0.1
    batch_size = 40000
    Niter4condThresh = 1e4

    ### create data path
    #path = os.path.join(root_dir, "z_tulda_indicator_value_stationary_local_uninvertable_%dlags_%dlen_%dvariables_%dindicatorlen" % (lags, length,latent_size,lags_indicator))
    
    path = os.path.join(root_dir, "seed%d_%dd%d" %(seed, latent_size, obs_size))
    os.makedirs(path, exist_ok=True)
    
    ### generate transition matrix
    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    ### select transition matrix with condition number lager than top 25% percentile
    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        e = 0.99
        # B = np.array([
        #     [0,e,0,0,0,0],
        #     [0,0,e,0,0,0],
        #     [0,0,0,e,0,0],
        #     [0,0,0,0,e,0],
        #     [0,0,0,0,0,e],
        #     [e,0,0,0,0,0],
        # ])
        B = np.array([
            [0,e,0],
            [0,0,e],
            [e,0,0],
        ])
        transitions.append(B)
    
    ### generate mixing matrix for each layer with orthogonal matrix
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        if obs_size > 1:
            A = ortho_group.rvs(obs_size)  # generateUniformMat(Ncomp, condThresh)
        else:
            A = np.array([[1.0]])
        mixingList.append(A)


    ### initialize the first lags latent variable
    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
    yl_ind = np.copy(y_l)

    ### pass the latent variable through the maxing matrix for the first lags
    yt = []; xt = []; xt_orig =[]
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l[:,:,:obs_size])

    # pdb.set_trace()
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
        xt_orig.append(x_l[:,i,:])
        
    # Mixing function
    for i in range(length):
        if i < lags_indicator-lags:
            assert(0)
        else:
            # Transition function
            y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
            y_t = y_t * np.mean(y_l, axis=1)
            # y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()

            for l in range(lags):
                y_t += np.dot(y_l[:,l,:], transitions[l])
            yt.append(y_t)
            # Mixing function
            
            # import pdb; pdb.set_trace()
            # pdb.set_trace()
            mixedDat = np.copy(y_t[:, :obs_size])
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt_orig.append(x_t)
            
            x_t_tulda = np.copy(mixedDat)
            # import ipdb; ipdb.set_trace()
            # x_t_tulda[:,-1] = x_t[:,-1] + y_t[:, -1]  #  add the indicator value to the last dimension of x_t, It is uninvertable
            # x_t_tulda[:,-1] = np.exp(x_t[:,-1])+lambda_ * indicator_value.squeeze()  #  add the indicator value to the last dimension of x_t, It is uninvertable
            # x_t_tulda[:,-1] = x_t[:,-1]*indicators #  time the indicator to the last dimension of x_t, It is uninvertable
            xt.append(x_t_tulda)
            # import pdb; pdb.set_trace()
            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
            yl_ind = np.concatenate((yl_ind, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
    
    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); xt_orig = np.array(xt_orig).transpose(1,0,2)
    
    import ipdb; ipdb.set_trace()
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)

            
if __name__ == "__main__":
    stationary_local_uninvertable_z_tulda()