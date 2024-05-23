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
    lags_indicator = 2
    Nlayer = 3
    length = 9
    condList = []
    negSlope = 0.2
    latent_size = 3
    transitions = []
    indicator_matrix=[]
    noise_scale = 0.1
    batch_size = 40000
    Niter4condThresh = 1e4

    ### create data path
    #path = os.path.join(root_dir, "z_tulda_indicator_value_stationary_local_uninvertable_%dlags_%dlen_%dvariables_%dindicatorlen" % (lags, length,latent_size,lags_indicator))
    
    lambda_ = 1
    path = os.path.join(root_dir, "seed%_mixinglag" % seed)
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
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()
    
    ### generate mixing matrix for each layer with orthogonal matrix
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

    ### initialize the first lags latent variable
    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
    yl_ind = np.copy(y_l)

    ### pass the latent variable through the maxing matrix for the first lags
    yt = []; xt = []; xt_orig =[]
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
        xt_orig.append(x_l[:,i,:])

    ### generate the indicator matrix
    indicator_matrix.append(np.array([[0.1],[0.6],[0.3]]))
    indicator_matrix.append(np.array([[0.5],[0.2],[0.3]]))
    # for i in range(lags_indicator):
    #     C_matrix = np.random.uniform(0, 1, (latent_size, 1))
    #     indicator_matrix.append(C_matrix)
    # import ipdb; ipdb.set_trace()
        
    # Mixing function
    for i in range(length):
        if i < lags_indicator-lags:
            # Transition function
            # import ipdb; ipdb.set_trace()
            y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
            y_t = y_t * np.mean(y_l, axis=1)

            # y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()

            for l in range(lags):
                y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)
            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)
            xt_orig.append(x_t)
            #print(y_l.shape)
            yl_ind = np.concatenate((yl_ind, y_t[:,np.newaxis,:]),axis=1)
            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:] # the history information in lags used for current frame
            
            #print(y_l.shape,x_l.shape)
        else:
            # Transition function
            y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
            y_t = y_t * np.mean(y_l, axis=1)
            # y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()

            for l in range(lags):
                y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)
            # Mixing function
            indicator_value = np.zeros((batch_size,1))
            # print(i,y_l.shape,y_t.shape,lags_indicator,)
            for l in range(lags_indicator):
                indicator_value += np.dot(yl_ind[:,l,:], indicator_matrix[l])
            indicators = np.where(indicator_value > 0.1, 1, -1).squeeze()
            # import ipdb; ipdb.set_trace()
            
            # y_t_tulda = np.copy(y_t)
            # #print(y_t_tulda.shape,indicators.shape)
            # y_t_tulda[:,-1] = y_t_tulda[:,-1]+indicator_value.squeeze()
            
            # import pdb; pdb.set_trace()
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt_orig.append(x_t)
            
            x_t_tulda = np.copy(mixedDat)
            # import ipdb; ipdb.set_trace()
            x_t_tulda[:,-1] = x_t[:,-1]+lambda_ * indicator_value.squeeze()  #  add the indicator value to the last dimension of x_t, It is uninvertable
            # x_t_tulda[:,-1] = np.exp(x_t[:,-1])+lambda_ * indicator_value.squeeze()  #  add the indicator value to the last dimension of x_t, It is uninvertable
            # x_t_tulda[:,-1] = x_t[:,-1]*indicators #  time the indicator to the last dimension of x_t, It is uninvertable
            xt.append(x_t_tulda)
            # import pdb; pdb.set_trace()
            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
            yl_ind = np.concatenate((yl_ind, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
            # import ipdb; ipdb.set_trace()
    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); xt_orig = np.array(xt_orig).transpose(1,0,2)
    import ipdb; ipdb.set_trace()

    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)
    
    np.save(os.path.join(path,"IM"),np.array(indicator_matrix))

# def stationary_local_uninvertable_z_tulda(NClass=5):
#     ### inilization of parameters
#     lags = 1
#     lags_indicator = 2
#     Nlayer = 3
#     length = 9
#     condList = []
#     negSlope = 0.2
#     latent_size = 3
#     transitions = []
#     indicator_matrix=[]
#     noise_scale = 0.1
#     batch_size = 40000
#     Niter4condThresh = 1e4
    
#     ### create data path
#     #path = os.path.join(root_dir, "z_tulda_indicator_value_stationary_local_uninvertable_%dlags_%dlen_%dvariables_%dindicatorlen" % (lags, length,latent_size,lags_indicator))
    
#     path = os.path.join(root_dir, "indicator_Matrix_z_tulda")
    
#     os.makedirs(path, exist_ok=True)
    
#     ### generate transition matrix
#     for i in range(int(Niter4condThresh)):
#         # A = np.random.uniform(0,1, (Ncomp, Ncomp))
#         A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
#         for i in range(latent_size):
#             A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
#         condList.append(np.linalg.cond(A))
        
#     ### generate the indicator matrix
#     for i in range(lags_indicator):
#         C_matrix = np.random.uniform(0, 1, (latent_size, 1))
#         indicator_matrix.append(C_matrix)

#     ### select transition matrix with condition number lager than top 25% percentile
#     condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
#     for l in range(lags):
#         B = generateUniformMat(latent_size, condThresh)
#         transitions.append(B)
#     transitions.reverse()
    
#     ### generate mixing matrix for each layer with orthogonal matrix
#     mixingList = []
#     for l in range(Nlayer - 1):
#         # generate causal matrix first:
#         A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
#         mixingList.append(A)

#     ### initialize the first lags latent variable
#     y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
#     y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

#     ### pass the latent variable through the maxing matrix for the first lags
#     yt = []; xt = []
#     for i in range(lags):
#         yt.append(y_l[:,i,:])
#     mixedDat = np.copy(y_l)
#     for l in range(Nlayer - 1):
#         mixedDat = leaky_ReLU(mixedDat, negSlope)
#         mixedDat = np.dot(mixedDat, mixingList[l])
#     x_l = np.copy(mixedDat)
#     for i in range(lags):
#         xt.append(x_l[:,i,:])
        
#     # Mixing function
#     for i in range(length):
#         if i < lags_indicator-lags:
#             # Transition function
#             y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
#             for l in range(lags):
#                 y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
#             y_t = leaky_ReLU(y_t, negSlope)
#             yt.append(y_t)
#             # Mixing function
#             mixedDat = np.copy(y_t)
#             for l in range(Nlayer - 1):
#                 mixedDat = leaky_ReLU(mixedDat, negSlope)
#                 mixedDat = np.dot(mixedDat, mixingList[l])
#             x_t = np.copy(mixedDat)
#             xt.append(x_t)
#             #print(y_l.shape)
#             y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)
#             x_l = np.concatenate((x_l, x_t[:,np.newaxis,:]),axis=1)
#             #print(y_l.shape,x_l.shape)
#         else:
#             # Transition function
#             y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
#             for l in range(lags):
#                 y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
#             y_t = leaky_ReLU(y_t, negSlope)
#             yt.append(y_t)
#             # Mixing function
#             indicator_value = np.zeros((batch_size,1))
#             print(i,y_l.shape,y_t.shape,lags_indicator,)
#             for l in range(lags_indicator):
#                 indicator_value += np.dot(y_l[:,l,:], indicator_matrix[l])
#             indicators = np.where(indicator_value > 0, 1, -1).squeeze()
            
#             y_t_tulda = np.copy(y_t)
#             #print(y_t_tulda.shape,indicators.shape)
#             y_t_tulda[:,-1] = y_t_tulda[:,-1]+indicator_value.squeeze()
#             mixedDat = np.copy(y_t_tulda)
#             for l in range(Nlayer - 1):
#                 mixedDat = leaky_ReLU(mixedDat, negSlope)
#                 mixedDat = np.dot(mixedDat, mixingList[l])
#             x_t = np.copy(mixedDat)
#             xt.append(x_t)
#             y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
#             x_l = np.concatenate((x_l, x_t[:,np.newaxis,:]),axis=1)[:,1:,:]
            
#     # pdb.set_trace()
#     yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)

#     np.savez(os.path.join(path, "data"), 
#             yt = yt, 
#             xt = xt)

#     for l in range(lags):
#         B = transitions[l]
#         np.save(os.path.join(path, "W%d"%(lags-l)), B)
    
#     np.save(os.path.join(path,"IM"),np.array(indicator_matrix))
    
    
    
 

# def stationary_local_uninvertable(NClass=5):
#     ### inilization of parameters
#     lags = 1
#     lags_indicator = 2
#     Nlayer = 3
#     length = 9
#     condList = []
#     negSlope = 0.2
#     latent_size = 3
#     transitions = []
#     indicator_matrix=[]
#     noise_scale = 0.1
#     batch_size = 40000
#     Niter4condThresh = 1e4
    
#     ### create data path
#     path = os.path.join(root_dir, "indicateValue_stationary_local_uninvertable_%dlags_%dlen_%dvariables_%dindicatorlen" % (lags, length,latent_size,lags_indicator))
#     os.makedirs(path, exist_ok=True)
    
#     ### generate transition matrix
#     for i in range(int(Niter4condThresh)):
#         # A = np.random.uniform(0,1, (Ncomp, Ncomp))
#         A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
#         for i in range(latent_size):
#             A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
#         condList.append(np.linalg.cond(A))
        
#     ### generate the indicator matrix
#     for i in range(lags_indicator):
#         C_matrix = np.random.uniform(0, 1, (latent_size, 1))
#         indicator_matrix.append(C_matrix)

#     ### select transition matrix with condition number lager than top 25% percentile
#     condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
#     for l in range(lags):
#         B = generateUniformMat(latent_size, condThresh)
#         transitions.append(B)
#     transitions.reverse()
    
#     ### generate mixing matrix for each layer with orthogonal matrix
#     mixingList = []
#     for l in range(Nlayer - 1):
#         # generate causal matrix first:
#         A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
#         mixingList.append(A)

#     ### initialize the first lags latent variable
#     y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
#     y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

#     ### pass the latent variable through the maxing matrix for the first lags
#     yt = []; xt = []
#     for i in range(lags):
#         yt.append(y_l[:,i,:])
#     mixedDat = np.copy(y_l)
#     for l in range(Nlayer - 1):
#         mixedDat = leaky_ReLU(mixedDat, negSlope)
#         mixedDat = np.dot(mixedDat, mixingList[l])
#     x_l = np.copy(mixedDat)
#     for i in range(lags):
#         xt.append(x_l[:,i,:])
        
#     # Mixing function
#     for i in range(length):
#         if i < lags_indicator-lags:
#             # Transition function
#             y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
#             for l in range(lags):
#                 y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
#             y_t = leaky_ReLU(y_t, negSlope)
#             yt.append(y_t)
#             # Mixing function
#             mixedDat = np.copy(y_t)
#             for l in range(Nlayer - 1):
#                 mixedDat = leaky_ReLU(mixedDat, negSlope)
#                 mixedDat = np.dot(mixedDat, mixingList[l])
#             x_t = np.copy(mixedDat)
#             xt.append(x_t)
#             y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
#             x_l = np.concatenate((x_l, x_t[:,np.newaxis,:]),axis=1)
#         else:
#             # Transition function
#             y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
#             for l in range(lags):
#                 y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
#             y_t = leaky_ReLU(y_t, negSlope)
#             yt.append(y_t)
#             # Mixing function
#             indicator_value = np.zeros((batch_size,1))
#             for l in range(lags_indicator):
#                 indicator_value += np.dot(x_l[:,l,:], indicator_matrix[l])
#             indicators = np.where(indicator_value > 0, 1, -1).squeeze()
            
#             y_t_tulda = np.copy(y_t)
#             y_t_tulda[:,-1] = y_t_tulda[:,-1]+indicators.squeeze()
#             mixedDat = np.copy(y_t_tulda)
#             for l in range(Nlayer - 1):
#                 mixedDat = leaky_ReLU(mixedDat, negSlope)
#                 mixedDat = np.dot(mixedDat, mixingList[l])
#             x_t = np.copy(mixedDat)
#             xt.append(x_t)
#             y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
#             x_l = np.concatenate((x_l, x_t[:,np.newaxis,:]),axis=1)[:,1:,:]
            
#     # pdb.set_trace()
#     yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)

#     np.savez(os.path.join(path, "data"), 
#             yt = yt, 
#             xt = xt)

#     for l in range(lags):
#         B = transitions[l]
#         np.save(os.path.join(path, "W%d"%(lags-l)), B) 
          
          
          
          
          
          
          
          
            
            
# def pnl_gaussian_ts():
#     lags = 1
#     lags_indicator = 2
#     Nlayer = 3
#     length = 9
#     condList = []
#     negSlope = 0.2
#     latent_size = 3
#     transitions = []
#     indicator_matrix=[]
#     noise_scale = 0.1
#     batch_size = 40000
#     Niter4condThresh = 1e4

#     path = os.path.join(root_dir, "pnl_ts_2lag_value0")
#     os.makedirs(path, exist_ok=True)

#     for i in range(int(Niter4condThresh)):
#         # A = np.random.uniform(0,1, (Ncomp, Ncomp))
#         A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
#         for i in range(latent_size):
#             A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
#         condList.append(np.linalg.cond(A))

#     condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
#     for l in range(lags):
#         B = generateUniformMat(latent_size, condThresh)
#         transitions.append(B)
#     transitions.reverse()

#     mixingList = []
#     for l in range(Nlayer - 1):
#         # generate causal matrix first:
#         A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
#         mixingList.append(A)

#     y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
#     y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)


#     yt = []; xt = []
#     for i in range(lags):
#         yt.append(y_l[:,i,:])
#     mixedDat = np.copy(y_l)
#     for l in range(Nlayer - 1):
#         mixedDat = leaky_ReLU(mixedDat, negSlope)
#         mixedDat = np.dot(mixedDat, mixingList[l])
#     x_l = np.copy(mixedDat)
#     for i in range(lags):
#         xt.append(x_l[:,i,:])
        
#     # Mixing function
#     for i in range(length):

#         # Transition function
#         y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
#         # import ipdb; ipdb.set_trace()
#         y_t = y_t * np.mean(y_l, axis=1)
#         for l in range(lags):
#             y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
#         y_t = leaky_ReLU(y_t, negSlope)
#         yt.append(y_t)
#         # Mixing function

#         mixedDat = np.copy(y_t)
#         for l in range(Nlayer - 1):
#             mixedDat = leaky_ReLU(mixedDat, negSlope)
#             mixedDat = np.dot(mixedDat, mixingList[l])
#         x_t = np.copy(mixedDat)
#         xt.append(x_t)
#         y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]


#     yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
#     import ipdb; ipdb.set_trace()

#     np.savez(os.path.join(path, "data"), 
#             yt = yt, 
#             xt = xt)

#     for l in range(lags):
#         B = transitions[l]
#         np.save(os.path.join(path, "W%d"%(lags-l)), B) 

#     indicator_matrix.append(np.array([[0.1],[0.6],[0.3]]))
#     indicator_matrix.append(np.array([[0.5],[0.2],[0.3]]))
#     np.save(os.path.join(path,"IM"),np.array(indicator_matrix))
            
if __name__ == "__main__":
        #noisecoupled_gaussian_ts()
    # pnl_gaussian_ts()
    stationary_local_uninvertable_z_tulda()