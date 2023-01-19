import numpy as np
import pandas as pd

    
def generate_data(nK, pH, location, scale, random_state = None):
    
    ''''
    n is the number os objects
    p is the number of variables
    p2 is the number of irrelevant variables. (If there are irrelevant variables, then the number of relevant variables is p - p2)
    K is the number of object clusters
    H is the number of variables clusters
    location1 is a matrix K x H with the means of the gaussian distribution to generate each block
    scale1 is a matrix K x H with the standard deviations of the gaussian distribution to generate each block
    location2 is a vector of means of size p2 for multivariate normal distribution used to generate the irrelevant variables
    scale2  is a scale or a vector of stanrdart deviations for multivariate normal distribution used to generate the irrelevante variables
    random_state is the seed for generate the data, i.e controls randomness to generate the data
    
    PS: if p2 is zero, then location2 and scale2 must be None
    '''

    K = len(nK)
    H = len(pH)
    n = int(nK.sum())
    p = int(pH.sum())

    X = pd.DataFrame(np.zeros((n,p)))
    
    lk = np.cumsum(nK,dtype='int64')
    lk = np.concatenate((np.array([0]),lk))
    
    lh = np.cumsum(pH,dtype='int64')
    lh = np.concatenate((np.array([0]),lh))
   
    labels = np.array([])
    for k in range(K):
        labels = np.concatenate((labels,np.repeat(k,nK[k])))
        ik = np.arange(lk[k],lk[k+1])
        for h in range(H):
            jh = np.arange(lh[h],lh[h+1])
            np.random.seed(random_state)
            X.iloc[ik,jh]= np.random.normal(loc = location[k,h],scale = scale[k,h], size = ( nK[k], pH[h] )) 

    return X.to_numpy(),labels.astype('int64')
    

