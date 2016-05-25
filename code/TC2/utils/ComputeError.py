"""
Project: TC2
Description: Methods to compute imputation error 

Note: 
    1. npPCT: pct = 0 if no pairwise completed obs, or no variance

To-do:

Author: Jingshu Liu
Log: 
    2016-05-22 JL: Program first created
  
Input:
    X, Y: numpy arrays with the same dimensions
    
Output: 
    errorStat
"""

import tensorflow as tf
import numpy as np

def tfL2Error(X, Y):    
    X2 = tf.convert_to_tensor(X, dtype = 'float32')
    Y2 = tf.convert_to_tensor(Y, dtype = 'float32')

    dif = tf.squared_difference(X2,Y2)
    l2error = tf.reduce_mean(dif)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    return sess.run(l2error)

def npL2Error(X, Y):
    dif = X - Y
    l2error = np.nanmean(dif**2)
    return l2error
    
def npPCT(X, Y, dimType):
    """
    Compute the pearson correlation, treating the inputs as two vectors
    Input: 
        dimType: ['all', 'd', 'g', 'c']
    
    """
    X2 = X.flatten()
    Y2 = Y.flatten()
    data = [X2, Y2]
    maskedarr = np.ma.array(data, mask=np.isnan(data))
    pct = np.ma.corrcoef(maskedarr).data[0,1] # pct = 0 if no pairwise completed obs, or no variance
    return pct
    
def npGCP(X, Y):
    gcp = 0
    return gcp
    
def npAUC(X, Y):
    gcp = 0
    return gcp