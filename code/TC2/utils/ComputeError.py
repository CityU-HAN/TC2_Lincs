"""
Project: TC2
Description: Methods to compute imputation error 

Note: 
    1. npPCT: pct = 0 if no pairwise completed obs, or no variance
    2. npAUC: nans are ranked at bottom

To-do:

Author: Jingshu Liu
Log: 
    2016-05-22 JL: Program first created
  
Input:
    X, Y: numpy arrays with the same dimensions, except for otherwise specified
    
Output: 
    errorStat
"""

import tensorflow as tf
import numpy as np
from scipy.stats import rankdata
from sklearn import metrics

import TC2.utils.MatrixTransform as mt

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
    
def npPCT(X, Y):
    """
    Compute the pearson correlation, treating the inputs as two vectors    
    """
    X2 = X.flatten()
    Y2 = Y.flatten()
    data = [X2, Y2]
    maskedarr = np.ma.array(data, mask=np.isnan(data))
    pct = np.ma.corrcoef(maskedarr) # pct = 0 if no pairwise completed obs, or no variance
    pct_data = pct.data
    pct_data[pct.mask] = np.nan # make masked value to be missing
    return pct_data[0,1]

def npPCT_allMode(X, Y):
    """
    Compute the pearson correlation of different modes. E.g., for the same drug, the correlation between true and imputed (gene X cell)   
    Output:
        lsPCTd, lsPCTg, lsPCTc: list of PCT of the corresponding mode
    """
    dlen, glen, clen = X.shape
    lsPCTd = [npPCT(X[i,:,:], Y[i,:,:]) for i in range(0, dlen)]
    lsPCTg = [npPCT(X[:,i,:], Y[:,i,:]) for i in range(0, glen)]
    lsPCTc = [npPCT(X[:,:,i], Y[:,:,i]) for i in range(0, clen)]
    
    return lsPCTd, lsPCTg, lsPCTc

def npGCP(Xm, Ym):
    """
    1. For one cell, compute the gene correlation matrix
    2. Evaluate cell-specific gene correlation preservation by the PCT of the correlation matrices
    Input: 
        Xm, Ym: two matrices with shape as drug X gene
    Output:
        GCP: Gene correlation preservation measure
    """
    vt_corMatX = mt.np_matCorToVector(Xm, rowvar = False)
    vt_corMatY = mt.np_matCorToVector(Ym, rowvar = False)
    
    GCP = npPCT(vt_corMatX, vt_corMatY)
    return GCP
        
def npGCP_allCell(X, Y):
    """
    Output:
        lsGCPc: list of GCP for each cell type
    """
    dlen, glen, clen = X.shape
    lsGCPc = [npGCP(X[:,:,i], Y[:,:,i]) for i in range(0, clen)]
    return lsGCPc
    
def npAUC(X, Y, topQuantile = 0.01):
    """
    Rank gene expression in each drug+cell profile, compare the association with true rank.
    True rank is denoted by 0/+-1, where +1 means value >= top quantile (abs), while -1 means value <= -top quantile (abs)
    """
    XFill = np.abs(X)
    XFill[np.isnan(X)] = 0 # Fill nan as 0 so that they are ranked at the bottom
    XRank = np.apply_along_axis(rankdata, 1, XFill) 
    XRank[np.isnan(X)] = np.nan # put the ranking of nans to nan
    
    YRank = np.abs(np.apply_along_axis(lambda y: mt.np_quantileTo01(y, topQuantile), 1, Y)) 

    XRank_rmNan, YRank_rmNan = mt.np_pairFlatRmNA(XRank, YRank)
    AUC = metrics.roc_auc_score(YRank_rmNan, XRank_rmNan)
    return AUC
  