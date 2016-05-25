"""
Project: TC2
Description: Methods for matrix transformations

Note: 

To-do:

Author: Jingshu Liu
Log: 
    2016-05-24 JL: Program first created
  
Input:
    Xm: numpy 2-d arrays 
    Xv: numpy 1-d vector
    
"""

import numpy as np
from scipy.stats import rankdata

def np_matCorToVector(Xm, rowvar = False):
    """
    Get the upper triangular (excluding diagonal) of the correlation matrix based on Xm
    Input:
        rowvar: True/False, whether use row as variable or not
    Output:
        vt_corMatX: vector of the correlations
    """
    Xm = np.asarray(Xm)
    maskedarrX = np.ma.array(Xm, mask=np.isnan(Xm))
    corMatX = np.ma.corrcoef(maskedarrX, rowvar = rowvar) # pct = 0 if no pairwise completed obs, or no variance
    
    corMatX_data = corMatX.data    
    corMatX_data[corMatX.mask] = np.nan # Set pct to missing if not pairwise completed
    
    vt_corMatX = []
    nrow, ncol = Xm.shape
    for i in range(0, nrow):
        for j in range(i+1, ncol):
            vt_corMatX.append(corMatX_data[i,j])
    
    vt_corMatX = np.asarray(vt_corMatX)
    return vt_corMatX
    
def np_quantileTo01(Xv, top_quantile = 0.01):
    """
    Transform top Xv values to +-1, rest to 0, by passing quantile threshold or not
    """
    Xv_out = np.zeros(len(Xv))
    quantile = np.nanpercentile(abs(Xv), 100-100*top_quantile)
    Xv_out[Xv >= quantile] = 1
    Xv_out[Xv <= -quantile] = -1
    return Xv_out
    
    
    
    