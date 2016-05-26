"""
Project: TC2
Description: Implementing the 1D-mean and 2D-mean methods for tensor completion

Note: 
    1. Default alpha for 2D-mean is 0.5
    2. Clarify - current 2D-mean will be nan if one of the component is nan
To-do:

Author: Jingshu Liu
Log: 
    2016-05-22 JL: Program first created
"""
#import tensorflow as tf
import numpy as np
import copy

class MeanTC:
    """
    Input:
        T: Input tensor
        paras: parameters
    Output:
        T_model: completed tensor
    
    """
    def __init__(self, T, paras):
        #self.T = tf.convert_to_tensor(T, dtype = 'float32')
        self.T = T
        self.paras = paras
        self.dlen, self.glen, self.clen = self.T.shape
        
    def _run1D(self, sess):
        """
        Average the value of the drug-gene combination across all cell types
        """        
        T_model = copy.copy(self.T)
        mtMean = np.nanmean(self.T, axis = 2) # Get drug-gene specific mean
        T_Mean = mtMean.reshape(self.dlen, self.glen, 1).repeat(self.clen, 2)        
        T_model[np.isnan(self.T)] = T_Mean[np.isnan(self.T)]
        return T_model
        
    def _run2D(self, sess): 
        """
        T_Meanc: Average the value of the drug-gene combination across all cell types, same as _run1D
        T_Meand: Average the value of the gene-cell combination across all drugs
        T_Mean = (1-alpha) * T_Meanc + alpha * T_Meand, alpha default to 0.5
        """        
        self.alpha = self.paras.get('alpha', 0.5) #alpha default to 0.5
        T_model = copy.copy(self.T)
        
        mtMeanc = np.nanmean(self.T, axis = 2) # Get drug-gene specific mean
        T_Meanc = mtMeanc.reshape(self.dlen, self.glen, 1).repeat(self.clen, 2)

        mtMeand = np.nanmean(self.T, axis = 0) # Get cell-gene specific mean   
        T_Meand = mtMeand.reshape(1, self.glen, self.clen).repeat(self.dlen, 0) 
        
        T_Mean = (1-self.alpha) * T_Meanc + self.alpha * T_Meand
        T_model[np.isnan(self.T)] = T_Mean[np.isnan(self.T)]
        return T_model
