"""
Project: TC2
Description: Compute imputation error by cross validation

To-do:
    1. Clarify on K-fold CV, finish CV_LOO/Kfold methods
    2. Enable remove slice 
    3. Add parallel methods

Author: Jingshu Liu
Log: 
    2016-05-23 JL: Program first created
"""
#import tensorflow as tf
import time
import copy
import numpy as np
from TC2 import CompleteTensor as ct

class CV_CompleteTensor:
    """
    Input:
        T: Tensor input as numpy array
        CVType: ['leave-one-out', 'k-fold']
        TCmethod: CompleteTensor method
        paras: CompleteTensor parameters
    
    Output: 
        T_CV: cross-validated (imputed) tensor
        errorStats
    """
    
    def __init___(self, T, TCmethod, kfold, paras):
        self.T = T.astype('float')
        self.TCmethod = TCmethod
        self.paras = paras
        if kfold:
            self.kfold = kfold
        else:
            self.kfold = len(T) # Modify here, number of iterations
        
    def _run(self):
        switcher = {
        'leave-one-out': self.CV_LOO,
        'k-fold': self.CV_Kfold,
        }

        func = switcher.get(self.TCmethod, "Unexpected method")    
        if func == 'Unexpected method':
            raise NameError('Unexpected method')
        else:
            start = time.clock()  
            
            T_CV = func(self)
            
            end = time.clock()
            runtime = end - start
            return T_CV, runtime
    
    def onePass(self, idx_removed):
        T_input = copy.copy(self.T)

        T_input[idx_removed] = 'nan'        
        T_out = ct.CompleteTensor(T_input, self.TCmethod, self.paras)
        #---- LEFT HERE ------
        return T_out
        

    def CV_LOO(self):
        T_CV = 0
        return T_CV

    def CV_Kfold(self):
        T_CV = 0
        return T_CV
         