"""
Project: TC2
Description: Imputate tensor by cross validation

Note: 
    1. Assume the dimension of the data is always [drug, gene, cell]
    2. For CV on fibers, only do gene-fiber
    3. For CV on slices, consider all combinations

To-do:
    1. Enable remove slice 
    2. Add parallel methods

Author: Jingshu Liu
Log: 
    2016-05-23 JL: Program first created
"""
#import tensorflow as tf
import time
import copy
import numpy as np
from TC2 import CompleteTensor as ct
#import pdb

class CV_CompleteTensor:
    """
    Input:
        T: Tensor input as numpy array
        CVType: ['loo fiber', 'k-fold fiber', 'loo slice', 'k-fold slice']
        kfold: number of folds, or don't assign if use 'leave-one-out'
        TCmethod: CompleteTensor method
        paras: CompleteTensor parameters
    
    Output: 
        T_CV: cross-validated (imputed) tensor
        timePerFold: running time per fold
    """
    
    def __init__(self, CVType, TCmethod, paras):
        self.CVType = CVType
        self.TCmethod = TCmethod
        self.paras = paras
                
    def _run(self, T, kfold = 1):
        self.T = T.astype('float')
        folds = {
        'loo fiber': self.T.shape[0] * self.T.shape[2],
        #'loo slice': self.T.shape[0],
        }
        
        self.kfold = folds.get(self.CVType, kfold) 
        #If no folds number passed, and not loo method, assume no CV (1 fold)

        switcher = {
        'loo fiber': self.CV_fLOO,
        'k-fold fiber': self.CV_fKfold,
        #'loo slice': self.CV_sLOO,
        #'k-fold slice': self.CV_sKfold,        
        }
        func = switcher.get(self.CVType, "Unexpected method")  
        
        if func == 'Unexpected method':
            raise NameError('Unexpected method')
        else:
            start = time.clock()              
            T_CV = func()            
            end = time.clock()
            timePerFold = (end - start)*1.0/self.kfold
            return T_CV, timePerFold
    
    def _splitKfold(self, k):
        """
        Get the value splits for k-fold cross validation. 
        E.g., 4-fold, lsSplit = [[0,0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.01]]
        For the ith-fold, take lsSplit[i][0] <= rand_idx < lsSplit[i][1]
        """
        splitDist = 1.0/k
        lsSplit = []
        for i in range(0, k):
            valMin = i*splitDist
            if i == k - 1:
                valMax = 1.01
            else:
                valMax = (i + 1)*splitDist
            lsSplit.append([valMin, valMax])
        return lsSplit
         
         
    def onePass(self, idx_removed):
        T_input = copy.copy(self.T)
        T_input[idx_removed] = 'nan'        
        T_out, runtime = ct.CompleteTensor(T_input, self.TCmethod, self.paras)
        return T_out
        
    def CV_fLOO(self):
        """
        Leave-one-out CV by fiber
        """
        T_CV = np.empty(self.T.shape) # Initiate output tensor
        lsCombo = [[i,k] for i in range(0, self.T.shape[0]) for k in range(0, self.T.shape[2])]
        
        i = 0
        for combo in lsCombo:
            idx_removed = np.full(self.T.shape, False, dtype=bool) # Initiate index tensor
            idx_removed[combo[0], : , combo[1]] = True
            T_out = self.onePass(idx_removed)
            T_CV[idx_removed] = T_out[idx_removed]
            
            if i%100 == 0:
                print 'Fold count: ' + str(i+1)            
            i = i + 1
                        
        idx_miss = np.isnan(self.T) # Force T_CV to have the same missing pattern as input
        T_CV[idx_miss] = 'nan'        
        
        return T_CV

    def CV_fKfold(self):
        """
        K-fold CV by fiber
        """
        T_CV = np.empty(self.T.shape) # Initiate output tensor

        lsCombo = [[i,k] for i in range(0, self.T.shape[0]) for k in range(0, self.T.shape[2])]
        
        # Get random splits
        rand_state = np.random.RandomState(seed=1)
        lsRand = np.ravel(rand_state.rand(len(lsCombo),1))
        
        lsSplit = self._splitKfold(self.kfold)

        for i in range(0, self.kfold):
            idxCV = (lsRand >= lsSplit[i][0]) & (lsRand < lsSplit[i][1])
            idx_removed = np.full(self.T.shape, False, dtype=bool) # Initiate index tensor
            for j in range(0, len(lsCombo)):  
                if idxCV[j]:
                    idx_removed[lsCombo[j][0], : , lsCombo[j][1]] = True

            T_out = self.onePass(idx_removed)
            T_CV[idx_removed] = T_out[idx_removed]
            print 'Fold completed: ' + str(i+1)

        idx_miss = np.isnan(self.T) # Force T_CV to have the same missing pattern as input
        T_CV[idx_miss] = 'nan'        
            
        return T_CV
         