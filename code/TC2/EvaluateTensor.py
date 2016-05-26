"""
Project: TC2
Description: Evaluate CV output 

Note: 

To-do:
    1. Add scatter plot
    2. Add ROC plot

Author: Jingshu Liu
Log: 
    2016-05-26 JL: Program first created
  
Input:
    d6_subset
    
Output: 
    evaluation summary output
"""
try: 
  import cPickle as pickle
except:
  import pickle
import numpy as np
import pandas as pd
import os

from TC2.dataProcess import loadData
from TC2 import CrossValidateTensor as cv
from TC2.utils import ComputeError as ce

srcDir = os.path.dirname(__file__)
dataDir = os.path.join(srcDir, "../../data")

def main():
    # Load test data
    d6_subset, names = loadData.loadSubsetD6()
    print 'data loaded'
    print names
    
    # Get CV Tensor
    lsMethods = ['mean1d', 'mean2d', 'FaLRTC']
    paras = {}
    d6_subset_CVout = []
    PCT, PCTd, PCTg, PCTc = [],[],[],[]
    AUC, GCP  = [], []
    
    for method in lsMethods:
        cv1 = cv.CV_CompleteTensor('loo fiber', method, paras)
        T_out, runtime = cv1._run(d6_subset)
        print str(method) + ' running time per fold: ' + str(runtime)
        d6_subset_CVout.append(T_out)
    
        # scatter plots/PCT (Fig 3A)
        pct = ce.npPCT(T_out, d6_subset)
        PCT.append(pct)
        
        # ROC curves for predicting DEGs/AUC (Fig 3B)
        auc = ce.npAUC(T_out, d6_subset, topQuantile = 0.01)
        AUC.append(auc)
        
        # preservation of gene correlation structure (Fig 4)
        lsGcp = ce.npGCP_allCell(T_out, d6_subset)
        GCP.append(np.nanmean(lsGcp))
    
        # mode-specific accuracy (Fig 6)
        lsPctd, lsPctg, lsPctc = ce.npPCT_allMode(T_out, d6_subset)
        PCTd.append(np.nanmean(lsPctd))
        PCTg.append(np.nanmean(lsPctg))
        PCTc.append(np.nanmean(lsPctc))
    
    dfSummary = pd.DataFrame({'Method': lsMethods, 
                              'PCT': PCT,
                              'AUC_DEG1': AUC,
                              'GCP': GCP,
                              'PCTd': PCTd,
                              'PCTg': PCTg,
                              'PCTc': PCTc
                              })
    
    dfSummary.to_csv(str(dataDir) + '/V1/outputs/d6_subset_smry_python.csv', index= False)    
    pickle.dump(d6_subset_CVout, open(str(dataDir) + '/V1/outputs/d6_subset_CVout.p', 'wb'))                  

if __name__ == '__main__':
    main()