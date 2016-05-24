"""
Project: TC2
Description: Load R data sample

Note: 
To-do:

Author: Jingshu Liu
Log: 
    2016-05-22 JL: Program first created
"""

import rpy2.robjects as robjects
import numpy as np
import os

# srcDir = "~/Google Drive/NYUProjects/Rachel_DrugRepurpose/code/TC2/dataProcess"
srcDir = os.path.dirname(__file__)
dataDir = os.path.join(srcDir, "../../../data")

def getNames(robject):
    """
    Assume the data is always [drug, gene, cell]
    Get the names of each dimension
    """
    names = {
    'dnames': [x for x in robject.names[0]], 
    'gnames': [x for x in robject.names[1]], 
    'cnames': [x for x in robject.names[2]]
    }
    return names    

def loadSubsetD6():
    robjects.r['load'](str(dataDir) + "/V1/inputs/d6_subset.RData")
    a = robjects.r['T_meas']
    out_d6subset = np.asarray(a)
    names = getNames(a)
    return out_d6subset, names