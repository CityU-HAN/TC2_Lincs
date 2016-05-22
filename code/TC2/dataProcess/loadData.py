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

# srcDir = ""~/Google Drive/NYUProjects/Rachel_DrugRepurpose/code/TC2/dataProcess"
srcDir = os.path.dirname(__file__)
dataDir = os.path.join(srcDir, "../../../data")

def loadSubsetD6():
    robjects.r['load'](str(dataDir) + "/V1/inputs/d6_subset.RData")
    out_d6subset = np.asarray(robjects.r['T_meas'])
    return out_d6subset