"""
Project: TC2
Description: Wrapper to impute tensor

Note: 
To-do:
    1. Add normalization

Author: Jingshu Liu
Log: 
    2016-05-22 JL: Program first created


Inputs:
    T: Tensor dataset
    method: TC method (i.e. name of algorithm), including ['mean1d', 'mean2d', 'FaLRTC']
    paras: User-specified hyperparameters for the specific method    
Outputs:
    T_model: Completed tensor
    runtime: Runtime    
"""

import time
import tensorflow as tf
from TC2.completionMethods import CTMean, CTFaLRTC

def CompleteTensor(T, method, paras):
    switcher = {
        'mean1d': CTMean.MeanTC(T, paras)._run1D,
        'mean2d': CTMean.MeanTC(T, paras)._run2D,
        'FaLRTC': CTFaLRTC.FaLRTC(T, paras)._run,
    }

    func = switcher.get(method, "Unexpected method")    
    if func == 'Unexpected method':
        raise NameError('Unexpected method')
    else:
        start = time.clock()  
        
        sess = tf.Session()
        T_model = func(sess)
        
        end = time.clock()
        runtime = start - end
        return (T_model, runtime)

