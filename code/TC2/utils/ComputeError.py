"""
Project: TC2
Description: Methods to compute imputation error 

To-do:

Author: Jingshu Liu
Log: 
    2016-05-22 JL: Program first created
  
Input:
    X, Y: numpy arrays
    
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
    
