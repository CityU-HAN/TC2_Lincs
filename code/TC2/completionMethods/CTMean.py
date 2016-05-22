"""
Project: TC2
Description: Implementing the 1D-mean and 2D-mean methods for tensor completion

Note: Current version is a placeholder
To-do:
    1. Replace the placeholder script with the real method

Author: Jingshu Liu
Log: 
    2016-05-22 JL: Program first created
"""
import tensorflow as tf
import numpy as np

class CTMean:
    """
    Input:
        T: Input tensor
        paras: parameters
    Output:
        T_model: completed tensor
    
    """
    def __init__(self):
        
        
    def Mean1D(self, T, paras):
        # Fake output = all zero tensor with the same shape as input
        T2 = tf.convert_to_tensor(T, dtype = 'float32')
        T_model = tf.ones([i for i in T2._shape])
        return T_model
        
    def Mean2D(self, T, paras):
        # Fake output = all zero tensor with the same shape as input
        T2 = tf.convert_to_tensor(T, dtype = 'float32')
        T_model = tf.constant(2, shape = [i for i in T2._shape])
        return T_model