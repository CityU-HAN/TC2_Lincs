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
#import numpy as np

class MeanTC:
    """
    Input:
        T: Input tensor
        paras: parameters
    Output:
        T_model: completed tensor
    
    """
    def __init__(self, T, paras):
        self.T = tf.convert_to_tensor(T, dtype = 'float32')
        self.paras = paras
        
        
    def _run1D(self, sess):
        # Fake output = all 1 tensor with the same shape as input
        self.T_model = tf.ones([i for i in self.T._shape])
        sess.run(tf.initialize_all_variables())
        
        T_model = sess.run(self.T_model)
        return T_model
        
    def _run2D(self, sess):
        # Fake output = all 2 tensor with the same shape as input
        self.T_model = tf.constant(2, shape = [i for i in self.T._shape])
        sess.run(tf.initialize_all_variables())

        T_model = sess.run(self.T_model)
        return T_model