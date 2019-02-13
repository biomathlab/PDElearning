import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import Lambda

'''
activation functions
'''

# soft approx to ReLU
def softplus(x):
    return tf.log(1.0 + tf.exp(x))

'''
layers
'''

def softplus_layer(**kwargs):
    def layer(x):
        return softplus(x)
    return Lambda(layer, **kwargs)

'''
loss functions
'''

# proportional error loss function 
def gls_loss(y_true, y_pred):
    
    # proportionality constant
    gamma = 1.0
    
    # threshold cutoff
    threshold = 0.0001
    
    # |y_pred| entries < threshold get set to 1
    y_pred_abs = tf.abs(y_pred)
    y_prop = tf.cast(y_pred_abs >= threshold, y_pred.dtype)*y_pred_abs + \
                     tf.cast(y_pred_abs < threshold, y_pred.dtype)
    
    # (y_pred - y_true)/(|y_pred|^gamma)
    residual = (y_pred - y_true)/(y_prop**gamma)
    
    # mean square proportional error
    mspe = K.mean(K.square(residual), axis=-1) 
    
    return mspe 

# gls loss with penalty for invalid outputs
def gls_thresh_loss(y_true, y_pred):
    
    # proportionality constant
    gamma = 1.0
    
    # threshold cutoff
    threshold = 0.0001
    
    # |y_pred| entries < threshold get set to 1
    y_pred_abs = tf.abs(y_pred)
    y_prop = tf.cast(y_pred_abs >= threshold, y_pred.dtype)*y_pred_abs + \
             tf.cast(y_pred_abs < threshold, y_pred.dtype)
    
    # mean square proportional error, (y_pred - y_true)/(|y_pred|^gamma)
    mspe = K.mean(tf.square((y_pred - y_true)/(y_prop**gamma)), axis=-1) 
    
    # mean square loss penalty for values not in [0, 1]
    penalty = K.mean(tf.square(tf.cast(y_pred > 1.0, y_pred.dtype)*y_pred_abs + \
                               tf.cast(y_pred < 0.0, y_pred.dtype)*(y_pred_abs + 1.0)))
    
    return mspe + penalty