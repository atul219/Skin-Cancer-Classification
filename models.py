
from config import *
import os, random, re, math, time
import numpy as np
import pandas as pd
import efficientnet.tfkeras as efficientnet
import tensorflow as tf
import tensorflow.keras.backend as K

def models(cfg):
    """
    a list of 7 efficient version of models to training

    : param cfg: config file
    : return a list of models
    """
    input = tf.keras.Input(shape=(cfg['model_input'], cfg['model_input'], 3), name='imgIn')

    temp = tf.keras.layers.Lambda(lambda x:x)(input)
    
    outputs = []    
    for i in range(cfg['models']):
        constructor = getattr(efficientnet, f'EfficientNetB{i}')
        
        x = constructor(include_top=False, weights='imagenet', 
                        input_shape=(cfg['model_input'], cfg['model_input'], 3), 
                        pooling='avg')(temp)
        
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        outputs.append(x)
        
    model = tf.keras.Model(input, outputs, name='aNetwork')
    model.summary()
    return model



def compile_models(cfg, loss = 'binary'): 
    """
    Compile model and calculate loss

    :param: cfg: config file
    :param loss: loss can be binary (Binary Cross Entropy or focal-loss)
    :return : a compile model to train
    """   
    with strategy.scope():
        model = models(cfg)

        
        if loss == 'focal':
            losses = [BinaryFocalLoss(gamma = 2, label_smoothing = cfg['label_smoothing']) 
                      for i in range(cfg['models'])]
        else:
            losses = [tf.keras.losses.BinaryCrossentropy(label_smoothing = cfg['label_smoothing'])
                  for i in range(cfg['models'])]
        
        model.compile(
            optimizer = cfg['optimizer'],
            loss      = losses,
            metrics   = [tf.keras.metrics.AUC(name='auc')])
        
    return model