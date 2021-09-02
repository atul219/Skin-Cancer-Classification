import os, random, re, math, time


import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efficientnet
from kaggle_datasets import KaggleDatasets
from tqdm import tqdm

from load_dataset import dataset, step_counts
from models import compile_models
from utils import *
from config import CONFIG

random.seed(21)

TRAIN_DATA     = dataset(files_train, CONFIG, augment=True, shuffle=True, repeat=True)
TRAIN_DATA     = TRAIN_DATA.map(lambda img, label: (img, tuple([label] * CONFIG['models'])))

steps_train  = step_counts(files_train) / (CONFIG['batch_size'] * REPLICAS)

model        = compile_models(CONFIG)
history      = model.fit(TRAIN_DATA, 
                         verbose          = 1,
                         steps_per_epoch  = steps_train, 
                         epochs           = CONFIG['epochs'],
                         callbacks        = [get_lr_callback(CONFIG)])


CONFIG['batch_size'] = 256

steps_test   = step_counts(files_test)
steps      = steps_test / (CONFIG['batch_size'] * REPLICAS) * CONFIG['tta_steps']
ds_testAug = dataset(files_test, CONFIG, augment=True, repeat=True, 
                         labeled=False, return_image_names=False)

probs = model.predict(ds_testAug, verbose=1, steps=steps)

probs = np.stack(probs)
probs = probs[:,:steps_test * CONFIG['tta_steps']]
probs = np.stack(np.split(probs, CONFIG['tta_steps'], axis=1), axis=1)
probs = np.mean(probs, axis=1)