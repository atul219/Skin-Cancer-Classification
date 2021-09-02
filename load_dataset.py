from config import *
from transformations import transform
import os, random, re, math, time
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

import PIL


def read_labeled_tfrecord(example):
    """
    read tfrecord data examples and return a image

     : param example: input data example
    : return: dataset example of image
    """
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),
        'target'                       : tf.io.FixedLenFeature([], tf.int64)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['target']


def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['image_name'] if return_image_name else 0


def prepare_image(img, cfg=None, augment=True):    
    """
    apply transformation to an image and return transformed images

    : param img: input image
    : param cfg: config file
    : param augment (bool): True or False
    : return: augmented image
    """
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg['img_size'], cfg['img_size']])
    img = tf.cast(img, tf.float32) / 255.0
    
    if augment:
        img = transform(img, cfg)
        img = tf.image.random_crop(img, [cfg['crop_size'], cfg['crop_size'], 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)

    else:
        img = tf.image.central_crop(img, cfg['crop_size'] / cfg['img_size'])
                                   
    img = tf.image.resize(img, [cfg['model_input'], cfg['model_input']])
    img = tf.reshape(img, [cfg['model_input'], cfg['model_input'], 3])
    return img


def step_counts(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)


def dataset(files, cfg, augment = False, shuffle = False, repeat = False, 
                labeled=True, return_image_names=True):
    """
    read image files and return batch of dataset
     : param file: input image file
    : param cfg: config file
    : param augment (bool): True or False
    : param shuffle (bool): True or False
    : param repeat (bool): True or False
    : param labeled (bool): True or False
    : param return_image_names (bool) True or False
    : return: dataset

    """
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()
    
    if repeat:
        ds = ds.repeat()
    
    if shuffle: 
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
        
    if labeled: 
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 
                    num_parallel_calls=AUTO)      
    
    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, cfg=cfg), 
                                               imgname_or_label), 
                num_parallel_calls=AUTO)
    
    ds = ds.batch(cfg['batch_size'] * REPLICAS)
    ds = ds.prefetch(AUTO)
    return ds
