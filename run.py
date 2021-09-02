!pip install efficientnet
!pip install focal-loss

import os, random, re, math, time

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efficientnet

import PIL

from kaggle_datasets import KaggleDatasets

from tqdm import tqdm

random.seed(21)

CONFIG = dict(
    models = 7, 
    batch_size = 16, 
    img_size = 512, 
    crop_size = 500, 
    model_input = 384, 
    optimizer = 'adam', 
    label_smoothing = 0.05,
    learning_rate = 0.0005,
    lr_max = 0.0002,
    lr_min = 0.00001,
    lr_ramp_epochs = 7,
    lr_sustain_epochs = 0,
    lr_exp_decay = 0.8,
    epochs = 20,
    rot = 180.0,
    shr = 2.0,
    hzoom = 8.0,
    wzoom = 8.0,
    hshift = 8.0,
    wshift = 8.0,
    DEVICE = 'TPU',
    loss = 'binary',
    tta_steps =  25
)

PATH = "../input/siim-isic-melanoma-classification"
TRAIN_CSV = os.path.join(PATH, 'train.csv')
TEST_CSV = os.path.join(PATH, 'test.csv')
SAMPLE_SUBMISSION_CSV = os.path.join(PATH, 'sample_submission.csv')


# PATH FOR DIFFERENT DATASET 
TF_PATH    = KaggleDatasets().get_gcs_path('melanoma-512x512')
files_train = np.sort(np.array(tf.io.gfile.glob(TF_PATH + '/train*.tfrec')))
files_test  = np.sort(np.array(tf.io.gfile.glob(TF_PATH + '/test*.tfrec')))


################# CHECKING TPU #############################

DEVICE = CONFIG['DEVICE']

t  = np.sort(np.array(tf.io.gfile.glob(TF_PATH + '/test*.tfrec')))

if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except :
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')


################# TRANSFORMATION #############################

def mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))


def transform(image, cfg):    
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = cfg["model_input"]
    XDIM = DIM%2 #fix for size 331
    
    rot = cfg['rot'] * tf.random.normal([1], dtype='float32')
    shr = cfg['shr'] * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['hzoom']
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['wzoom']
    h_shift = cfg['hshift'] * tf.random.normal([1], dtype='float32') 
    w_shift = cfg['wshift'] * tf.random.normal([1], dtype='float32') 

    # GET TRANSFORMATION MATRIX
    m = mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM, DIM,3])


################# READ TFRECORD DATASET #############################




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


############# IMAGE TRANSFORMATION################################################

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


############ LEARNING RATE UPDATE #####################################
def get_lr_callback(cfg):
    """
    will update learning rate on epochs
    : param cfg: config file
    : return: updated learning rate
    """
   
    def lrfn(epoch):
        if epoch < cfg['lr_ramp_epochs']:
            lr = (cfg['lr_max'] * strategy.num_replicas_in_sync - cfg['learning_rate']) / cfg['lr_ramp_epochs'] * epoch + cfg['learning_rate']
            
        elif epoch < cfg['lr_ramp_epochs'] + cfg['lr_sustain_epochs']:
            lr = cfg['lr_max'] * strategy.num_replicas_in_sync
            
        else:
            lr = (cfg['lr_max'] * strategy.num_replicas_in_sync - cfg['lr_min']) * cfg['lr_exp_decay']**(epoch - cfg['lr_ramp_epochs'] - cfg['lr_sustain_epochs']) + cfg['lr_min']
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


############ MODELS ####################

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
TEST_DATA = dataset(files_test, CONFIG, augment=True, repeat=True, 
                         labeled=False, return_image_names=False)

probs = model.predict(TEST_DATA, verbose=1, steps=steps)

probs = np.stack(probs)
probs = probs[:,:steps_test * CONFIG['tta_steps']]
probs = np.stack(np.split(probs, CONFIG['tta_steps'], axis=1), axis=1)
probs = np.mean(probs, axis=1)

