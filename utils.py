import os
import PIL
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from load_dataset import get_dataset

from config import CONFIG

# DATASET PATHS

PATH = "../input/siim-isic-melanoma-classification"
TRAIN_CSV = os.path.join(PATH, 'train.csv')
TEST_CSV = os.path.join(PATH, 'test.csv')
SAMPLE_SUBMISSION_CSV = os.path.join(PATH, 'sample_submission.csv')


# PATH FOR DIFFERENT DATASET 
TF_PATH    = KaggleDatasets().get_gcs_path('melanoma-512x512')
files_train = np.sort(np.array(tf.io.gfile.glob(TF_PATH + '/train*.tfrec')))
files_test  = np.sort(np.array(tf.io.gfile.glob(TF_PATH + '/test*.tfrec')))






# to plot data
def show_dataset(thumb_size, cols, rows, ds):
    mosaic = PIL.Image.new(mode='RGB', size=(thumb_size*cols + (cols-1), 
                                             thumb_size*rows + (rows-1)))
   
    for idx, data in enumerate(iter(ds)):
        img, target_or_imgid = data
        ix  = idx % cols
        iy  = idx // cols
        img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        img = img.resize((thumb_size, thumb_size), resample=PIL.Image.BILINEAR)
        mosaic.paste(img, (ix*thumb_size + ix, 
                           iy*thumb_size + iy))

    display(mosaic)
    
# ds = get_dataset(files_train, CFG).unbatch().take(12*5)   
# show_dataset(64, 12, 5, ds)


# test augmentations

# ds = tf.data.TFRecordDataset(files_train, num_parallel_reads=AUTO)
# ds = ds.take(1).cache().repeat()
# ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
# ds = ds.map(lambda img, target: (prepare_image(img, cfg=CFG, augment=True), target), 
#             num_parallel_calls=AUTO)
# ds = ds.take(12*5)
# ds = ds.prefetch(AUTO)

# show_dataset(64, 12, 5, ds)


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