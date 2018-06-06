import numpy as np
import pickle
import pandas as pd
from math import ceil
import cv2
import zipfile
from tqdm import tqdm
from multiprocessing import Value, Pool
import gzip
from pathlib import PurePath
from scipy import sparse
from threading import Thread
from queue import Queue
import gc
import os
from PIL import Image

fname = '/home/takamisato/hdd/works/kaggle_avito/input/train_jpg.zip'
n_channels = 3
im_dim = 92
limit = None  # Limit number of images processed (useful for debug)
bar_iterval = 10  # in seconds
empty_im = np.zeros((im_dim, im_dim, n_channels), dtype=np.uint8)  # Used when no image is present


def proc(item_id):

    zfile = '../input/train_jpg/{}.jpg'.format(item_id)
    cv_img = cv2.imread(zfile)
    if cv_img is None:
        height = 0
        width = 0
        size = 0
        return [0 for _ in range(11)]
    else:
        img = Image.open(zfile)
        img_size = [img.size[0], img.size[1]]
        (means, stds) = cv2.meanStdDev(cv_img)
        mean_color = np.mean(cv_img.flatten())
        std_color = np.std(cv_img.flatten())
        color_stats = np.concatenate([means, stds]).flatten()
        img_size + [mean_color] + [std_color] + color_stats.tolist()
        size = os.path.getsize(zfile)
        ret = img_size + [mean_color] + [std_color] + color_stats.tolist() + [size]
        return ret


if __name__ == '__main__':
    n_items = Value('i', -1)  # Async number of items
    features = []
    # items_ids = []
    ids = pd.read_csv('../input/train.csv', usecols=['image'], nrows=limit)['image'].tolist()
    with Pool() as p:
        features = list(p.map(proc, tqdm(ids), chunksize=100))
    print('Concating matrix...')
    features = np.vstack(features)
    print(features.shape)
    pd.DataFrame(features, columns=[f'img_baseinfo_{i}' for i in range(
        features.shape[1])]).to_feather('train_img_baseinfo.ftr')
    print('Saving matrix...')
    # with open('img_hist.pkl', 'wb') as f:
    #    pickle.dump(features, f, -1)

    #np.savetxt('train_img_hist.npy', features)
    #print('All done! Good luck')
