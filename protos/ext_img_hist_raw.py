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

fname = '/home/takamisato/hdd/works/kaggle_avito/input/test_jpg.zip'
n_channels = 3
im_dim = 92
limit = None  # Limit number of images processed (useful for debug)
bar_iterval = 10  # in seconds
empty_im = np.zeros((im_dim, im_dim, n_channels), dtype=np.uint8)  # Used when no image is present


def proc(item_id):

    zfile = '../input/test_jpg/{}.jpg'.format(item_id)
    im = cv2.imread(zfile)
    if im is None:
        im = empty_im

    # Replace None with empty image
    b, g, r = im[:, :, 0], im[:, :, 1], im[:, :, 2]

    # 方法1(NumPyでヒストグラムの算出)
    hist_r, bins = np.histogram(r.ravel(), 256, [0, 256])
    hist_g, bins = np.histogram(g.ravel(), 256, [0, 256])
    hist_b, bins = np.histogram(b.ravel(), 256, [0, 256])
    if hist_r.sum() > 0:
        hist_r = hist_r.astype(float) / hist_r.sum()
    if hist_g.sum() > 0:
        hist_g = hist_g.astype(float) / hist_g.sum()
    if hist_b.sum() > 0:
        hist_b = hist_b.astype(float) / hist_b.sum()

    hist_rgb = np.hstack([hist_r, hist_g, hist_b])

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hists = []

    for (i, col) in enumerate(['Hue', 'Saturation', 'Brightness']):
        if col == 'Hue':
            hist = cv2.calcHist([hsv], [i], None, [180], [0, 180])[:, 0]
        else:
            hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])[:, 0]
        hist /= hist.sum()
        hists.append(hist)

    hist_hsv = np.hstack(hists)
    hist = np.hstack([hist_rgb, hist_hsv])
    return hist


if __name__ == '__main__':
    n_items = Value('i', -1)  # Async number of items
    features = []
    # items_ids = []
    ids = pd.read_csv('../input/test.csv', usecols=['image'], nrows=limit)['image'].tolist()
    with Pool() as p:
        features = list(p.map(proc, tqdm(ids), chunksize=100))
    print('Concating matrix...')
    features = np.vstack(features)
    print(features)
    pd.DataFrame(features, columns=[f'img_hist_{i}' for i in range(features.shape[1])]).to_feather('test_img_hist.ftr')
    print('Saving matrix...')
    # with open('img_hist.pkl', 'wb') as f:
    #    pickle.dump(features, f, -1)

    #np.savetxt('train_img_hist.npy', features)
    #print('All done! Good luck')
