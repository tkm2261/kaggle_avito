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
from skimage import feature
from queue import Queue
import gc
import os
import operator
from collections import defaultdict
from PIL import Image
from scipy.stats import itemfreq
from numba import jit
n_channels = 3
im_dim = 92
limit = None  # Limit number of images processed (useful for debug)
bar_iterval = 10  # in seconds
empty_im = np.zeros((im_dim, im_dim, n_channels), dtype=np.uint8)  # Used when no image is present


def _color_analysis(img):
    # obtain the color palatte of the image
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1

    # sort the colors present in the image
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse=True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]):  # dull : too much darkness
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]):  # bright : too much whiteness
            light_shade += x[1]
        shade_count += x[1]

    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent


def perform_color_analysis(im):
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    light_percent1, dark_percent1 = _color_analysis(im1)
    light_percent2, dark_percent2 = _color_analysis(im2)

    light_percent = (light_percent1 + light_percent2)/2
    dark_percent = (dark_percent1 + dark_percent2)/2
    return dark_percent, light_percent


def get_average_pixel_width(im):
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100


def get_dominant_color(cv_img):
    arr = np.float32(cv_img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(cv_img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color


def get_blurrness_score(cv_img):
    image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


def proc(item_id):

    zfile = '../input/train_jpg/{}.jpg'.format(item_id)
    cv_img = cv2.imread(zfile)
    if cv_img is None:
        return [0 for _ in range(15)]
    else:
        img = Image.open(zfile)

        img_size = [img.size[0], img.size[1]]
        (means, stds) = cv2.meanStdDev(cv_img)
        mean_color = np.mean(cv_img.flatten())
        std_color = np.std(cv_img.flatten())
        color_stats = np.concatenate([means, stds]).flatten()

        size = os.path.getsize(zfile)

        light_percent, dark_percent = perform_color_analysis(img)
        average_pixel_width = get_average_pixel_width(img)
        #red_dom, green_dom, blue_dom = get_dominant_color(cv_img)
        blurrness = get_blurrness_score(cv_img)

        ret = img_size + [mean_color] + [std_color] + color_stats.tolist() + [size] + \
            [light_percent, dark_percent, average_pixel_width, blurrness]
        return ret


if __name__ == '__main__':
    n_items = Value('i', -1)  # Async number of items
    features = []
    # items_ids = []
    ids = pd.read_csv('../input/train.csv', usecols=['image'], nrows=limit)['image'].tolist()
    with Pool() as p:
        features = list(p.map(proc, tqdm(ids), chunksize=100))
    # features = [proc(i) for i in tqdm(ids)]  # list(map(proc, tqdm(ids)))
    print('Concating matrix...')
    features = np.vstack(features)
    print(features.shape)

    pd.DataFrame(features.astype('float32'), columns=['height', 'width', 'mean_color', 'std_color',
                                                      'color_stats_mean_r', 'color_stats_std_r',
                                                      'color_stats_mean_g', 'color_stats_std_g',
                                                      'color_stats_mean_b', 'color_stats_std_b',
                                                      'size', 'light_percent', 'dark_percent', 'average_pixel_width',
                                                      'blurrness']).to_feather('train_img_baseinfo_more.ftr')
    print('Saving matrix...')
    # with open('img_hist.pkl', 'wb') as f:
    #    pickle.dump(features, f, -1)

    #np.savetxt('train_img_hist.npy', features)
    #print('All done! Good luck')
