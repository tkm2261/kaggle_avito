import pickle
import pandas as pd
import numpy as np
import glob
import pickle
import re
import gc
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool

from logging import getLogger
logger = getLogger(__name__)


def read_csv(filename):
    logger.info(filename)
    df = pd.read_csv(filename, parse_dates=['activation_date'])
    return df


def load_data(path):
    logger.info('enter')
    df = read_csv(path)
    df['data_id'] = np.arange(df.shape[0])
    df.drop(['title', 'description'], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger, NullHandler
    logger = getLogger()

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler('load_data.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    for path in ['../input/train.csv', '../input/test.csv', '../input/train_active.csv', '../input/test_active.csv']:
        load_data(path).to_csv(path + '.notext.csv', index=False)
