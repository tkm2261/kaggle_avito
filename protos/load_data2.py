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

TRAIN_DATA_PATH = '../data/dmt_train_0526/'
TEST_DATA_PATH = '../data/dmt_test_0526/'


def read_csv(filename):
    logger.info(filename)
    df = pd.read_csv(filename, parse_dates=['t_activation_date'])
    df.drop(['t_image', 'p_item_id', 'i_item_id', 'u_user_id', 'c_category_name',
             'ct_city', 'isn_item_seq_number', 'pc_parent_category_name',
             'r_region', 'ut_user_type', 'it1_image_top_1'],
            axis=1, inplace=True, errors='ignore')
    
    df.sort_values('t_data_id', inplace=True)
    return df


def _run_col(df_col):
    col = df_col.name
    enc = LabelEncoder()
    enc.fit(df_col.astype(str).unique().tolist() + ['NULL'])
    logger.info(f'{col}: {len(enc.classes_)}')
    return (col, enc)


def _map_data(df_col):
    col = df_col.name
    cnt = Counter(df_col.astype(str).values.tolist())
    map_2id = {}
    for i, (key, n) in enumerate(cnt.most_common()):
        if n == 1:
            break
        map_2id[key] = i
    logger.info(f'{col}: {len(map_2id)}')
    return (col, map_2id)


def run_col(df_col):
    col = df_col.name
    if col in ['t_item_id', 't_user_id']:
        return _map_data(df_col)
    else:
        return _run_col(df_col)


def make_map_encoder():
    logger.info('enter')
    paths = sorted(glob.glob(TRAIN_DATA_PATH + '*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)

    map_encoder = {}

    p = Pool()
    tmp = p.map(run_col, tqdm([df[c] for c in df.columns.values if df[c].dtype == object]), chunksize=3)
    map_encoder = dict(tmp)
    p.close()
    p.join()
    with open('map_encoder.pkl', 'wb') as f:
        pickle.dump(map_encoder, f, -1)

    logger.info('exit')


def _proc_cat(col, df, enc):
    logger.info(f'start: {col}')    
    df_col = df.copy()
    df_col[col] = df[col].fillna('NULL')
    try:
        df_col[col] = enc.transform(df_col[col].astype(str))
    except ValueError as e:
        logger.info(f'no labels found: {col}' + e.args[0])
        df_col.loc[~df_col[col].isin(enc.classes_), col] = 'NULL'
        df_col[col] = enc.transform(df_col[col].astype(str))
    return col, df_col


def _proc_map(col, df, enc):
    df_col = df.copy()
    df_col[col] = df[col].fillna('NULL')
    df_col[col] = [enc.get(ele, -1) for ele in df[col].values]
    return col, df_col


def _proc_data(col, df, enc):
    logger.info(col)
    if col in ['t_item_id', 't_user_id']:
        return _proc_map(col, df, enc)
    else:
        return _proc_cat(col, df, enc)


def load_train_data():
    logger.info('enter')
    paths = sorted(glob.glob(TRAIN_DATA_PATH + '*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)

    with open('map_encoder.pkl', 'rb') as f:
        map_encoder = pickle.load(f)
    list_proc_data = []
    p = Pool()
    for col in map_encoder:
        if df[col].dtype == object:
            enc = map_encoder[col]
            list_proc_data.append(p.apply_async(_proc_data, (col, df[[col]], enc)))
    list_proc_data = [pr.get() for pr in list_proc_data]
    p.close()
    p.join()

    logger.info('merge start')
    cols = np.hstack([data[1].columns.values for data in list_proc_data])
    datas = np.hstack([data[1].values for data in list_proc_data])
    tmp = pd.DataFrame(datas, columns=cols)
    logger.info('set end')
    df.drop([data[0] for data in list_proc_data], axis=1, inplace=True)

    df = pd.concat([df, tmp], axis=1)

    df.sort_values('t_data_id', inplace=True)
    df.drop('t_data_id', axis=1, inplace=True)

    logger.info(f'data size: {df.shape}')
    logger.info('merge end')
    logger.info('exit')

    return df


def load_test_data():
    logger.info('enter')
    paths = sorted(glob.glob(TEST_DATA_PATH + '*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)

    with open('map_encoder.pkl', 'rb') as f:
        map_encoder = pickle.load(f)
    list_proc_data = []

    p = Pool()
    for col in map_encoder:
        if df[col].dtype == object:
            enc = map_encoder[col]
            list_proc_data.append(p.apply_async(_proc_data, (col, df[[col]], enc)))
    list_proc_data = [pr.get() for pr in list_proc_data]
    p.close()
    p.join()

    logger.info('merge start')
    cols = np.hstack([data[1].columns.values for data in list_proc_data])
    datas = np.hstack([data[1].values for data in list_proc_data])
    tmp = pd.DataFrame(datas, columns=cols)
    logger.info('set end')
    df.drop([data[0] for data in list_proc_data], axis=1, inplace=True)

    df = pd.concat([df, tmp], axis=1)
    df.sort_values('t_data_id', inplace=True)
    df.drop('t_data_id', axis=1, inplace=True)

    logger.info(f'data size: {df.shape}')
    logger.info('merge end')
    logger.info('exit')

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

    make_map_encoder()
    # print(load_train_data().head())
    load_train_data().to_csv('train_0526.csv', index=False)
    load_test_data().to_csv('test_0526.csv', index=False)
