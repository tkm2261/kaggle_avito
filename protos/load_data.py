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

TRAIN_DATA_PATH = '../input/train.csv'
TEST_DATA_PATH = '../input/test.csv'

NOT_CAT_COLS = ['title', 'description', 'image', 'activation_date']


def read_csv(filename):
    logger.info(filename)
    df = pd.read_csv(filename, parse_dates=['activation_date'])

    # Feature Engineering
    df['text_feat'] = (df['param_1'] + ' ' +
                       df['param_2'] + ' ' +
                       df['param_3'])

    # Meta Text Features
    textfeats = ["description", "text_feat", "title"]
    for cols in textfeats:
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].astype(str).fillna('')  # FILL NA
        df[cols] = df[cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_chars'] = df[cols].apply(len)  # Count number of Characters
        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split()))
        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols + '_num_unique_words'] / df[cols + '_num_words'] * 100

    # df.drop([], inplace=True, axis=1)
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
    if col in ['item_id', 'user_id']:
        return _map_data(df_col)
    else:
        return _run_col(df_col)


def make_map_encoder():
    logger.info('enter')
    df = read_csv(TRAIN_DATA_PATH)

    df.drop(NOT_CAT_COLS, axis=1, inplace=True)
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
    if col in ['item_id', 'user_id']:
        return _proc_map(col, df, enc)
    else:
        return _proc_cat(col, df, enc)


def load_train_data():
    logger.info('enter')
    df = read_csv(TRAIN_DATA_PATH)
    period = pd.read_csv('../input/periods_train.csv', parse_dates=['activation_date', 'date_from', 'date_to'])
    period['activation_dur'] = (period.date_to - period.date_from).dt.days
    period['activation_dur2'] = (period.date_from - period.activation_date).dt.days
    df = df.merge(period[['item_id', 'activation_date', 'activation_dur', 'activation_dur2']],
                  how='left', on=['item_id', 'activation_date'])

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

    df.drop(['title', 'description', 'image'], axis=1, inplace=True)

    df = pd.concat([df, tmp], axis=1)
    logger.info('merge end')
    logger.info('exit')

    return df


def load_test_data():
    logger.info('enter')
    df = read_csv(TEST_DATA_PATH)
    period = pd.read_csv('../input/periods_test.csv', parse_dates=['activation_date', 'date_from', 'date_to'])
    period['activation_dur'] = (period.date_to - period.date_from).dt.days
    period['activation_dur2'] = (period.date_from - period.activation_date).dt.days
    df = df.merge(period[['item_id', 'activation_date', 'activation_dur', 'activation_dur2']],
                  how='left', on=['item_id', 'activation_date'])
    with open('map_encoder.pkl', 'rb') as f:
        map_encoder = pickle.load(f)
    list_proc_data = []
    col = 'item_id'
    enc = map_encoder[col]

    _proc_data(col, df[[col]], enc)

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
    df.drop(['title', 'description', 'image'], axis=1, inplace=True)

    df = pd.concat([df, tmp], axis=1)
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

    # make_map_encoder()
    # print(load_train_data().head())
    load_train_data().to_csv('train2.csv', index=False)
    load_test_data().to_csv('test2.csv', index=False)
