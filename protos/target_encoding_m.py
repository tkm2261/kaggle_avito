import re
import pandas as pd
import numpy as np
from scipy import sparse
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
# import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm

from load_data import load_train_data, load_test_data
import sys
DIR = 'target_tmp/'  # sys.argv[1]  # 'result_1008_rate001/'
print(DIR)


def proc(df_test, df, col):
    logger.info(f'target col: {col}')
    target_col = f'target_encoding_{col}'

    df_test[col] = df_test[col].fillna(-111).astype(np.int64)

    tmp = df.groupby(col)[['t_deal_probability']].mean().reset_index()
    tmp.columns = [col, target_col]

    return target_col, pd.merge(df_test, tmp, how='left', on=col)[target_col].values


def train():

    # df = load_train_data()  # .sample(10000000, random_state=42).reset_index(drop=True)
    # , parse_dates=['t_activation_date'], float_precision='float32')
    df = pd.read_feather('train_0612.ftr')  # , parse_dates=['t_activation_date'], float_precision='float32')
    #cols = [col for col in df if df[col].dtype != object and col not in ('t_data_id', 't_activation_date')]
    #df[cols] = df[cols].astype(DTYPE)
    gc.collect()
    logger.info(f'load 1 {df.shape}')
    y_train = df['t_deal_probability'].values
    '''
    #train, test = train_test_split(np.arange(df.shape[0]), test_size=0.1, random_state=42)
    df.drop(['t_activation_date', 't_item_id'] + ['i_sum_item_deal_probability', 'u_sum_user_deal_probability', 'isn_sum_isn_deal_probability', 'it1_sum_im1_deal_probability', 'pc_sum_pcat_deal_probability', 'ct_sum_city_deal_probability',
                                                  'c_sum_category_deal_probability', 'ut_sum_usertype_deal_probability', 'r_sum_region_deal_probability'] + ['i_avg_item_deal_probability', 'it1_avg_im1_deal_probability', 'u_avg_user_deal_probability'] + ['ui_avg_user_deal_probability', 'ir_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'uit_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'iit_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'uca_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'ic_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'uu_avg_user_deal_probability'

                                                                                                                                                                                                                                                              'ip_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'ica_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'ii_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'up_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'ur_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'uc_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'ip_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'uu_avg_user_deal_probability',
                                                                                                                                                                                                                                                              'iu_avg_user_deal_probability'
                                                                                                                                                                                                                                                              ], axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0601_useritemcols/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0609_basemore/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0611_rate001/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)
    logger.info(f'load dropcols {df.shape}')
    gc.collect()

    cv = KFold(n_splits=5, shuffle=True, random_state=871)

    df_ret = pd.DataFrame()
    for col in tqdm(df.columns):
        logger.info(f'target col: {col} {df[col].dtype}')
        if col == 't_deal_probability':
            continue
        df[col] = df[col].fillna(-111).astype(np.int64)
        uq_num = len(set(df[col].values.tolist()))
        logger.info(f'uq rate col: {uq_num} {uq_num / df.shape[0]}')

        target_col = f'target_encoding_{col}'
        logger.info(f'target col: {col}')
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv.split(df, y_train):
            trn_x = df.iloc[train]  # [[i for i in range(x_train.shape[0]) if train[i]]]
            val_x = df.iloc[test]  # [[i for i in range(x_train.shape[0]) if test[i]]]

            tmp = trn_x.groupby(col)[['t_deal_probability']].mean().reset_index()
            tmp.columns = [col, target_col]
            all_pred[test] = pd.merge(val_x, tmp, how='left', on=col)[target_col].values
        df_ret[target_col] = all_pred

    logger.info(f'target encoding size: {df_ret.shape}')
    print(df_ret.head())
    df_ret = df_ret.reset_index(drop=True)
    df_ret.to_feather(DIR + 'train_target_enc.ftr')
    logger.info('end train')
    '''
    df_test = pd.read_feather('test_0612.ftr')  # , parse_dates=['t_activation_date'], float_precision='float32')
    df_ret_test = pd.DataFrame()
    df_ret = pd.read_feather(DIR + 'train_target_enc.ftr')

    list_p = []
    with Pool() as p:
        for col in tqdm(df_ret.columns):
            col = col.replace('target_encoding_', '')
            list_p.append(p.apply_async(proc, (df_test[[col]], df[[col, 't_deal_probability']], col)))
        list_p = [pp.get() for pp in tqdm(list_p)]
    for target_col, val in list_p:
        df_ret_test[target_col] = val

    df_ret_test.to_feather(DIR + 'test_target_enc.ftr')


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'target_encoding.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    train()
    predict()
