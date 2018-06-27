import pandas as pd
import numpy as np
from scipy import sparse
import pickle
import glob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
# import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm
import keras as ks

import sys
DIR = 'ens_tmp2/'  # sys.argv[1]  # 'result_1008_rate001/'
DTYPE = 'float32'
print(DIR)
print(DTYPE)

from keras import backend as K


def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def RMSE_C(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def get_model(n_dim):
    clf = ks.Sequential()
    clf.add(ks.layers.Dense(1, input_shape=(n_dim, )))
    # clf.add(ks.layers.Dropout(0.3))
    #clf.add(ks.layers.Dense(int(n_dim / 4) + 1))
    # clf.add(ks.layers.Dropout(0.3))
    # clf.add(ks.layers.Dense(1))

    clf.add(ks.layers.Activation('sigmoid'))
    clf.compile(optimizer=ks.optimizers.Adam(lr=0.1), loss='binary_crossentropy', metrics=[RMSE, RMSE_C])
    return clf


def train(x_train):

    # y_train = pd.read_feather('../protos/train_0618.ftr')['t_deal_probability'].values
    # np.savetxt('y_train.npy', y_train)
    y_train = np.loadtxt('y_train.npy')
    usecols = x_train.columns.values.tolist()

    cv = KFold(n_splits=5, shuffle=True, random_state=871)

    with open(DIR + 'usecols.pkl', 'wb') as f:
        pickle.dump(usecols, f, -1)
    for _, test in cv.split(x_train, y_train):
        x_train = x_train.iloc[test].values
        y_train = y_train[test]
        break

    # {'boosting_type': 'gbdt', 'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_bin': 255, 'max_depth': -1, 'metric': 'rmse', 'min_child_weight': 20, 'min_split_gain': 0.01, 'num_leaves': 31, 'objective': 'xentropy', 'reg_alpha': 1, 'scale_pos_weight': 1, 'seed': 114514, 'subsample': 1, 'subsample_freq': 0, 'verbose': -1}
    all_params = {'min_child_weight': [150],
                  'subsample': [1],
                  'subsample_freq': [0],
                  'seed': [114514],
                  'colsample_bytree': [0.8],
                  'learning_rate': [0.1],
                  'max_depth': [-1],
                  'min_split_gain': [0.01],
                  'reg_alpha': [1],
                  'max_bin': [255],
                  'num_leaves': [31],
                  'objective': ['xentropy'],
                  'scale_pos_weight': [1],
                  'verbose': [-1],
                  'boosting_type': ['gbdt'],
                  'metric': ['rmse'],
                  # 'skip_drop': [0.7],
                  }

    use_score = 0
    min_score = (100, 100, 100)
    cv = KFold(n_splits=3, shuffle=True, random_state=871)
    for params in tqdm(list(ParameterGrid(all_params))):
        cnt = -1
        list_score = []
        list_score2 = []
        list_itr = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv.split(x_train, y_train):
            cnt += 1
            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]
            val_y = y_train[test]

            gc.collect()
            metric = 'val_loss'
            mode = 'min'
            clbs = [ks.callbacks.EarlyStopping(monitor=metric,
                                               patience=10,
                                               verbose=1,
                                               min_delta=1e-6,
                                               mode=mode), ]
            clf = get_model(x_train.shape[1])
            clf.fit(trn_x, trn_y, batch_size=500000, epochs=10000000, validation_data=(val_x, val_y), callbacks=clbs)
            list_itr.append(clbs[0].stopped_epoch)
            pred = clf.predict(val_x).clip(0, 1)

            all_pred[test] = pred

            _score = np.sqrt(mean_squared_error(val_y, pred))
            _score2 = _score  # - roc_auc_score(val_y, pred)

            logger.info('   _score: %s' % _score)
            logger.info('   _score2: %s' % _score2)

            list_score.append(_score)
            list_score2.append(_score2)

            with open(DIR + 'train_cv_pred_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(pred, f, -1)
            with open(DIR + 'model_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(clf, f, -1)
            gc.collect()
        with open(DIR + 'train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        logger.info('param: %s' % (params))
        logger.info('cv: {})'.format(list_score))
        logger.info('cv2: {})'.format(list_score2))

        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('all loss: {}'.format(mean_squared_error(y_train, all_pred)))

        if min_score[use_score] > score[use_score]:
            min_score = score
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))

        logger.info('best params: {}'.format(min_params))
        logger.info('best iters: {}'.format(list_itr))

    logger.info('all data size {}'.format(x_train.shape))


def train2(x_train, epoch):
    logger.info('train start')
    y_train = np.loadtxt('y_train.npy')
    usecols = x_train.columns.values.tolist()
    clf = get_model(x_train.shape[1])

    clf.fit(x_train, y_train, batch_size=100000, epochs=epoch)

    logger.info('save end')
    return clf


def predict(x_test, clf):

    with open(DIR + 'usecols.pkl', 'rb') as f:
        usecols = pickle.load(f)

    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv(DIR + 'feature_importances.csv')
    logger.info('imp use {} {}'.format(imp[imp.imp > 0].shape, n_features))

    x_test = x_test[usecols]
    if x_test.shape[1] != n_features:
        raise Exception('Not match feature num: %s %s' % (x_test.shape[1], n_features))

    logger.info('test load end')

    p_test = clf.predict(x_test)
    with open(DIR + 'test_tmp_pred.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)

    logger.info('test save end')

    sub = pd.DataFrame()
    sub['item_id'] = pd.read_csv('../input/test.csv', usecols=['item_id'])['item_id'].values
    sub['deal_probability'] = p_test.clip(0, 1)
    sub.to_csv(DIR + 'submit.csv', index=False)
    logger.info('exit')


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    paths = [
        'result_0623_external_dart/',
        'result_0620_dart_rate001/',
        'result_0618_tfidfall/',
        'result_0625_teppei/',
        'result_0624_xgbdart/',
        'result_0619_dart_newdata/',
        'result_0616_dart/',
        'result_0618_check/',
        'result_0622_nndata/',
        'result_0618_tfidf_mat/',
        'result_0622_external/',
        'result_0624_teppei_white/',
        'result_0622_external2/',
        'result_0618_newdata_rate002/',
        'result_0615_xentropy/',
        'result_0619_newdata_tuned/',
        'result_0611_rate001/',
        'result_0615_exif/',
        'result_0612_newgroupby/',
        'result_0610_rate002/',
        'result_0611_teppei/',
        'result_0609_basemore/',
        'result_0618_newdata/',
        'result_0605_baseinfo/',
    ]

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for path in tqdm(paths):
        with open(path + '/train_cv_tmp.pkl', 'rb') as f:
            df_train[path] = pickle.load(f).clip(0, 1)
        with open(path + '/test_tmp_pred.pkl', 'rb') as f:
            df_test[path] = pickle.load(f).clip(0, 1)

    train(df_train)
    # clf = train2(df_train, 10000)
    # predict(df_test, clf)
