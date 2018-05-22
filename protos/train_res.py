import sys
sys.path.append('/home/takamisato/works/ResFGB/')
from resfgb.models import ResFGB, get_hyperparams

import re
import pandas as pd
import numpy as np
from scipy import sparse
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
#import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm

from load_data import load_train_data, load_test_data
import sys
DIR = 'result_tmp/'  # sys.argv[1]  # 'result_1008_rate001/'
print(DIR)


def cst_metric_xgb(pred, dtrain):
    label = dtrain.get_label().astype(np.int)
    preds = pred.reshape((21, -1)).T
    preds = np.array([np.argmax(x) for x in preds], dtype=np.int)
    sc = log_loss(label, preds)
    return 'qwk', sc, True


def dummy(pred, dtrain):
    return 'dummy', pred, True


def callback(data):
    if (data.iteration + 1) % 10 != 0:
        return

    clf = data.model
    trn_data = clf.train_set
    val_data = clf.valid_sets[0]
    preds = [ele[2] for ele in clf.eval_valid(dummy) if ele[1] == 'dummy'][0]
    preds = preds.reshape((21, -1)).T
    preds = np.array([np.argmax(x) for x in preds], dtype=np.int)
    labels = val_data.get_label().astype(np.int)
    sc = log_loss(labels, preds)
    sc2 = roc_auc_score(labels, preds)
    logger.info('cal [{}] {} {}'.format(data.iteration + 1, sc, sc2))


from scipy import sparse


def train():

    # df = load_train_data()  # .sample(10000000, random_state=42).reset_index(drop=True)
    df = pd.read_csv('train2.csv', parse_dates=['activation_date'])
    df["weekday"] = df['activation_date'].dt.weekday
    train = df['activation_date'] < '2017-03-25'
    test = df['activation_date'] >= '2017-03-25'

    with open('../fasttext/fast_max_train_title.pkl', 'rb') as f:
        fast_data = np.array(pickle.load(f), dtype='float32')
    fast_max_data_title = pd.DataFrame(fast_data, columns=[f'fast_title_{i}' for i in range(fast_data.shape[1])])
    df = pd.concat([df,
                    fast_max_data_title,
                    # nn_data,
                    # img_data
                    ], axis=1)
    y_train = (df['deal_probability'].values > 0.5).astype('int32')

    df = df.drop(['deal_probability', 'activation_date', 'item_id'], axis=1).fillna(-1)
    x_train = df.values.astype('float32')

    logger.info('train data size {}'.format(x_train.shape))

    usecols = df.columns.values.tolist()
    #usecols = list(range(x_train.shape[1]))

    with open(DIR + 'usecols.pkl', 'wb') as f:
        pickle.dump(usecols, f, -1)

    # 2018-05-19 00:32:28,463 root 157 [INFO][train] best score: 0.22563737033010078
    # fast title 0.22455
    # {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_bin': 255, 'max_depth': -1, 'metric': 'rmse', 'min_child_weight': 5, 'min_split_gain': 0.0001, 'num_leaves': 255, 'objective': 'regression_l2', 'reg_alpha': 0, 'scale_pos_weight': 1, 'seed': 114, 'subsample': 1, 'subsample_freq': 1, 'verbose': -1}
    all_params = {'min_child_weight': [5],
                  'subsample': [1],
                  'subsample_freq': [1],
                  'seed': [114],
                  'colsample_bytree': [0.7],
                  'learning_rate': [0.1],
                  'max_depth': [-1],
                  'min_split_gain': [0.0001],
                  'reg_alpha': [0.1, 0],
                  'max_bin': [255],
                  'num_leaves': [255],
                  'objective': ['regression_l2'],
                  'metric': ['rmse'],
                  'scale_pos_weight': [1],
                  'verbose': [-1],
                  #'device': ['gpu'],
                  }
    use_score = 0
    min_score = (100, 100, 100)
    # for params in tqdm(list(ParameterGrid(all_params))):
    if 1:
        cnt = -1
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        # for train, test in cv.split(x_train, y_train):
        if 1:
            cnt += 1
            trn_x = x_train[[i for i in range(x_train.shape[0]) if train[i]]]
            val_x = x_train[[i for i in range(x_train.shape[0]) if test[i]]]
            trn_y = y_train[train]
            val_y = y_train[test]
            params = get_hyperparams(trn_x.shape[0], trn_x.shape[1], 1)
            clf = ResFGB(**params)
            best_iters, _, _ = clf.fit(trn_x, trn_y, val_x, val_y, use_best_iter=True)
            pred = clf.predict(val_x)

            #all_pred[test] = pred

            _score = np.sqrt(mean_squared_error(val_y, pred))
            _score2 = _score  # - roc_auc_score(val_y, pred)

            logger.info('   _score: %s' % _score)
            logger.info('   _score2: %s' % _score2)

            list_score.append(_score)
            list_score2.append(_score2)

            if clf.best_iteration != 0:
                list_best_iter.append(best_iters)
            else:
                list_best_iter.append(params['n_estimators'])

            with open(DIR + 'train_cv_pred_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(pred, f, -1)
            with open(DIR + 'model_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(clf, f, -1)
            gc.collect()
        with open(DIR + 'train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
        #trees = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        logger.info('param: %s' % (params))
        logger.info('cv: {})'.format(list_score))
        logger.info('cv2: {})'.format(list_score2))

        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('qwk: {} (avg min max {})'.format(score2[use_score], score2))

        if min_score[use_score] > score[use_score]:
            min_score = score
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))

        logger.info('best params: {}'.format(min_params))

    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv(DIR + 'feature_importances_0.csv')
    logger.info('imp use {} {}'.format(imp[imp.imp > 0].shape, n_features))

    del val_x
    del trn_y
    del val_y
    del train_data
    del test_data
    gc.collect()

    trees = np.mean(list_best_iter)

    logger.info('all data size {}'.format(x_train.shape))

    train_data = lgb.Dataset(x_train,
                             label=y_train,
                             feature_name=usecols
                             )
    del x_train
    gc.collect()
    logger.info('train start')
    clf = lgb.train(min_params,
                    train_data,
                    int(trees * 1.1),
                    valid_sets=[train_data],
                    verbose_eval=10
                    )
    logger.info('train end')
    with open(DIR + 'model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    #del x_train
    gc.collect()

    logger.info('save end')


def predict():
    with open(DIR + 'model.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open(DIR + 'usecols.pkl', 'rb') as f:
        usecols = pickle.load(f)

    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv(DIR + 'feature_importances.csv')
    logger.info('imp use {} {}'.format(imp[imp.imp > 0].shape, n_features))

    #df = load_test_data()
    df = pd.read_csv('test2.csv', parse_dates=['activation_date'])
    df["weekday"] = df['activation_date'].dt.weekday

    # with open('nn_test.pkl', 'rb') as f:
    #    _nn_data = pickle.load(f)
    #nn_data = pd.DataFrame(_nn_data, columns=[f'nn_{i}' for i in range(_nn_data.shape[1])])

    # with open('nn_test_chargram.pkl', 'rb') as f:
    #    _nn_data = pickle.load(f)
    #nn_data_chargram = pd.DataFrame(_nn_data, columns=[f'nn_chargram_{i}' for i in range(_nn_data.shape[1])])

    # with open('../fasttext/fast_max_test_title.pkl', 'rb') as f:
    #    fast_data = np.array(pickle.load(f), dtype='float32')
    #fast_max_data_title = pd.DataFrame(fast_data, columns=[f'fast_title_{i}' for i in range(fast_data.shape[1])])
    # with open('../fasttext/fast_max_test_desc.pkl', 'rb') as f:
    #    fast_data = np.array(pickle.load(f), dtype='float32')
    #fast_max_data_desc = pd.DataFrame(fast_data, columns=[f'fast_desc_{i}' for i in range(fast_data.shape[1])])
    df = pd.concat([df,
                    # fast_max_data_title,
                    # nn_data,
                    # img_data
                    ], axis=1)
    with open('test_nn.pkl', 'rb') as f:
        tfidf = pickle.load(f).tocsc()
        cols = pd.read_csv('tfidf_cols.csv')['col'].values
        tfidf = tfidf[:, cols].tocsr()

    logger.info('data size {}'.format(df.shape))

    # for col in usecols:
    #    if col not in df.columns.values:
    #        df[col] = np.zeros(df.shape[0])
    #        logger.info('no col %s' % col)

    x_test = df[[col for col in usecols if 'tfidf' not in col]]
    x_test = sparse.hstack([x_test.values.astype('float32'), tfidf], format='csr')

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

    train()
    # train2()
    predict()
