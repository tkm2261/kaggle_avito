import re
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, log_loss, mean_squared_error
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm


DIR = 'ens_tmp/'

from numba import jit

#a = set(path.split('/')[0] + '/' for path in glob.glob('result_06*/train_cv_tmp.pkl'))
#b = set(path.split('/')[0] + '/' for path in glob.glob('result_06*/test_tmp_pred.pkl'))


DIRS = [
    'result_0623_external_dart/',
    'result_0620_dart_rate001/',
    'result_0618_tfidfall/',
    'result_0619_dart_newdata/',
    'result_0616_dart/',
    'result_0618_check/',
    'result_0622_nndata/',
    'result_0618_tfidf_mat/',
    'result_0622_external/',
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

index = np.loadtxt('index.npy').astype(int)


def activation(x):
    return np.log1p(x)


def deactivation(x):
    return np.expm1(x)


def load_pred(path):
    with open(path + 'train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)[index]
    return activation(pred)


def load_test(path):
    with open(path + 'test_tmp_pred.pkl', 'rb') as f:
        pred = pickle.load(f)
    return activation(pred)


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def optimize(list_path, score_func, trials):
    space = {}
    for i, path in enumerate(list_path):
        # space[i] = hp.choice(path, [True, False])
        space[i] = hp.quniform(path, 0, 1, 0.01)
    np.random.seed(114514)
    min_params = None
    min_score = 100
    for i in range(10):
        trials = Trials()
        best = fmin(score_func, space, algo=tpe.suggest, trials=trials, max_evals=1000)
        sc = score_func({i: best[path] for i, path in enumerate(list_path)})
        if min_score > sc:
            min_score = sc
            min_params = best
        logger.warn('attempt %s: %s' % (i + 1, sc))
    return min_params


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('WARN')
    handler.setFormatter(log_fmt)
    logger.setLevel('WARN')
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'ens.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    list_dir = DIRS  # [DIR1, DIR2, DIR3, DIR4]

    list_preds = [load_pred(path) for path in list_dir]
    y_train = np.loadtxt('y_train.npy')[index]

    for i, path in enumerate(list_dir):
        pp = list_preds[i]
        logger.info('{}'.format((i, path, np.sqrt(mean_squared_error(y_train, pp.clip(0, 1))))))

    def score_func(params):
        # use_preds = [pred for i, pred in enumerate(list_preds) if params[i]]
        use_preds = [params[i] * pred for i, pred in enumerate(list_preds)]
        if len(use_preds) == 0:
            return 100

        # pred = np.mean(use_preds, axis=0)
        pred = np.sum(use_preds, axis=0)
        pred /= sum(params.values())
        pred = deactivation(pred)

        sc = np.sqrt(mean_squared_error(y_train, pred.clip(0, 1)))
        return sc

    def pred_func(params):
        # use_preds = [pred for i, pred in enumerate(list_preds) if params[i]]
        use_preds = [params[i] * pred for i, pred in enumerate(list_preds)]
        if len(use_preds) == 0:
            return 100

        # pred = np.mean(use_preds, axis=0)
        pred = np.sum(use_preds, axis=0)
        pred /= sum(params.values())

        return pred.clip(0, 1)

    def predict(params):
        # use_preds = [pred for i, pred in enumerate(list_preds) if params[i]]
        use_preds = [params[path] * load_test(path) for path in list_dir]
        # pred = np.mean(use_preds, axis=0)
        pred = np.sum(use_preds, axis=0)
        pred /= sum(params.values())
        pred = deactivation(pred)
        return pred.clip(0, 1)

    trials = Trials()
    min_params = optimize(list_dir, score_func, trials)
    logger.info(f'min params: {min_params}')
    preds = pred_func({i: min_params[path] for i, path in enumerate(list_dir)})

    sc = np.sqrt(mean_squared_error(y_train, preds))
    logger.warn('search: %s' % sc)

    list_test = [load_test(path) for path in list_dir]
    p_test = predict(min_params)

    sub = pd.DataFrame()
    sub['item_id'] = pd.read_csv('../input/test.csv', usecols=['item_id'])['item_id'].values
    sub['deal_probability'] = p_test.clip(0, 1)
    sub.to_csv(DIR + 'submit_ens.csv', index=False)

    logger.info('exit')
