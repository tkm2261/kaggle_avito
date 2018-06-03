import re
import pandas as pd
import numpy as np
from scipy import sparse
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import Ridge
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_squared_error
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm

from load_data import load_train_data, load_test_data
import sys
DIR = 'result_ridge_tmp/'  # sys.argv[1]  # 'result_1008_rate001/'
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

    # df = load_train_data()
    df = pd.read_csv('train_0526.csv', parse_dates=['t_activation_date'])

    train = df['t_activation_date'] < '2017-03-26'
    test = df['t_activation_date'] >= '2017-03-26'
    # train, test = train_test_split(np.arange(df.shape[0]), test_size=0.1, random_state=42)

    tx_data = pd.read_csv('train2.csv')
    tx_data = tx_data[[col for col in tx_data if "description" in col or "text_feat" in col or "title" in col]
                      ]
    # img_data = sparse.load_npz('features.npz').todense()
    # img_data = pd.DataFrame(img_data, columns=[f'vgg16_{i}' for i in range(img_data.shape[1])])
    # with open('nn_train.pkl', 'rb') as f:
    #    _nn_data = pickle.load(f)
    # nn_data = pd.DataFrame(_nn_data, columns=[f'nn_{i}' for i in range(_nn_data.shape[1])])

    with open('train_tfidf.pkl', 'rb') as f:
        tfidf_title = pickle.load(f)  # .tocsc()
        cols = pd.read_csv('tfidf_cols4.csv')['col'].values
        tfidf_title = tfidf_title[:, cols].tocsr()
    """
    with open('train_tfidf_desc.pkl', 'rb') as f:
        tfidf = pickle.load(f)  # .tocsc()
        cols = pd.read_csv('tfidf_desc_cols.csv')['col'].values
        tfidf_desc = tfidf[:, cols].tocsr()
    """
    # with open('nn_train_chargram.pkl', 'rb') as f:
    #    _nn_data = pickle.load(f)
    # nn_data_chargram = pd.DataFrame(_nn_data, columns=[f'nn_chargram_{i}' for i in range(_nn_data.shape[1])])

    # with open('../fasttext/fast_max_train_title.pkl', 'rb') as f:
    #    fast_data = np.array(pickle.load(f), dtype='float32')
    # fast_max_data_title = pd.DataFrame(fast_data, columns=[f'fast_title_{i}' for i in range(fast_data.shape[1])])
    # with open('../fasttext/fast_max_train_desc.pkl', 'rb') as f:
    #    fast_data = np.array(pickle.load(f), dtype='float32')
    # fast_max_data_desc = pd.DataFrame(fast_data, columns=[f'fast_desc_{i}' for i in range(fast_data.shape[1])])
    df = pd.concat([df,
                    tx_data,
                    # fast_max_data_title,
                    # nn_data,
                    # img_data
                    ], axis=1).fillna(-1)
    y_train = df['t_deal_probability'].values

    df = df.drop(['t_deal_probability', 't_activation_date', 't_item_id'] + ['i_sum_item_deal_probability', 'u_sum_user_deal_probability', 'isn_sum_isn_deal_probability', 'it1_sum_im1_deal_probability', 'pc_sum_pcat_deal_probability', 'ct_sum_city_deal_probability', 'c_sum_category_deal_probability', 'ut_sum_usertype_deal_probability', 'r_sum_region_deal_probability'] + ['i_avg_item_deal_probability',
                                                                                                                                                                                                                                                                                                                                                                                      'it1_avg_im1_deal_probability',
                                                                                                                                                                                                                                                                                                                                                                                      'u_avg_user_deal_probability'], axis=1)
    """
    x_train = df
    x_train = sparse.hstack([x_train.values.astype('float32'),
                             tfidf_title,
                             # tfidf_desc
                             ], format='csr')
    """
    x_train = tfidf_title
    logger.info('train data size {}'.format(x_train.shape))
    cv = KFold(n_splits=5, shuffle=True, random_state=871)

    usecols = df.columns.values.tolist() + [f'tfidf_title_{i}' for i in range(tfidf_title.shape[1])]
    #          + [f'tfidf_desc_{i}' for i in range(tfidf_desc.shape[1])]
    # usecols = list(range(x_train.shape[1]))

    with open(DIR + 'usecols.pkl', 'wb') as f:
        pickle.dump(usecols, f, -1)

    _params = {
        'alpha': 20.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
        'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': 114
    }
    all_params = {p: [v] for p, v in _params.items()}
    use_score = 0
    min_score = (100, 100, 100)
    for params in tqdm(list(ParameterGrid(all_params))):
        cnt = -1
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv.split(x_train, y_train):
            cnt += 1
            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]
            val_y = y_train[test]

            clf = Ridge(**params)
            clf.fit(trn_x, trn_y)
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

        logger.info('trees: {}'.format(list_best_iter))
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

    logger.info('all data size {}'.format(x_train.shape))

    logger.info('train start')
    clf = Ridge(**min_params)

    clf.fit(x_train, y_train)
    logger.info('train end')
    with open(DIR + 'model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    # del x_train
    gc.collect()

    logger.info('save end')


def predict():
    with open(DIR + 'model.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open(DIR + 'usecols.pkl', 'rb') as f:
        usecols = pickle.load(f)
    # df = load_test_data()
    df = pd.read_csv('test_0526.csv', parse_dates=['t_activation_date'])
    tx_data = pd.read_csv('test2.csv')
    tx_data = tx_data[[col for col in tx_data if "description" in col or "text_feat" in col or "title" in col]]

    # with open('nn_test.pkl', 'rb') as f:
    #    _nn_data = pickle.load(f)
    # nn_data = pd.DataFrame(_nn_data, columns=[f'nn_{i}' for i in range(_nn_data.shape[1])])

    # with open('nn_test_chargram.pkl', 'rb') as f:
    #    _nn_data = pickle.load(f)
    # nn_data_chargram = pd.DataFrame(_nn_data, columns=[f'nn_chargram_{i}' for i in range(_nn_data.shape[1])])

    # with open('../fasttext/fast_max_test_title.pkl', 'rb') as f:
    #    fast_data = np.array(pickle.load(f), dtype='float32')
    # fast_max_data_title = pd.DataFrame(fast_data, columns=[f'fast_title_{i}' for i in range(fast_data.shape[1])])
    # with open('../fasttext/fast_max_test_desc.pkl', 'rb') as f:
    #    fast_data = np.array(pickle.load(f), dtype='float32')
    # fast_max_data_desc = pd.DataFrame(fast_data, columns=[f'fast_desc_{i}' for i in range(fast_data.shape[1])])
    df = pd.concat([df,
                    tx_data,
                    # fast_max_data_title,
                    # nn_data,
                    # img_data
                    ], axis=1).fillna(-1)
    with open('test_tfidf.pkl', 'rb') as f:
        tfidf_title = pickle.load(f)  # .tocsc()
        cols = pd.read_csv('tfidf_cols4.csv')['col'].values
        tfidf_title = tfidf_title[:, cols].tocsr()

    logger.info('data size {}'.format(df.shape))

    x_test = df[[col for col in usecols if 'tfidf' not in col]]
    x_test = tfidf_title  # sparse.hstack([x_test.values.astype('float32'), tfidf_title], format='csr')

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

    # train()
    # train2()
    predict()
