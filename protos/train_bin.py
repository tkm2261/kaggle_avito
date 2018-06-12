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
from sklearn.metrics import mean_squared_error
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm

from load_data import load_train_data, load_test_data
import sys
DIR = 'result_tmp/'  # sys.argv[1]  # 'result_1008_rate001/'
print(DIR)

DTYPE = 'float32'


def cst_metric_xgb(pred, dtrain):
    label = dtrain.get_label().astype(np.int)
    sc1 = np.sqrt(mean_squared_error(label, pred))
    sc = 100  # np.sqrt(mean_squared_error(label, pred.clip(0, 1)))
    return 'rmse_clip', max(sc1, sc), False


def dummy(pred, dtrain):
    return 'dummy', pred, False


from scipy import sparse


def train():

    # df = load_train_data()  # .sample(10000000, random_state=42).reset_index(drop=True)
    df = pd.read_feather('train_0602.ftr')  # , parse_dates=['t_activation_date'], float_precision='float32')
    for col in df:
        if df[col].dtype != object and col not in ('t_data_id', 't_activation_date'):
            df[col] = df[col].astype(DTYPE)
    gc.collect()
    y_train_raw = df['t_deal_probability'].values
    y_train = df['t_deal_probability'].values != 0

    df.drop(['t_deal_probability'], axis=1, errors='ignore', inplace=True)

    train = df['t_activation_date'] < '2017-03-26'
    test = df['t_activation_date'] >= '2017-03-26'
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

    gc.collect()

    tx_data = pd.read_csv('train2.csv')
    tx_data = tx_data[[col for col in tx_data if "description" in col or "text_feat" in col or "title" in col]]
    # with open('result_0527_ridge/train_cv_tmp.pkl', 'rb') as f:
    #    df['ridge'] = pickle.load(f)  # .tocsc()

    #img_data = sparse.load_npz('features.npz').todense()
    #img_data = pd.DataFrame(img_data, columns=[f'vgg16_{i}' for i in range(img_data.shape[1])])
    # with open('nn_train.pkl', 'rb') as f:
    #    _nn_data = pickle.load(f)
    # nn_data = pd.DataFrame(_nn_data, columns=[f'nn_{i}' for i in range(_nn_data.shape[1])])

    with open('train_tfidf.pkl', 'rb') as f:
        tfidf_title = pickle.load(f)  # .tocsc()
        cols = pd.read_csv('tfidf_cols4.csv')['col'].values
        tfidf_title = tfidf_title[:, cols].tocsr()

    # with open('train_tfidf_desc.pkl', 'rb') as f:
    #    tfidf = pickle.load(f)  # .tocsc()
    #    cols = pd.read_csv('tfidf_desc_cols.csv')['col'].values
    #    tfidf_desc = tfidf[:, cols].tocsr()
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
                    pd.read_feather('train_img_baseinfo_more.ftr')
                    # fast_max_data_title,
                    # nn_data,
                    # img_data
                    ], axis=1).astype(DTYPE)
    del tx_data
    gc.collect()

    logger.info(f'train df size {df.shape}')
    cols = pd.read_csv('result_tf_0607/tfidf_cols2.csv')['col'].values
    """
    tfidf2 = sparse.hstack([sparse.load_npz('result_tf_0607/train_tfidf_matrix_title.npz'),
                            sparse.load_npz('result_tf_0607/train_tfidf_matrix_description.npz')
                            ], format='csr', dtype=DTYPE)[:, cols]
    """
    x_train = df.values

    x_train = sparse.hstack([x_train,
                             tfidf_title,
                             # tfidf2,
                             # tfidf_desc
                             ], format='csr', dtype=DTYPE)

    usecols = df.columns.values.tolist()
    usecols += [f'tfidf_title_{i}' for i in range(tfidf_title.shape[1])]
    #usecols += [f'tfidf2_{i}' for i in range(tfidf2.shape[1])]
    #          + [f'tfidf_desc_{i}' for i in range(tfidf_desc.shape[1])]
    # usecols = list(range(x_train.shape[1]))

    del df
    gc.collect()

    logger.info('train data size {}'.format(x_train.shape))
    cv = KFold(n_splits=5, shuffle=True, random_state=871)

    with open(DIR + 'usecols.pkl', 'wb') as f:
        pickle.dump(usecols, f, -1)

    # {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_bin': 255, 'max_depth': -1, 'metric': 'rmse', 'min_child_weight': 5, 'min_split_gain': 0, 'num_leaves': 255, 'objective': 'regression_l2', 'reg_alpha': 1, 'scale_pos_weight': 1, 'seed': 114, 'subsample': 1, 'subsample_freq': 1, 'verbose': -1}
    """
    all_params = {'min_child_weight': [5],
                  'subsample': [1],
                  'subsample_freq': [1],
                  'seed': [114],
                  'colsample_bytree': [0.7],
                  'learning_rate': [0.01],
                  'max_depth': [-1],
                  'min_split_gain': [0],
                  'reg_alpha': [1],
                  'max_bin': [255],
                  'num_leaves': [255],
                  'objective': ['regression_l2'],
                  'scale_pos_weight': [1],
                  'verbose': [-1],
                  # 'device': ['gpu'],
                  }
    """
    """
    _params = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_bin': 255, 'max_depth': -1, 'min_child_weight': 5, 'min_split_gain': 0,
               'num_leaves': 255, 'objective': 'regression_l2', 'reg_alpha': 1, 'scale_pos_weight': 1, 'seed': 114, 'subsample': 1, 'subsample_freq': 1, 'verbose': -1, 'metric': 'rmse'}
    """

    min_params = {
        'min_child_weight': 5,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        # 'max_depth': 15,
        'num_leaves': 300,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.85,
        # 'bagging_freq': 5,
        'learning_rate': 0.1,  # 0.02
        'metric': 'binary_logloss'
    }
    all_params = {p: [v] for p, v in min_params.items()}

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
            trn_x = x_train[train]  # [[i for i in range(x_train.shape[0]) if train[i]]]
            val_x = x_train[test]  # [[i for i in range(x_train.shape[0]) if test[i]]]
            trn_y = y_train[train]
            val_y = y_train[test]
            val_y_raw = y_train_raw[test]

            def callback(data):
                if (data.iteration + 1) % 10 != 0:
                    return

                clf = data.model
                val_data = clf.valid_sets[0]
                preds = [ele[2] for ele in clf.eval_valid(dummy) if ele[1] == 'dummy'][0]
                labels = val_y_raw
                sc = np.sqrt(mean_squared_error(labels, preds.clip(0, 1)))
                logger.info('cal [{}] {}'.format(data.iteration + 1, sc))

            train_data = lgb.Dataset(trn_x,  # .values.astype(np.float32),
                                     label=trn_y,
                                     feature_name=usecols
                                     )
            test_data = lgb.Dataset(val_x,  # .values.astype(np.float32),
                                    label=val_y,
                                    feature_name=usecols
                                    )
            del trn_x
            gc.collect()
            clf = lgb.train(params,
                            train_data,
                            100000,  # params['n_estimators'],
                            early_stopping_rounds=30,
                            valid_sets=[test_data],
                            # feval=cst_metric_xgb,
                            callbacks=[callback],
                            verbose_eval=10
                            )
            pred = clf.predict(val_x).clip(0, 1)

            all_pred[test] = pred

            _score = np.sqrt(mean_squared_error(val_y, pred))
            _score2 = _score  # - roc_auc_score(val_y, pred)

            logger.info('   _score: %s' % _score)
            logger.info('   _score2: %s' % _score2)

            list_score.append(_score)
            list_score2.append(_score2)

            if clf.best_iteration != 0:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])

            with open(DIR + 'train_cv_pred_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(pred, f, -1)
            with open(DIR + 'model_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(clf, f, -1)
            gc.collect()
            # break
        with open(DIR + 'train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
        # trees = np.mean(list_best_iter, dtype=int)
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
                    feval=cst_metric_xgb,
                    verbose_eval=10
                    )
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

    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv(DIR + 'feature_importances.csv')
    logger.info('imp use {} {}'.format(imp[imp.imp > 0].shape, n_features))

    # df = load_test_data()
    df = pd.read_feather('test_0602.ftr')  # , parse_dates=['t_activation_date'])
    tx_data = pd.read_csv('test2.csv')
    tx_data = tx_data[[col for col in tx_data if "description" in col or "text_feat" in col or "title" in col]]

    with open('result_0527_ridge/test_tmp_pred.pkl', 'rb') as f:
        df['ridge'] = pickle.load(f)  # .tocsc()

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

    #img_data = sparse.load_npz('features_test.npz').todense()
    #img_data = pd.DataFrame(img_data, columns=[f'vgg16_{i}' for i in range(img_data.shape[1])])
    df = pd.concat([df,
                    tx_data,
                    pd.read_feather('test_img_baseinfo_more.ftr')
                    # fast_max_data_title,
                    # nn_data,
                    # img_data
                    ], axis=1)

    with open('test_tfidf.pkl', 'rb') as f:
        tfidf_title = pickle.load(f)  # .tocsc()
        cols = pd.read_csv('tfidf_cols4.csv')['col'].values
        tfidf_title = tfidf_title[:, cols].tocsr()
    """
    cols = pd.read_csv('result_tf_0607/tfidf_cols.csv')['col'].values
    tfidf2 = sparse.hstack([sparse.load_npz('result_tf_0607/test_tfidf_matrix_title.npz'),
                            sparse.load_npz('result_tf_0607/test_tfidf_matrix_description.npz')
                            ], format='csr')[:, cols]
    """
    logger.info('data size {}'.format(df.shape))

    # for col in usecols:
    #    if col not in df.columns.values:
    #        df[col] = np.zeros(df.shape[0])
    #        logger.info('no col %s' % col)

    x_test = df[[col for col in usecols if 'tfidf' not in col]]
    x_test = sparse.hstack([x_test.values,
                            tfidf_title,
                            # tfidf2
                            ], format='csr', dtype=DTYPE)

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