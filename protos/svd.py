import pandas as pd
import numpy as np
from scipy import sparse
import pickle
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

from load_data import load_train_data, load_test_data
import sys
DIR = 'result_tmp/'  # sys.argv[1]  # 'result_1008_rate001/'
DTYPE = 'float32'
print(DIR)
print(DTYPE)


from scipy import sparse


def train():
    n_dim = 64
    with open('train_tfidf.pkl', 'rb') as f:
        tfidf_title = pickle.load(f)  # .tocsc()
        cols = pd.read_csv('tfidf_cols5.csv')['col'].values
        tfidf_title = tfidf_title[:, cols].tocsc()
    logger.info(f'load tfidf_data {tfidf_title.shape}')

    img_svd = TruncatedSVD(n_dim, random_state=0)
    img_data = pd.DataFrame(img_svd.fit_transform(tfidf_title), columns=[f'tfidf_svd_{i}' for i in range(n_dim)])
    with open('img_svd.pkl', 'wb') as f:
        pickle.dump(img_svd, f, -1)
    img_data.to_feather(f'train_tfidf_svd_{n_dim}.ftr')

    logger.info('train_end')
    with open('test_tfidf.pkl', 'rb') as f:
        tfidf_title = pickle.load(f)  # .tocsc()
        cols = pd.read_csv('tfidf_cols5.csv')['col'].values
        tfidf_title = tfidf_title[:, cols].tocsr()
    img_data = pd.DataFrame(img_svd.fit_transform(tfidf_title), columns=[f'tfidf_svd_{i}' for i in range(n_dim)])
    img_data.to_feather(f'test_tfidf_svd_{n_dim}.ftr')


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    train()
