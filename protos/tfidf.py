
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import pickle
from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

from logging import getLogger

logger = getLogger(None)
from nltk.corpus import stopwords
stops = set(stopwords.words("russian")) | set(['?', ',', '.', ';', ':', '"', "'", '(', ')', '[', ']'])


def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['title'] = df['title'].fillna('') + ' ' + df['parent_category_name'].fillna('')
    df['description'] = df['description'].fillna('')
    df['description'] = (df['description'].fillna('') + ' ' +
                         df['title'] + ' ' +
                         df['region'].fillna('') + ' ' +
                         df['city'].fillna('') + ' ' +
                         df['category_name'].fillna('') + ' ' +
                         df['param_1'].fillna('') + ' ' +
                         df['param_2'].fillna('') + ' ' +
                         df['param_3'].fillna('') + ' ' +
                         df['user_type'].fillna('')
                         )
    df['price'] = np.log1p(df['price'].fillna(-1) + 1)
    return df[['title', 'description', 'price', 'item_seq_number']]


def train():
    vectorizer = make_union(
        on_field('title', Tfidf(max_features=1000000, min_df=5,
                                # token_pattern='[\w\?,\.;:\(\)\[\]]+',
                                token_pattern='\w+',
                                stop_words=stops, ngram_range=(1, 2),
                                # lowercase=True,
                                # smooth_idf=False
                                )),
        on_field('description', Tfidf(max_features=1000000, min_df=5,
                                      lowercase=True,
                                      # token_pattern='[\w\?,\.;:\(\)\[\]]+',
                                      token_pattern='\w+',
                                      ngram_range=(1, 2),
                                      # stop_words=stops,
                                      # smooth_idf=False
                                      )
                 ),
        n_jobs=4)
    df = pd.DataFrame()
    list_size = []
    #
    #
    for path in ['../input/train.csv', '../input/test.csv', '../input/train_active.csv', '../input/test_active.csv']:
        _df = pd.read_csv(path,
                          usecols=['title',
                                   'description',
                                   'price', 'item_seq_number',
                                   'parent_category_name',
                                   'region', 'city', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type'
                                   ])
        list_size.append(_df.shape[0])
        df = pd.concat([df, _df], axis=0, ignore_index=True)
    data = vectorizer.fit_transform(preprocess(df)).astype(np.float32)
    X_train = data[:list_size[0], :]
    X_test = data[list_size[0]:list_size[0] + list_size[1], :]
    with open('train_tfidf_all.pkl', 'wb') as f:
        pickle.dump(X_train, f, -1)
    with open('test_tfidf_all.pkl', 'wb') as f:
        pickle.dump(X_test, f, -1)

    with open('vectorizer_tfidf_all.pkl', 'wb') as f:
        pickle.dump(vectorizer, f, -1)


if __name__ == '__main__':
    train()
