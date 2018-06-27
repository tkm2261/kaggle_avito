
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import pickle
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

from multiprocessing import Pool
from logging import getLogger

from scipy import sparse

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
        on_field('title', CountVectorizer(max_features=1000000, min_df=5,
                                          # token_pattern='[\w\?,\.;:\(\)\[\]]+',
                                          token_pattern='\w+',
                                          stop_words=stops, ngram_range=(1, 2),
                                          # lowercase=True,
                                          # smooth_idf=False
                                          )),
        on_field('description', CountVectorizer(max_features=1000000, min_df=5,
                                                lowercase=True,
                                                # token_pattern='[\w\?,\.;:\(\)\[\]]+',
                                                token_pattern='\w+',
                                                ngram_range=(1, 2),
                                                # stop_words=stops,
                                                # smooth_idf=False
                                                )
                 ),
        n_jobs=2)
    df = pd.DataFrame()
    list_size = []
    #
    #
    # , '../input/train_active.csv', '../input/test_active.csv']:
    for path in tqdm(['../input/train.csv', '../input/test.csv']):
        _df = pd.read_csv(path,
                          usecols=['title',
                                   'description',
                                   'price', 'item_seq_number',
                                   'parent_category_name',
                                   'region', 'city', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type'
                                   ])
        _df = preprocess(_df)
        list_size.append(_df.shape[0])
        df = pd.concat([df, _df], axis=0, ignore_index=True)
    data = vectorizer.fit_transform(df)

    df = []
    for path in tqdm(['../input/train_active.csv', '../input/test_active.csv']):
        df = list(tqdm(pd.read_csv(path,
                                   usecols=['title',
                                            'description',
                                            'price', 'item_seq_number',
                                            'parent_category_name',
                                            'region', 'city', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type'
                                            ], chunksize=1000000)))
        with Pool() as p:
            df = list(p.map(preprocess, tqdm(df)), chunksize=10)
        with Pool() as p:
            df = list(p.map(vectorizer.transform, df), chunksize=10)
    data2 = sparse.vstack(df)
    print("stack1")
    bow_matrix = sparse.vstack([data, data2])

    print(bow_matrix.shape)
    idf_matrix = sparse.csr_matrix(bow_matrix)
    idf_matrix.data = np.ones(idf_matrix.nnz, dtype=idf_matrix.dtype)

    idf_matrix = idf_matrix.sum(axis=0)
    idf_matrix = np.log(bow_matrix.shape[0] / idf_matrix)
    print("idf", idf_matrix.shape)
    tfidf_matrix = bow_matrix * sparse.diags(np.array(idf_matrix)[0], 0)
    train_num = list_size[0]
    test_num = list_size[1]
    sparse.save_npz('train_tfidf_matrix.npz', tfidf_matrix[:train_num])
    sparse.save_npz('test_tfidf_matrix.npz', tfidf_matrix[train_num: train_num + test_num])

    with open('vectorizer_tfidf_all.pkl', 'wb') as f:
        pickle.dump(vectorizer, f, -1)


if __name__ == '__main__':
    train()
