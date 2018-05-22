
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import pickle
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from logging import getLogger

logger = getLogger(None)


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, logs[k]) for k in sorted(logs)))

        logger.info(msg)


def get_model(ndim):
    model_in = ks.Input(shape=(ndim,), dtype='float32', sparse=True)
    out = ks.layers.Dense(192, activation='relu', name='mid1')(model_in)
    out = ks.layers.Dense(64, activation='relu', name='mid2')(out)
    out = ks.layers.Dense(64, activation='relu', name='mid22')(out)
    out = ks.layers.Dense(32, activation='relu', name='mid3')(out)
    out = ks.layers.Dense(1, activation='sigmoid')(out)
    model = ks.Model(model_in, out)
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.RMSprop(lr=1e-5))

    return model


def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['title'] = df['title'].fillna('') + ' ' + df['parent_category_name'].fillna('')
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


def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')


def train():

    vectorizer = make_union(
        on_field('title', Tfidf(ngram_range=(3, 3), analyzer='char', max_features=1000000, min_df=5)),
        on_field('description', Tfidf(ngram_range=(3, 3), analyzer='char', max_features=1000000, min_df=5)),
        on_field(['item_seq_number'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        FunctionTransformer(itemgetter(['price']), validate=False),
        # n_jobs=4
    )

    df = pd.read_csv('../input/train.csv',
                     usecols=['title', 'description', 'price', 'item_seq_number',
                              'parent_category_name',
                              'region', 'city', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type',
                              'deal_probability'])
    print('load end')
    target = df['deal_probability']
    X_train = vectorizer.fit_transform(preprocess(df)).astype(np.float32)
    with open('train_nn_chargram.pkl', 'wb') as f:
        pickle.dump(X_train, f, -1)
    with open('vectorizer_chargram.pkl', 'wb') as f:
        pickle.dump(vectorizer, f, -1)

    y_train = target.values
    metric = 'val_loss'
    mode = 'min'

    callbacks = [EarlyStopping(monitor=metric,
                               patience=10,
                               verbose=1,
                               min_delta=1e-6,
                               mode=mode),
                 ReduceLROnPlateau(monitor=metric,
                                   factor=0.1,
                                   patience=2,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode=mode),

                 ModelCheckpoint(monitor=metric,
                                 filepath='weights/best_weights_chargram.hdf5',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode=mode),
                 TensorBoard(log_dir='logs'),
                 LoggingCallback()
                 ]

    model = get_model(X_train.shape[1])
    cv = KFold(n_splits=5, shuffle=True, random_state=871)
    for train, test in cv.split(X_train, y_train):
        trn_x = X_train[train, :]
        val_x = X_train[test, :]
        trn_y = y_train[train]
        val_y = y_train[test]
        break
    model.fit(x=trn_x, y=trn_y,
              validation_data=(val_x, val_y),
              batch_size=2**11,
              epochs=1000,
              callbacks=callbacks)
    # X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)


def predict():
    with open('vectorizer_chargram.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    df = pd.read_csv('../input/train.csv',
                     usecols=['title', 'description', 'price', 'item_seq_number',
                              'parent_category_name',
                              'region', 'city', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type',
                              'deal_probability'])
    X_test = vectorizer.transform(preprocess(df)).astype(np.float32)
    model = get_model(X_test.shape[1])
    model.load_weights(filepath='weights/best_weights_chargram.hdf5')
    layer_name = 'mid3'
    intermediate_layer_model = ks.Model(inputs=model.input,
                                        outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(X_test, batch_size=2**11)
    with open('nn_train_chargram.pkl', 'wb') as f:
        pickle.dump(intermediate_output, f, -1)

    df = pd.read_csv('../input/test.csv',
                     usecols=['title', 'description', 'price', 'item_seq_number',
                              'parent_category_name',
                              'region', 'city', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type',
                              ])
    X_test = vectorizer.transform(preprocess(df)).astype(np.float32)
    intermediate_output = intermediate_layer_model.predict(X_test, batch_size=2**11)
    with open('nn_test_chargram.pkl', 'wb') as f:
        pickle.dump(intermediate_output, f, -1)


if __name__ == '__main__':
    train()
    predict()
