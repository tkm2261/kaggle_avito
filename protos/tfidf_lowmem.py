import pickle
from scipy import sparse
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import gc

SPLIT = re.compile('\w+')


def preprocess(df: pd.DataFrame):
    """    
    df['title'] = df['title'].fillna('') + ' ' + df['parent_category_name'].fillna('')
    """
    df['description'] = df['description'].fillna('')
    df['description'] = (df['description'].fillna('') + ' ' +
                         df['region'].fillna('') + ' ' +
                         df['city'].fillna('') + ' ' +
                         df['category_name'].fillna('') + ' ' +
                         df['param_1'].fillna('') + ' ' +
                         df['param_2'].fillna('') + ' ' +
                         df['param_3'].fillna('') + ' ' +
                         df['user_type'].fillna('')
                         )

    return df['description'].values  # df['title'].values  # , 'description']]


def make_spmat():
    postfix = 'description'

    map_words = {}
    idx = 0
    row_idx = []
    col_idx = []
    data = []
    i = 0
    #
    for path in tqdm(['../input/train.csv', '../input/test.csv', '../input/train_active.csv', '../input/test_active.csv']):
        df = pd.read_csv(path,
                         usecols=[  # 'title',
                             'description',
                             'price', 'item_seq_number',
                             # 'parent_category_name',
                             'region', 'city', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type'
                         ])
        rows = preprocess(df)
        for row in tqdm(rows):
            for word, cnt in Counter(SPLIT.split(row)).most_common():
                if word in map_words:
                    j = map_words[word]
                else:
                    j = idx
                    map_words[word] = idx
                    idx += 1
                row_idx.append(i)
                col_idx.append(j)
                data.append(cnt)
            i += 1
        del df
        gc.collect()
    bow_matrix = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(i, idx))
    sparse.save_npz(f'bow/bow_matrix_{postfix}.npz', bow_matrix)
    with open(f'bow/bow_dict_{postfix}.pkl', 'wb') as f:
        pickle.dump(map_words, f, -1)


def make_tfidf_split():
    train_num = pd.read_csv('../input/train.csv').shape[0]
    test_num = pd.read_csv('../input/test.csv').shape[0]

    postfix = 'title'
    bow_matrix = sparse.load_npz(f'bow/bow_matrix_{postfix}.npz')
    print(bow_matrix.shape)
    idf_matrix = sparse.csr_matrix(bow_matrix)
    idf_matrix.data = np.ones(idf_matrix.nnz, dtype=idf_matrix.dtype)

    idf_matrix = idf_matrix.sum(axis=0)
    idf_matrix = np.log(bow_matrix.shape[0] / idf_matrix)

    tfidf_matrix = bow_matrix * sparse.diags(np.array(idf_matrix)[0], 0)
    sparse.save_npz(f'bow/train_tfidf_matrix_{postfix}.npz', tfidf_matrix[:train_num])
    sparse.save_npz(f'bow/test_tfidf_matrix_{postfix}.npz', tfidf_matrix[train_num: train_num + test_num])


if __name__ == '__main__':
    # make_spmat()
    make_tfidf_split()
