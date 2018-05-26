import pandas as pd

"""
df = pd.read_csv('result_0524_tfidfcol3/feature_importances.csv')
df = df[df['imp'] > 0]
cols = [int(col.split('_')[-1]) for col in df.col.values if 'tfidf' in col]
print(len(cols))
pd.Series(cols, name='col').to_csv('tfidf_cols.csv', index=False, header=True)
"""

df = pd.read_csv('result_0524_check/feature_importances.csv')
df = df[df['imp'] > 0]
cols = [int(col.split('_')[-1]) for col in df.col.values if 'tfidf' in col]

_cols = pd.read_csv('tfidf_cols.csv')['col'].values[cols]

#pd.Series(_cols, name='col').to_csv('tfidf_cols2.csv', index=False, header=True)

df = pd.read_csv('result_0524_selectcol2/feature_importances.csv')
df = df[df['imp'] > 0]
cols = [int(col.split('_')[-1]) for col in df.col.values if 'tfidf' in col]

_cols = _cols[cols]
print(len(_cols))

#pd.Series(_cols, name='col').to_csv('tfidf_cols3.csv', index=False, header=True)

df = pd.read_csv('result_0524_selectcol3/feature_importances.csv')
df = df[df['imp'] > 0]
cols = [int(col.split('_')[-1]) for col in df.col.values if 'tfidf' in col]

_cols = _cols[cols]
print(len(_cols))

pd.Series(_cols, name='col').to_csv('tfidf_cols4.csv', index=False, header=True)
