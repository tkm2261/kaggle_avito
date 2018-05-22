import pandas as pd

"""
df = pd.read_csv('result_tmp/feature_importances.csv')
df = df[df['imp'] > 0]
cols = [int(col.split('_')[1]) for col in df.col.values if 'tfidf' in col]
print(len(cols))

pd.Series(cols, name='col').to_csv('tfidf_cols.csv', index=False, header=True)
"""

df = pd.read_csv('result_0521_tfidfcol/feature_importances.csv')
df = df[df['imp'] > 0]
cols = [int(col.split('_')[1]) for col in df.col.values if 'tfidf' in col]

cols = pd.read_csv('tfidf_cols.csv')['col'].values[cols]
pd.Series(cols, name='col').to_csv('tfidf_cols2.csv', index=False, header=True)
