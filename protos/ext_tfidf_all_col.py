import pandas as pd


df = pd.read_csv('result_0617_tfsp_1/feature_importances.csv')
df = df[df['imp'] > 0]
_cols = [int(col.split('_')[-1]) for col in df.col.values if 'tfidf' in col]
print(_cols)
pd.Series(_cols, name='col').to_csv('tfidf_sp_cols.csv', index=False, header=True)

"""
df = pd.read_csv('result_0617_tf2/feature_importances.csv')
df = df[df['imp'] > 0]
cols = [int(col.split('_')[-1]) for col in df.col.values if 'tfidf' in col]

_cols = pd.read_csv('tfidf_all_cols.csv')['col'].values[cols]

#pd.Series(_cols, name='col').to_csv('tfidf_all_cols2.csv', index=False, header=True)

df = pd.read_csv('result_0617_tf3/feature_importances.csv')
df = df[df['imp'] > 0]
cols = [int(col.split('_')[-1]) for col in df.col.values if 'tfidf' in col]

_cols = _cols[cols]

#pd.Series(_cols, name='col').to_csv('tfidf_all_cols3.csv', index=False, header=True)

df = pd.read_csv('result_0617_tf4/feature_importances.csv')
df = df[df['imp'] > 0]
cols = [int(col.split('_')[-1]) for col in df.col.values if 'tfidf' in col]

_cols = _cols[cols]

pd.Series(_cols, name='col').to_csv('tfidf_all_cols4.csv', index=False, header=True)
"""
