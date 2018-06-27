import pandas as pd
import numpy as np


df_sub = pd.read_csv('submit_ens_0624_external.csv').sort_values('deal_probability').reset_index(drop=True)

df = pd.read_csv('../input/train.csv', usecols=['deal_probability'])

df = df.groupby('deal_probability')['deal_probability'].count()
df.name = 'hoge'

df = df.reset_index()
df = df[df['hoge'] > 1].copy()

val_score = df.sort_values('deal_probability')['deal_probability'].values


ret = []
idx = 0
idx_p = 0
preds = df_sub['deal_probability'].values

while idx_p < preds.shape[0]:
    try:
        curr = val_score[idx]
    except IndexError:
        curr = 1
    try:
        nxt = val_score[idx + 1]
    except IndexError:
        nxt = 1

    p = preds[idx_p]
    diff_c = p - curr
    diff_n = nxt - p
    if p < curr:
        ret.append(curr)
        idx_p += 1
        #print(p, curr, nxt, ret[-1])
    elif p > nxt:
        idx += 1
    elif diff_c < diff_n:
        ret.append(curr)
        idx_p += 1
        #print(p, curr, nxt, ret[-1])
    elif diff_n <= diff_c:
        ret.append(nxt)
        idx_p += 1
        #print(p, curr, nxt, ret[-1])

df_sub['deal_probability'] = ret
df_sub.to_csv('submit_adjust.csv', index=False)
