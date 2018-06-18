import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm
ret = []
TYPE = 'test'

ids = pd.read_csv(f'../input/{TYPE}.csv', usecols=['image'])['image'].tolist()


def proc(img_id):
    if img_id != img_id:
        return {}
    df_new = {}

    for i, line in enumerate(open(f'../data/img_info/{img_id}.jpg.txt', 'r')):
        line = line.strip().replace(' ', '').split(':', 1)
        line[0] = f'{i:0>3}_{line[0]}'
        col, val = line
        if col in {'004_Geometry', '063_Pagegeometry'}:
            tmp = val.split('+')[0].split('x')
            for i in range(2):
                df_new[col + f'_{i}'] = float(tmp[i])
        elif col in {'017_min', '018_max', '019_mean', '020_standarddeviation', '025_min', '026_max', '027_mean', '028_standarddeviation', '033_min', '034_max', '035_mean', '036_standarddeviation', '042_min', '043_max', '044_mean', '045_standarddeviation'}:
            val = val if val != '' else '-1(-1)'
            tmp = val[:-1].split('(')
            for i in range(2):
                df_new[col + f'_{i}'] = float(tmp[i])
        elif col in {'052_redprimary', '053_greenprimary', '054_blueprimary'}:
            tmp = val[1:-1].split(',')
            for i in range(2):
                df_new[col + f'_{i}'] = float(tmp[i])
        elif col in {'070_date', '071_date'}:
            df_new[col] = pd.to_datetime(val[7:]).timestamp()
        elif col in {'079_Filesize', '081_Pixelspersecond'}:
            val = val if val != '' else '0KB'
            df_new[col] = float(val[:-2])
        elif col in {'080_Numberpixels'}:
            val = val if val != '' else '0K'
            df_new[col] = float(val[:-1])
        elif col in {'081_Usertime', '082_Usertime'}:
            val = val if val != '' else '-1u'
            df_new[col] = float(val[:-1])
        elif col in {'082_Elapsedtime', '083_Elapsedtime'}:
            val = val if val != '' else '0:0'
            df_new[col] = float(val.split(':')[1])
        elif col in {'015_Pixels', '021_kurtosis', '022_skewness', '023_entropy', '029_kurtosis', '030_skewness', '031_entropy', '037_kurtosis', '038_skewness', '039_entropy', '046_kurtosis', '047_skewness', '048_entropy'}:
            df_new[col] = float(val)
    if '070_date' in df_new and '071_date' in df_new:
        df_new['exif_date_diff'] = df_new['070_date'] - df_new['071_date']
    return df_new


with Pool() as p:
    rets = list(p.map(proc, tqdm(ids), chunksize=100))
#rets = list(map(proc, tqdm(ids)))
print('proc end')
df = pd.DataFrame(rets)
#print('conv end', df.T)
df.to_feather(f'{TYPE}_img_exif.ftr')
