import subprocess
import pandas as pd
import glob
from multiprocessing import Pool


def run(id):
    path = f'../input/train_jpg/{id}.jpg'
    cmd = ['identify', '-verbose', path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_data, stderr_data = p.communicate()
    a = p.returncode, stdout_data, stderr_data
    basename = cmd[2].split('/')[-1]
    with open(f'../data/img_info/{basename}.txt', 'wb') as f:
        f.write(a[1])


ids = pd.read_csv('../input/train.csv', usecols=['image'])['image'].dropna().tolist()

with Pool() as p:
    p.map(run, ids)
