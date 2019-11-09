import h5py
import numpy as np
import os

import tensorflow as tf

data_path = './ModelNet40/'

for d in [['train', len(os.listdir(data_path + 'train'))], ['test', len(os.listdir(data_path + 'test'))]]:
    data = None
    labels = None
    for j in range(d[1]):
        file_name = data_path + d[0] + '/ply_data_{0}{1}.h5'.format(d[0], j)
        f = h5py.File(file_name, mode='r')
        if data is None:
            data = f['data']
            labels = f['label']
        else:
            data = np.vstack((data, f['data']))
            labels = np.vstack((labels, f['label']))
    f.close()
    save_name = data_path + '/ply_data_{0}.h5'.format(d[0])
    print(data.shape)
    print(labels.shape)
    h5_fout = h5py.File(save_name)
    h5_fout.create_dataset(
        'data', data=data,
        dtype='float32')
    h5_fout.create_dataset(
        'label', data=labels,
        dtype='float32')
    h5_fout.close()
