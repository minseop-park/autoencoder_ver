import numpy as np
import h5py
import tensorflow as tf
import os
import pydicom
import pickle
import cv2
from utils import apply_window


root_dir = '/home/mike/DataSet/'

def get_test_data(batch_size):
    file_dir = root_dir + 'severance_data/imgs/'
    input_list_pkl = root_dir + 'severance_data/pixel_diff/val_x.pkl'
    target_list_pkl = root_dir + 'severance_data/pixel_diff/val_y.pkl'
    with open(input_list_pkl, 'rb') as f:
        full_input_list = pickle.load(f)
    with open(target_list_pkl, 'rb') as f:
        full_target_list = pickle.load(f)

    assert (len(full_input_list) == len(full_target_list))
    indexes = np.arange(len(full_input_list))
    np.random.shuffle(indexes)
    full_input_list = np.array(full_input_list)[indexes]
    full_target_list = np.array(full_target_list)[indexes]
    input_imgs = []
    target_imgs = []
    for x, y in zip(full_input_list[:batch_size], full_target_list[:batch_size]):
        input_imgs.append(np.load(file_dir + x + '.npy'))
        target_imgs.append(np.load(file_dir + y + '.npy'))
    return np.array(input_imgs), np.array(target_imgs)

def get_train_pair(batch_size):
    file_dir = root_dir + 'severance_data/supervision/'
    d_list = os.listdir(file_dir)
    xs = []
    ys = []
    np.random.shuffle(d_list)
    for dname in d_list[:batch_size]:
        yimg = np.load(file_dir + dname)
        yimg = cv2.blur(yimg, (5,5))
        yimg[yimg < 10] = 0
        ys.append(yimg * 100)
        img_name = root_dir + 'severance_data/imgs/' + dname
        img = np.load(img_name)
        #img = apply_window(img)
        xs.append(img)
    return np.array(xs), np.array(ys)

if __name__=='__main__':
    get_test_data(10)
