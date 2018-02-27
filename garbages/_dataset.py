import numpy as np
import h5py
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import pydicom
import pickle

root_dir = '/home/mike/DataSet/'

def read_and_decode(serialized_example):
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image_x': tf.FixedLenFeature([], tf.string),
                'image_y': tf.FixedLenFeature([], tf.string)})
    img_x = tf.decode_raw(features['image_x'], tf.int16)
    img_y = tf.decode_raw(features['image_y'], tf.int16)
    return img_x, img_y

def normalize(img_x, img_y):
    img_x_out = tf.add(tf.cast(img_x, tf.float32), 2000.0)
    img_x_out = tf.divide(img_x_out, (6095.0))

    img_y_out = tf.add(tf.cast(img_y, tf.float32), 2000.0)
    img_y_out = tf.divide(img_y_out, (6095.0))
    return img_x_out, img_y_out

def reshape(img_x, img_y):
    img_x_out = tf.reshape(img_x, [256,256,1])
    img_y_out = tf.reshape(img_y, [256,256,1])
    return img_x_out, img_y_out

def next_batch(batch_size, train=True):
    f = 'train.tfrecords' if train else 'valid.tfrecords'
    with tf.name_scope('batch'):
        filename = os.path.join(root_dir + 'severance_data/' + f)
        dataset = tf.contrib.data.TFRecordDataset(filename)
        dataset = dataset.repeat(None)

        dataset = dataset.map(read_and_decode)
        dataset = dataset.map(normalize)
        dataset = dataset.map(reshape)
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def get_data(batch_size, set_type='train', wo_name=True):
    file_dir = root_dir + 'severance_data/ct_recon/'
    input_list_pkl = root_dir + 'severance_data/pixel_diff/' + set_type + '_x.pkl'
    target_list_pkl = root_dir + 'severance_data/pixel_diff/' + set_type + '_y.pkl'
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
        d1 = pydicom.dcmread(file_dir + x)
        d2 = pydicom.dcmread(file_dir + y)
        input_imgs.append(d1.pixel_array)
        target_imgs.append(d2.pixel_array)
    if wo_name:
        return np.array(input_imgs), np.array(target_imgs)
    else:
        return np.array(input_imgs), np.array(target_imgs), full_input_list[:batch_size]

def _get_test_data(batch_size):
    file_dir = root_dir + 'severance_data/ct_recon/'
    input_list_pkl = root_dir + 'severance_data/pixel_diff/x_list.pkl'
    target_list_pkl = root_dir + 'severance_data/pixel_diff/y_list.pkl'
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
        d1 = pydicom.dcmread(file_dir + x)
        d2 = pydicom.dcmread(file_dir + y)
        input_imgs.append(d1.pixel_array)
        target_imgs.append(d2.pixel_array)

    return np.array(input_imgs), np.array(target_imgs)

def _get_test_data(batch_size):
    file_dir = root_dir + 'severance_data/test_labels_same_loc/'
    label_list = os.listdir(file_dir)
    label_list = np.array(label_list)
    np.random.shuffle(label_list)

    t_pkl = root_dir + 'severance_data/full_target_list.pkl'
    with open(t_pkl, 'rb') as f:
        f_target_list = pickle.load(f)
    i_pkl = root_dir + 'severance_data/full_input_list.pkl'
    with open(i_pkl, 'rb') as f:
        f_input_list = pickle.load(f)

    input_list = []
    target_list = []
    for label in label_list[:batch_size]:
        file_name = label[6:].replace('.xml', '')
        img_name = root_dir + 'severance_data/ct_recon/' + file_name
        ds = pydicom.dcmread(img_name)
        img = ds.pixel_array
        input_list.append(img)
        for ind, key in enumerate(f_input_list):
            if file_name == key:
                break
        img_name = root_dir + 'severance_data/ct_recon/' + f_target_list[ind]
        ds = pydicom.dcmread(img_name)
        img = ds.pixel_array
        target_list.append(img)
    return np.array(input_list), np.array(target_list)

def get_data_with_supervision(batch_size):
    file_dir = root_dir + 'severance_data/labels_same_loc/'
    label_list = os.listdir(file_dir)
    label_list = np.array(label_list)
    np.random.shuffle(label_list)

    data_list = []
    sup_list = []

    for label in label_list[:batch_size]:
        file_name = label[6:].replace('.xml', '')
        img_name = root_dir +'severance_data/ct_recon/'+ file_name
        ds = pydicom.dcmread(img_name)
        img = ds.pixel_array
        data_list.append(img)
        supervision = np.zeros(img.shape)

        tree = ET.parse(file_dir + label)
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                mean_y = (int)(child[4][0].text)
                mean_x = (int)(child[4][1].text)
#                cov = np.identity(2) * 10
                cov = np.random.normal(size=[2,2]) * 1.0 + np.identity(2) * 10
                xp, yp = np.random.multivariate_normal([mean_x, mean_y], cov, 5000).T
                xp = xp.astype(int)
                yp = yp.astype(int)
                supervision[xp, yp] += 50.0
        sup_list.append(supervision)
    return np.array(data_list), np.array(sup_list)

class DataSet():
    def __init__(self):
        self.loc = None

    def _get_root_loc(self):
        return '/home/aitrics/user/mike/DataSet/'

    def next_batch(self, batch_size):
        x, y = self.x, self.y
        seq_ind = np.random.randint(x.shape[0], size=batch_size)

        batch_x = np.expand_dims(x[seq_ind], 3)
        batch_y = np.expand_dims(x[seq_ind], 3)
        return batch_x, batch_y

    def dataset_load(self):
        fo = h5py.File(self.loc, 'r')
        self.x = np.array(fo['input'])
        self.y = np.array(fo['target'])
        fo.close()

class SubDataSet(DataSet):
    def __init__(self, x, y):
        self.x = x
        self.y = y

#    def next_batch(self, batch_size):
#        x, y = self.x, self.y
#        seq_ind = np.random.randint(x.shape[0], size=batch_size)
#
#        batch_x = np.expand_dims(x[seq_ind], 3)
#        batch_y = np.expand_dims(x[seq_ind], 3)
#        return batch_x, batch_y

class SeveranceCT(DataSet):
    def __init__(self):
        self.loc = self._get_root_loc() + 'severance_data/' + \
                'ct_recon_256/pairs_resized.h5'
        self.dataset_load()

    def get_10_fold_val(self):
        np.random.seed(0)
        data_len = self.x.shape[0]
        shuffle_ind = np.arange(data_len)
        np.random.shuffle(shuffle_ind)

        split = int(data_len / 10)
        print (self.x[:split].shape)
        val_set = SubDataSet(self.x[:split], self.y[:split])
        train_set = SubDataSet(self.x[split:], self.y[split:])

        return train_set, val_set

if __name__=='__main__':
    get_test_data(10)
