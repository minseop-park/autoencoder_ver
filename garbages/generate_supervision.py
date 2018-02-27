from dataset import get_data
from utils import apply_window

from sklearn.cluster import KMeans
import numpy as np
import cv2
import time
import pickle
import pydicom

def get_mask(ori_img):
    w, h = ori_img.shape
    reshape = np.reshape(ori_img, [-1,1])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(reshape)
    markers = np.reshape(kmeans.labels_, [w, h])

    nw = int(w / 2)
    nh = int(h / 2)
    markers = (markers == markers[nw, nh]).astype(np.uint8)
    _, labels = cv2.connectedComponents(markers)
    mask = (labels == labels[nw, nh]).astype(np.uint8)
    # later version need to count most frequent value near center location
    return mask

def get_local_(img, min_max='min', local_size=3):
    # img: uint8
    width, height = img.shape
    out_img = np.zeros((width, height, (local_size*2+1)**2), dtype=np.uint8)
    ind = 0
    for w in range(-local_size, local_size+1):
        for h in range(-local_size, local_size+1):
            rolled = np.roll(img, w, axis=1)
            rolled = np.roll(rolled, h, axis=0)
            out_img[:,:,ind] = rolled
            ind += 1
    if min_max=='min':
        return np.min(out_img, axis=2)
    if min_max=='max':
        return np.max(out_img, axis=2)

def pixel_diff(ori, cont, local_size=2):
    width, height = ori.shape
    assert ori.shape == cont.shape

    out_img = np.zeros((width, height, (local_size*2+1)**2))
    ind = 0
    for w in range(-local_size, local_size+1):
        for h in range(-local_size, local_size+1):
            rolled = np.roll(ori, w, axis=1)
            rolled = np.roll(rolled, h, axis=0)
            out_img[:,:,ind] = rolled
            ind += 1
    out = abs(out_img - np.expand_dims(cont, axis=2))
    return np.min(out, axis=2)

root_dir = '/home/mike/DataSet/'
def get_data_all(set_type='train', wo_name=True):
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
    for x, y in zip(full_input_list, full_target_list):
        d1 = pydicom.dcmread(file_dir + x)
        d2 = pydicom.dcmread(file_dir + y)
        input_imgs.append(d1.pixel_array)
        target_imgs.append(d2.pixel_array)
    if wo_name:
        return np.array(input_imgs), np.array(target_imgs)
    else:
        return np.array(input_imgs), np.array(target_imgs), full_input_list

batch_size = 10
x, y, names =  get_data_all(wo_name=False)
supervisions = []
for i in range(x.shape[0]):
    start = time.time()
    a = apply_window(x[i])
    b = apply_window(y[i])

    mask = get_mask(a)
    mask2 = get_local_(mask, 'max', 5)
    mask2 = get_local_(mask2, 'min', 7)

    diff = pixel_diff(a,b)
#    supervisions.append(mask2 * diff)



    end = time.time()
    print ('time per img: %.3f,  step: %5d'%(end - start, i))

    np.save('supervision/' + names[i], mask2 * diff)
