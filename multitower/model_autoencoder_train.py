import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import _init_paths
from network import VGGNet
from utils import Avg, apply_window, avg_grad, get_mask
from dataset import get_train_pair

model_name = 'multitower_autoencoder'
save_dir = '/home/mike/models/' + model_name + '/'
max_iter = 10000
num_gpus = 4



model = VGGNet('model_name_'+'vgg')

with tf.Graph().as_default(), tf.device('/cpu:0'):
    img_x = tf.placeholder(tf.float32, [None,512,512])
    img_y = tf.placeholder(tf.float32, [None,512,512])
    isTr_ph = tf.placeholder(tf.bool)
    z_ph = tf.placeholder(tf.bool, [None,None,None])
    x_reshape = tf.expand_dims(img_x, axis=3)

    x_ph_split = tf.split(x_reshape, num_gpus)
    y_ph_split = tf.split(img_y, num_gpus)

    opt = tf.train.AdamOptimizer(1e-4)
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpus):
            with tf.device('/gpu:%d'%i):
                with tf.name_scope('tower_%d'%i) as scope:
                    codes = model.encoder(x_ph_split[i], isTr_ph)
                    recon = model.decoder(codes, isTr_ph)
                    mask = get_mask(y_ph_split[i], z_ph)
                    bloss = tf.reduce_sum(y_ph_split[i]*tf.log(recon+1e-10) \
                            + (1-y_ph_split[i])*tf.log(1-recon+1e-10)\
                            - abs(recon), axis=0)
                    loss = -tf.reduce_mean(bloss * mask) * 10000
                    tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
    grads = avg_grad(tower_grads)
    apply_grad_op = opt.apply_gradients(grads)
    mv_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(apply_grad_op, mv_update_op)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    avg = Avg(['loss'])
    for i in range(1, 1+max_iter):
        x, y = get_train_pair(8*num_gpus)
        zero_sampling = np.random.randint(100, size=[8,512,512])
        zero_sampling = zero_sampling < 2
        fd = {img_x: x, img_y: y, z_ph: zero_sampling, isTr_ph:True}
        _, l = sess.run([train_op, loss], fd)
        avg.add(l, 0)
        if i % 50 == 0:
            avg.show(i)
        if i % 100 == 0:
            rc, rx, ry = sess.run([recon, img_x, img_y], fd)
            for k in range(rc.shape[0]):
                np.save('sample_imgs/a_'+str(k)+'.npy', rc[k])
                np.save('sample_imgs/x_'+str(k)+'.npy', rx[k])
                np.save('sample_imgs/y_'+str(k)+'.npy', ry[k])
            avg.description()
            print (np.mean(rc), np.mean(ry), np.mean(rx))
            saver.save(sess, save_dir + 'a.ckpt')
