import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import _init_paths
from network import ResAutoEncoderTrainNet
from network import ResAutoEncoderTestNet
from utils import Avg, apply_window
from dataset import get_train_pair, get_test_data

model_name = 'autoencoder'
save_dir = '/home/mike/models/' + model_name + '/'
max_iter = 10000

img_x = tf.placeholder(tf.float32, [None,None,None])
img_y = tf.placeholder(tf.float32, [None,None,None])
b_ph = tf.placeholder(tf.bool, [None,None,None])
x_reshape = tf.expand_dims(img_x, axis=3)

model = ResAutoEncoderTrainNet('x_autoencoder')
#model = ResAutoEncoderTestNet('x_autoencoder')
codes = model.encoder(x_reshape)
recon = model.decoder2(codes)
#
mask = tf.greater(img_y, 10)
mask = tf.logical_or(mask, b_ph)
mask = tf.cast(mask, tf.float32)

#loss = tf.reduce_mean(    (img_y - recon)**2  * mask + abs(recon))
bloss = tf.reduce_sum(img_y*tf.log(recon+1e-10) \
        + (1-img_y)*tf.log(1-recon+1e-10) - abs(recon), axis=0)
loss = -tf.reduce_mean(bloss * mask) * 10000

#        + tf.reduce_mean( tf.abs(img_y - recon) )
#loss = tf.reduce_mean(tf.square(tf.subtract(img_y, recon)))# + tf.abs(recon))
opt = tf.train.AdamOptimizer(1e-4)
train_op = opt.minimize(loss)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
mav = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'moving_mean' in v.name]
mav = mav[0]
#with tf.control_dependencies(update_op):
#    train_op = opt.minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

#model.pretrain_load(sess)
saver = tf.train.Saver()
saver.restore(sess, save_dir + 'a.ckpt')

avg = Avg(['loss'])
for i in range(1, 1+max_iter):
    x, y = get_train_pair(8)
    rnd_bern = np.random.randint(100, size=[8,512,512])
    rnd_bern = rnd_bern < 2
    fd = {img_x: x, img_y: y, b_ph: rnd_bern}
    _, _, l = sess.run([train_op, update_op, loss], fd)
    avg.add(l, 0)

    if i % 30 == 0:
        avg.show(i)
    if i % 100 == 0:
        fd = {img_x: x, img_y: y}
        rc, rx, ry = sess.run([recon, img_x, img_y], fd)
        for k in range(rc.shape[0]):
            np.save('sample_imgs/a_'+str(k)+'.npy', rc[k])
            np.save('sample_imgs/x_'+str(k)+'.npy', rx[k])
            np.save('sample_imgs/y_'+str(k)+'.npy', ry[k])
        avg.description()
        print (np.mean(rc), np.mean(ry), np.mean(rx))
        saver.save(sess, save_dir + 'a.ckpt')
