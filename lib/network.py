import tensorflow as tf
import numpy as np

from utils import fc, deconv, unpool
import resnet_v2
slim = tf.contrib.slim

class ResAutoEncoderTrainNet():
    def __init__(self, name, loc=None):
        self.name = name
        if not loc:
            self.model_loc = 'models/'+name+'.ckpt'
        else:
            self.model_loc = loc

    def encoder(self, x_ph, reuse=False, trainable=True):
        with tf.variable_scope(self.name):
            x_reshape = tf.tile(x_ph, [1,1,1,3])
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits, _ = resnet_v2.resnet_v2_101(x_reshape, 1001, is_training=True)
        return logits

    def decoder(self, hid_feat, reuse=False, trainable=True):
        with tf.variable_scope(self.name):
            L = fc('dfc1', 8*8*256, hid_feat)
            L = tf.reshape(L, [-1,8,8,256])
            L = unpool(L)
            L = deconv('dc1', [3,3,256,256], L)
            L = unpool(L)
            L = deconv('dc2', [3,3,256,128], L)
            L = unpool(L)
            L = deconv('dc3', [3,3,128,64], L)
            L = unpool(L)
            L = deconv('dc4', [3,3,64,64], L)
            L = unpool(L)
            L = deconv('dc5', [3,3,64,64], L, activation=None)
            L = unpool(L)
            L = deconv('dc6', [3,3,64,1], L, activation=None)
            L = tf.squeeze(L)
        return L

    def saver_init(self):
        model_var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.name+'*')
        self.saver = tf.train.Saver(model_var_lists)

    def saver_save(self, sess):
        print ('save model at : ', self.model_loc)
        self.saver.save(sess, self.model_loc)

    def saver_load(self, sess):
        print ('load model from : ', self.model_loc)
        self.saver.restore(sess, self.model_loc)

    def pretrain_load(self, sess):
        model_loc = 'models/'+self.name+'_pretrained.ckpt'
        print ('load pretrained model from : ', model_loc)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.name+'/resnet*')
        saver = tf.train.Saver(var_list)
        saver.restore(sess, model_loc)

class ResOld(ResAutoEncoderTrainNet):
    def __init__(self, name):
        self.name = name

    def encoder(self, x_ph, reuse=False, trainable=True):
        x_reshape = tf.tile(x_ph, [1,1,1,3])
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_101(x_reshape, 1001, is_training=True)
        return logits

    def pretrain_load(self, sess):
        pretrain_loc = '/home/aitrics/user/mike/DataSet/pretrained_model/'
        version = 'resnet_v2_101.ckpt'
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet*')
        saver = tf.train.Saver(var_list)
        saver.restore(sess, pretrain_loc+version)
        print ('pretrained model restored from : ', version)

class ResAutoEncoderTestNet(ResAutoEncoderTrainNet):
    def __init__(self, name, loc=None):
        self.name = name
        if not loc:
            self.model_loc = 'models/'+name+'.ckpt'
        else:
            self.model_loc = loc

    def encoder(self, x_ph, reuse=False, trainable=False):
        with tf.variable_scope(self.name):
            x_reshape = tf.tile(x_ph, [1,1,1,3])
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits, _ = resnet_v2.resnet_v2_101(x_reshape, 1001, is_training=False)
        return logits
