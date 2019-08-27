# import data
import numpy as np
# import math
# from pathlib import Path
import tensorflow as tf
import model
# import random
# CLASSNAMES      = ['crack', 'inactive']
# BATCH_SIZE      = 10
# DATASET_TRAIN   = Path('.') / 'data' / 'train.csv'
# ds = data.Dataset(DATASET_TRAIN, CLASSNAMES, BATCH_SIZE, False, False)
# print(ds.image_shape)
# print(ds.get_labels()[:10])
# input()
# random.seed()
# print(random.randint(25,50))

with tf.Graph().as_default():
    with tf.variable_scope("Melissa"):
        # input = tf.placeholder(tf.float32, shape=(1, 32, 32, 1))
        # label = tf.placeholder(tf.float32, shape=(1, 2))
        # conv2d = tf.layers.conv2d(input, filters=10, kernel_size=3, strides=2, padding="same", trainable=True, name="conv2d")
        # batchnorm = tf.layers.batch_normalization(conv2d, training=True, name="batchnorm")
        # relu = tf.nn.relu(batchnorm, name="relu")
        # # conv2d_2 = tf.layers.conv2d(relu, filters=10, kernel_size=16, strides=1, padding="valid", trainable=True, name="conv2d_16_16")
        # flatten = tf.layers.Flatten()(relu)
        # fc = tf.layers.dense(flatten, units=2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc")
        #
        # loss = tf.losses.softmax_cross_entropy(label, fc)
        # optimizer = tf.train.AdamOptimizer(0.01)
        # train_op = optimizer.minimize(loss)
        input = tf.placeholder(tf.float32, shape=(10,224,224,1))
        alexN = model.inception_resnet(input, 2)
        summary = tf.summary.merge_all()
        sess = tf.Session()
        writer = tf.summary.FileWriter("./summary", sess.graph)

