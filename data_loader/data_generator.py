import numpy as np
import tensorflow as tf
import os
from utils.utils import parse_image_example


class DataGenerator:
    def __init__(self, config):
        self.dir_train = config.train_data_dir
        self.dir_test = config.test_data_dir
        self.image_batch = None
        self.label_batch = None
        self.image_batch_test = None
        self.label_batch_test = None
        self.image_size = config.image_size
        self.label_size = config.label_size
        self.image_pixels = config.image_size[0] * config.image_size[1]
        self.__prepare_data(config)

    # 准备训练和测试数据
    def __prepare_data(self, config):
        reader = tf.TFRecordReader()  # reader for TFRecord file
        if os.path.exists(self.dir_train) and os.path.exists(self.dir_test):
            train_queue = tf.train.string_input_producer([self.dir_train])
            test_queue = tf.train.string_input_producer([self.dir_test])  # files read queue
        else:
            raise Exception("%s or %s file doesn't exist" % (self.dir_train, self.dir_test))
        _, serialized_example = reader.read(train_queue)  # examples in TFRecord file
        _, serialized_test = reader.read(test_queue)
        features = parse_image_example(serialized_example)  # parsing features
        features_test = parse_image_example(serialized_test)
        pixels = tf.cast(features['pixels'], tf.int32)
        image = tf.decode_raw(features['image_raw'], tf.uint8)  # decode image data from string to image, a Tensor
        image.set_shape([self.image_pixels])  # pixels is 784
        label = tf.cast(features['label'], tf.int32)
        image_test = tf.decode_raw(features_test['image_raw'], tf.uint8)
        image_test.set_shape([self.image_pixels])
        label_test = tf.cast(features_test['label'], tf.int32)
        # self.image_batch, lb = tf.train.shuffle_batch(
        #     [image, label], batch_size=batch_size, capacity=capacity,
        #     min_after_dequeue=500)  # queue of image_batch, shuffle_batch mean random
        self.image_batch, lb = tf.train.batch(
            [image, label], batch_size=config.batch_size,
            capacity=10000)  # queue of image_batch, shuffle_batch mean random
        self.label_batch = tf.one_hot(lb, self.label_size)  # one_hot, 2 for [0,0,1,0,0,0,...]
        self.image_batch_test, lb_test = tf.train.shuffle_batch(
            [image_test, label_test], batch_size=config.test_size, capacity=10000, min_after_dequeue=0)
        self.label_batch_test = tf.one_hot(lb_test, self.label_size)

    def next_batch(self, sess):
        yield sess.run([self.image_batch, self.label_batch])

    def next_batch_test(self, sess):
        yield sess.run([self.image_batch_test, self.label_batch_test])
