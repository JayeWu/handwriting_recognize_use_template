import argparse
import tensorflow as tf


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


# 从 tfrecord 文件中解析结构化数据 （特征）
def parse_image_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),  # image data
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),  # number label
        })
    return features


#  输入数字，否则重新输
def input_num(input_str):
    while 1:
        a = input(input_str)
        if a == "":
            return a
        elif a.replace('.', '', 1).isdigit():
            return a
        else:
            print("wrong input!")


# 初始化化权重  for cnn
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))