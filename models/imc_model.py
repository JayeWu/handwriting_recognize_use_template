import tensorflow as tf
from base.base_model import BaseModel
from utils.utils import init_weights, input_num


class IMCModel(BaseModel):
    def __init__(self, config):
        super(IMCModel, self).__init__(config)
        self.params = self.Params()
        self.params.input_params()
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder("float", [None, self.config.image_size[0], self.config.image_size[1], 1],
                           name="X")  # placeholder for model input(image data)
        self.y = tf.placeholder("float", [None, self.config.label_size],
                           name="Y")  # placeholder for model output (prediction)
        self.py_x = self.model()  # 构建网络模型, 输出为py_x
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.y))  # 代价函数
        correct_prediction = tf.equal(tf.argmax(self.py_x, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.train_op = tf.train.RMSPropOptimizer(self.config.learning_rate, 0.9).minimize(self.cross_entropy)  # 训练操作 使用RMS优化器

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def model(self):
        layer_input = 1
        lb = self.x
        image_min_wid_hei = min(self.x.get_shape().as_list()[1:2])  # image 的 宽和高的较小值
        self.p_keep_conv = tf.placeholder("float", name="p_keep_conv")
        self.p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")
        for i in range(self.params.n_layer):
            layer_output = self.params.neurons[i]
            w = init_weights([self.params.patch_size[0], self.params.patch_size[1], layer_input, layer_output])
            la = tf.nn.relu(tf.nn.conv2d(lb, w, strides=[1, 1, 1, 1], padding='SAME'))  # 卷积层

            if round(image_min_wid_hei/(2**(i+1))) >= 4:
                l1 = tf.nn.max_pool(la, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            else:
                l1 = la
            if i == self.params.n_layer - 1:
                wide = round(image_min_wid_hei/(2**self.params.n_layer))  #
                if wide < 4:
                    wide = 4  # 池化(降采样)后，图片的最小宽度为4
                print("wide ", wide)
                w2 = init_weights([layer_output*wide*wide, self.params.output_dimension])
                l2 = tf.reshape(l1, [-1, w2.get_shape().as_list()[0]])
                print(l2)
            else:
                l2 = tf.nn.dropout(l1, self.p_keep_conv)

            layer_input = layer_output
            lb = l2
        w_o = init_weights([self.params.output_dimension, self.config.label_size])
        lo = tf.nn.relu(tf.matmul(lb, w2))
        lo = tf.nn.dropout(lo, self.p_keep_hidden)
        pyx = tf.matmul(lo, w_o)
        return pyx

    class Params:
        def __init__(self):
            self.batch_size = 128  # batch size of train data
            self.test_size = 300  # test data size 一次测试所用的数据量
            self.learning_rate = 0.001
            self.max_epochs = 10000
            self.expect_accuracy = 0.99
            self.n_layer = 3
            self.neurons = []
            self.output_dimension = 625
            self.patch_size = [3, 3]  # 卷积层的卷积核大小
            self.model_type = 'cnn'
            self.rnn_hidden_units = 128  # rnn hidden layer units

        def input_params(self):
            while 1:
                cr = input("please choose CNN or RNN (cnn/rnn, default cnn): ")
                if cr != '':
                    if cr != 'rnn' and cr != 'cnn':
                        print("wrong input! ")
                    else:
                        self.model_type = cr
                        break
                else:
                    break
            if self.model_type == 'cnn':
                f = input_num("please enter the number of convolution layer (3): ")
                if f != "":
                    self.n_layer = int(f)
                for i in range(self.n_layer):
                    g = input_num("%d convolution layer's neurons(%d): " % (i+1, 32*(2**i)))
                    if g != "":
                        self.neurons.append(int(g))
                    else:
                        self.neurons.append(32*(2**i))
                h = input_num("please enter the dimension of full connect layer(625): ")
                if h != "":
                    self.output_dimension = int(h)
            elif self.model_type == 'rnn':
                k = input_num("please enter the number of rnn hidden layer units(128): ")
                if k != "":
                    self.rnn_hidden_units = int(k)
            a = input_num("please enter the  batch size of train data (128): ")
            if a != "":
                self.batch_size = int(a)
            b = input_num("please enter the size of test data (300): ")
            if b != "":
                self.test_size = int(b)
            c = input_num("please enter the learning rate of optimizer (0.001): ")
            if c != "":
                self.learning_rate = float(c)
            d = input_num("please enter the maximal epochs of model training (10000): ")
            if d != "":
                self.max_epochs = int(d)
            e = input_num("please enter the expect accuracy of model (0.99): ")
            if e != "":
                self.expect_accuracy = float(e)

