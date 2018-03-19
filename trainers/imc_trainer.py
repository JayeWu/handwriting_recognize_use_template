from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from PIL import Image
from tensorflow.python.platform import gfile
import os
import tensorflow as tf


class IMCTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(IMCTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses=[]
        accs=[]
        for it in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        print("\nloss: ", loss, 'accuracy:', acc)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        summaries_dict['acc'] = acc
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)
        return acc

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        bx, batch_ys = next(self.data.next_batch(self.sess))
        batch_xs = bx.reshape(-1, self.config.image_size[0], self.config.image_size[1], 1)  # reshape 数据
        feed_dict = {self.model.x: batch_xs, self.model.y: batch_ys, self.model.is_training: True,  self.model.p_keep_conv: 0.8, self.model.p_keep_hidden: 0.5}
        _, loss, acc = self.sess.run([self.model.train_op, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        tex, tey = next(self.data.next_batch_test(self.sess))  # 取出测试数据
        tex = tex.reshape(-1, self.config.image_size[0], self.config.image_size[1], 1)
        accuracy = np.mean(np.argmax(tey, axis=1) ==
                           self.sess.run(self.model.predict_op, feed_dict={self.model.x: tex, self.model.p_keep_conv: 1.0, self.model.p_keep_hidden: 1.0}))
        return loss, accuracy

    def recognize(self):
        # 手写图片预处理
        dir_af = "test_num"
        with self.sess as sess:
            # 开始识别图片
            files = os.listdir(dir_af)
            cnt = len(files)
            correct = 0
            for i in range(cnt):
                actual_label = int(files[i][0])
                files[i] = dir_af + "/" + files[i]
                img = Image.open(files[i])  # 读取要识别的图片
                print("input: ", files[i])
                imga = np.array(img).reshape(-1, self.config.image_size[0], self.config.image_size[1], 1)
                # feed 数据给 张量predict_op
                predict_op = tf.argmax(self.model.py_x, 1, name="predict_op")
                prediction = predict_op.eval(feed_dict={self.model.x: imga, self.model.p_keep_conv: 1.0, self.model.p_keep_hidden: 1.0})
                # 输出
                print("output: ", prediction)
                if prediction == actual_label:
                    print("Correct!")
                    correct = correct + 1
                else:
                    print("Wrong!")
                print("\n")
            print("recognize finished")
            print("Verification accuracy is %.2f" % (correct / cnt))


