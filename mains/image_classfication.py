import tensorflow as tf
import os
from PIL import Image
from tensorflow.python.platform import gfile
import numpy as np

from data_loader.data_generator import DataGenerator
from models.imc_model import IMCModel
from trainers.imc_trainer import IMCTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import Logger


def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # config = process_config('configs/imc.json')
    create_dirs([config.summary_dir, config.checkpoint_dir])
    sess = tf.Session()
    # create instance of the model you want
    model = IMCModel(config)
    # load model if exist
    model.load(sess)
    # create your data generator
    imc_data = DataGenerator(config)

    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = IMCTrainer(sess, model, imc_data, config, logger)
    # # here you train your model
    trainer.train()

    # trainer.recognize()
    recogflag = input("recognize your image now? image in {} Y/N".format(config.dir_af))
    if recogflag == "Y" or recogflag == "y" or recogflag == '':
        recognize_image(config.dir_af, config.checkpoint_dir, config.graph_dir, config.image_size)
    else:
        print("Process end!")


def recognize_image(dir_af, ckpt_dir, graph_dir, image_size):
    # 手写图片预处理
    with tf.Session() as sess:
        ''' 加载保存好的图
        如果不进行这一步，也可以按照训练模型时一样的定义需要的张量
        '''
        with gfile.FastGFile(graph_dir+"/train.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        predict_op = sess.graph.get_tensor_by_name("predict_op:0")  # 取出预测值的张量
        X = sess.graph.get_tensor_by_name("X:0")  # 取出输入数据的张量
        p_keep_conv = sess.graph.get_tensor_by_name("p_keep_conv:0")  #
        p_keep_hidden = sess.graph.get_tensor_by_name("p_keep_hidden:0")  #

        # 加载保存好的模型
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('no ckpt files')
            exit(0)
        # 开始识别图片
        files = os.listdir(dir_af)
        cnt = len(files)
        correct = 0
        for i in range(cnt):
            actual_label = int(files[i][0])
            files[i] = dir_af + "/" + files[i]
            img = Image.open(files[i])  # 读取要识别的图片
            print("input: ", files[i])
            imga = np.array(img).reshape(-1, image_size[0], image_size[1], 1)
            # feed 数据给 张量predict_op
            prediction = predict_op.eval(feed_dict={X: imga, p_keep_conv: 1.0, p_keep_hidden: 1.0})
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


if __name__ == "__main__":
    main()


