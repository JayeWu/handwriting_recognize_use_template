from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


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

        return loss, acc


