
'''
python freeze_graph.py --ckpt-path "./end2end/z_donk/ep_6_loss_3.4e+02_bins_15.ckpt"
                       --graph-path "./end2end/z_donk/model.ckpt.meta"
                       --out-path "./end2end/z_donk/frozne.pb"
                       --outputs "donkey/throttle/Sigmoid" "donkey/steering_prediction"
'''
import tensorflow as tf
from tensorflow.python.layers.core import dense, dropout, flatten
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer as xavier
from tqdm import trange, tqdm
import numpy as np
import os
import json
from freeze_graph import freeze_meta, write_tensor_dict_to_json, load_tensor_names
import common

# If training on a multi GPU system these are used to select which GPU to use
# and what fraction to use
# common.setup_gpu_options(gpu_mask="0", gpu_fraction=5/11)

'''
intput_or_output = "input" or "output"
key: "descriptive name"
tensor: the tensorflow tensor
'''

class Model:
    def __init__(self, in_shape, classes, lr=0.001):
        '''
        classes:  List of class names or integers corrisponding to each class being classified
                  by the network. ie: ['left', 'straight', 'right'] or [0, 1, 2]
        '''
        # Define classes
        self.num_bins = len(classes)
        self.classes = np.array(classes, np.float32)
        self.class_lookup = [c for c in classes]

        # Define model
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, shape=[None,]+in_shape, name="input")
        self.y_steering = tf.placeholder(tf.int32,   shape=(None,))
        self.y_throttle = tf.placeholder(tf.float32, shape=(None,))
        self._training = tf.placeholder(tf.bool)
        self.training  = tf.get_variable("training", dtype=tf.bool,
                                          initializer=True, trainable=False)
        self.set_training = self.training.assign(self._training)

        relu    = tf.nn.relu
        sigmoid = tf.nn.sigmoid
        with tf.name_scope("donkey"):
            #            input   num  conv   stride   pad
            conv = conv2d(self.x, 24,  (5,5), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="conv1")
            conv = conv2d( conv,   32,  (5,5), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="conv2")
            conv = conv2d( conv,   64,  (5,5), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="conv3")
            conv = conv2d( conv,   64,  (3,3), (2,2),  "same", activation=relu, kernel_initializer=xavier(), name="conv4")
            conv = conv2d( conv,   64,  (3,3), (1,1),  "same", activation=relu, kernel_initializer=xavier(), name="conv5")
            conv = flatten(conv)
            #             in   num
            conv = dense(  conv, 100, activation=relu, kernel_initializer=xavier(), name="fc1")
            conv = dropout(conv, rate=0.1, training=self.training)

            conv = dense(  conv, 50, activation=relu, kernel_initializer=xavier(), name="fc2")
            conv = dropout(conv, rate=0.1, training=self.training)

            # Steering
            self.logits = dense(conv, self.num_bins, activation=None, kernel_initializer=xavier(), name="logits")
            self.steering_probs = tf.nn.softmax(self.logits, name="steeringi_probs")
            self.steering_prediction = tf.reduce_sum(tf.multiply(self.steering_probs, self.classes),
                                                     axis=1, name="steering_prediction")
            # Throttle
            self.throttle = dense(conv, 1, sigmoid, kernel_initializer=xavier(), name="throttle")

            # keep tensor names for easy freezing/loading later
            self._TENSOR_DICT = {common._IMAGE_INPUT : self.x.name,
                                 common._STEERING_PREDICTION : self.steering_prediction.name,
                                 common._STEERING_PROBS      : self.steering_probs.name,
                                 common._THROTTLE_PREDICTION : self.throttle.name}

        with tf.name_scope("loss"):
            self.loss_steering = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_steering, logits=self.logits)
            self.loss_steering = tf.reduce_mean(self.loss_steering)
            self.loss_throttle = tf.reduce_mean((self.throttle - self.y_throttle)**2)
            self.loss = 0.9*self.loss_steering + 0.001*self.loss_throttle

        tf.summary.scalar("weighted_loss", self.loss)
        tf.summary.scalar("steering_loss", self.loss_steering)
        tf.summary.scalar("throttle_loss", self.loss_throttle)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_step = optimizer.minimize(self.loss)

        self.init_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def train(self, train_gen, test_gen, save_dir, epochs=10, restart_ckpt=None):
        return_info = {"save_dir"          : save_dir}

        with tf.Session(config=tf.ConfigProto(gpu_options=common._GPU_OPTIONS)) as sess:
            merge = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(save_dir+"/logdir/train", sess.graph)
            test_writer  = tf.summary.FileWriter(save_dir+"/logdir/test")

            # Init
            sess.run(self.init_vars)

            # Used to keep track of the best performing iteration
            self._best_loss = 10**9

            # Saveing the meta graph stores the arcetecture of the network,
            # not the weights.
            abs_graph_path = self._save_meta_graph(save_dir)
            return_info["graph_path"] = abs_graph_path

            global_step = 0
            for e in range(epochs):
                # Toggle dropout
                sess.run(self.set_training, feed_dict={self._training: True})

                train_gen.reset()
                t_train = range(train_gen.steps_per_epoch)
                print(f"Training Epoch: {e+1}")
                for step in t_train:
                    images, steering, throttle = train_gen.get_next_batch()
                    _, summary = sess.run([self.train_step, merge],
                                feed_dict={self.x         : images,
                                           self.y_steering: steering,
                                           self.y_throttle: throttle})
                    train_writer.add_summary(summary, global_step)
                    global_step += 1


                sess.run(self.set_training, feed_dict={self._training: False})
                test_gen.reset()
                t_test = range(test_gen.steps_per_epoch)
                print('Testing')
                _test_loss = []
                for _ in t_test:
                    images, steering, throttle = test_gen.get_next_batch()
                    _loss, summary = sess.run([self.loss, merge],
                                feed_dict={self.x         : images,
                                           self.y_steering: steering,
                                           self.y_throttle: throttle})
                    _test_loss.append(_loss)
                    test_writer.add_summary(summary, global_step)
                    global_step += 1

                abs_ckpt_path = self._save_best_ckpt(sess, e, _test_loss, save_dir)
                return_info["best_ckpt"] = abs_ckpt_path

                cur_mean_loss = np.mean(_test_loss)
                print("-"*50)

        print(f"Done, final best loss: {self._best_loss:0.3}")
        return_info["best_loss"] = float(self._best_loss)
        json.dump(return_info, open(save_dir+"/return_info.json", 'w'))

    """
    TODO: Move graph save to seperate method
    """
    def _save_best_ckpt(self, sess, epoch, test_losses, save_dir):
        cur_mean_loss = np.mean(test_losses)
        self._best_loss = cur_mean_loss
        path = f"{save_dir}/ep_{epoch+1:03d}.ckpt"
        best_ckpt = self.saver.save(sess, path, write_meta_graph=False)
        return os.path.abspath(best_ckpt)

    def _save_meta_graph(self, save_dir):
        path = f"{save_dir}/graph.meta"
        self.saver.export_meta_graph(path)
        path = write_tensor_dict_to_json(save_dir, self._TENSOR_DICT)
        return os.path.abspath(path)
# Static methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

