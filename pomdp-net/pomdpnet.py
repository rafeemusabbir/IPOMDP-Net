from tensorpack import graph_builder
import tensorflow as tf
import numpy as np


class POMDPNet:
    """

    """
    def __init__(self, params, batch_size=1, step_size=1):
        """

        :param params: dotdict describing the domain and network hyperparameters
        :param batch_size: mini-batch size for training. batch_size = 1 for eval
        :param step_size: limit the number of steps for backpropagation through
                        time. step_size=1 for eval
        """
        self.params = params
        self.batch_size = batch_size
        self.step_size = step_size

        self.placeholders = None
        self.context_tensors = None
        self.belief = None  # used in QMDP setting
        self.update_belief_op = None # used in QMDP setting
        self.l0belief = None  # used in I-POMDP setting
        self.ibelief = None  # used in I-POMDP setting
        self.update_l0belief_op = None  # used in I-POMDP setting
        self.update_ibelief_op = None  # used in I-POMDP setting
        self.logits = None
        self.loss = None

        self.decay_step = None
        self.learning_rate = None
        self.train_op = None

    def build_placeholders(self):
        """
        This method creates placeholders for all inputs in self.placeholders
        :return: None
        """
        placeholders = list()

        self.placeholders = placeholders

    def build_inference(self, reuse=False):
        """
        This method creates placeholders and ops for inference and loss.
        It also unfolds filter and planner through time.
        Besides, it creates an op to update the belief.
        :param reuse:
        :return: None
        """

    def build_training(self, init_learning_rate):
        """

        :param init_learning_rate: The initial learning rate set manually.
        :return: None
        """
        # Decay learning rate by manually incrementing "decay_step"
        decay_step = tf.Variable(0.0, name='decay_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            init_learning_rate, decay_step, 1, 0.8, staircase=True,
            name='learning_rate'
        )

        trainable_vars = tf.trainable_variables()

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                              decay=0.9)

        grads = tf.gradients(self.loss, trainable_vars)
        grads, _ = tf.clip_by_global_norm(
            t_list=grads, clip_norm=1.0, use_norm=tf.global_norm(grads)
        )

        train_op = optimizer.apply_gradients(zip(grads, trainable_vars))

        self.decay_step = decay_step
        self.learning_rate = learning_rate
        self.train_op = train_op





