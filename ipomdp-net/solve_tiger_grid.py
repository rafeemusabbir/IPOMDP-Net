from tensorpack import graph_builder
import tensorflow as tf
import numpy as np

from ipomdpnet import IPOMDPNet
import nn_utils


class IPOMDPNetTigerGrid(IPOMDPNet):
    """
    Class implementing a QMDP-Net for the grid navigation domain
    """
    def build_placeholders(self):
        """
        Creates placeholders for all inputs in self.placeholders
        """
        grid_n = self.params.grid_n
        grid_m = self.params.grid_m
        num_state = self.params.num_state
        obs_len = self.params.obs_len
        step_size = self.step_size
        batch_size = self.batch_size

        placeholders = list()

        placeholders.append(tf.placeholder(dtype=tf.float32,
                                           shape=(batch_size, grid_n, grid_m),
                                           name='grid_map'))

        placeholders.append(tf.placeholder(dtype=tf.float32,
                                           shape=(batch_size, grid_n, grid_m),
                                           name='goal'))

        placeholders.append(tf.placeholder(dtype=tf.float32,
                                           shape=(batch_size, num_state),
                                           name='b0'))

        placeholders.append(tf.placeholder(dtype=tf.float32,
                                           shape=(batch_size,),
                                           name='is_start'))

        placeholders.append(tf.placeholder(dtype=tf.int32,
                                           shape=(step_size, batch_size),
                                           name='act'))

        placeholders.append(tf.placeholder(dtype=tf.float32,
                                           shape=(step_size,
                                                  batch_size,
                                                  obs_len),
                                           name='local_obs'))

        placeholders.append(tf.placeholder(dtype=tf.float32,
                                           shape=(step_size, batch_size),
                                           name='weights'))

        placeholders.append(tf.placeholder(dtype=tf.int32,
                                           shape=(step_size, batch_size),
                                           name='label_act'))

        self.placeholders = placeholders

    def build_inference(self, reuse=False):
        """
        Creates placeholders, ops for inference and loss
        Unfolds filter and planner through time
        Also creates an op to update the belief. It should be always evaluated
        together with the loss.
        :param reuse: reuse variables if True
        :return: None
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        self.build_placeholders()

        grid_map, goal, b0, is_start, act_in, obs_in, weight, act_label = \
            self.placeholders  # TODO clean up

        # Type   conversions
        is_start = tf.reshape(
            tensor=is_start,
            shape=[self.batch_size] + [1] * (b0.get_shape().ndims - 1))

        # Initialize the series of predicted actions
        outputs = list()

        # pre-compute context, fixed through time
        with tf.variable_scope("planner"):
            q_val, _, _ = PlannerNet.value_iteration(
                grid_map=grid_map,
                goal=goal,
                params=self.params)
        with tf.variable_scope("filter"):
            obs_func = FilterNet.f_obs_func(
                grid_map=grid_map)

        self.context_tensors = [q_val, obs_func]

        # Create variable for hidden belief
        # (equivalent to the hidden state of an RNN)
        self.belief = tf.Variable(
            np.zeros(b0.get_shape().as_list(), 'f'),
            trainable=False,
            name="hidden_belief")

        # figure out current b. b = b0 if isstart else blast
        b = (b0 * is_start) + (self.belief * (1-is_start))

        for step in range(self.step_size):
            # filter
            with tf.variable_scope("filter") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                b_img, b = FilterNet.belief_update(
                    obs_func=obs_func,
                    belief=b,
                    action=act_in[step],
                    local_obs=obs_in[step],
                    params=self.params)

            # planner
            with tf.variable_scope("planner") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                action_pred = PlannerNet.policy(
                    q_val=q_val,
                    belief=b_img,
                    params=self.params)
                outputs.append(action_pred)

        # create op that updates the belief
        self.update_belief_op = self.belief.assign(b)

        # Compute loss (cross-entropy)
        # Shape [step_size, batch_size, num_action]
        logits = tf.stack(values=outputs, axis=0)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=act_label)

        # Weight loss. weights are 0.0 for steps after the end of a trajectory,
        # otherwise 1.0
        loss = loss * weight
        loss = tf.reduce_mean(input_tensor=loss,
                              axis=[0, 1],
                              name='xentropy')

        self.logits = logits
        self.loss = loss

    def build_train(self, init_learning_rate):
        """
        """
        # Decay learning rate by manually incrementing decay_step
        decay_step = tf.Variable(0.0, name='decay_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            learning_rate=init_learning_rate,
            global_step=decay_step,
            decay_steps=1,
            decay_rate=0.9,
            staircase=True,
            name="learning_rate")

        trainable_variables = tf.trainable_variables()

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            decay=0.9,
            centered=True)
        # optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate=learning_rate)
        # clip gradients
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(
            t_list=grads,
            clip_norm=1.0,
            use_norm=tf.global_norm(grads))
        # grads_2, _ = tf.clip_by_norm(
        #     t=grads,
        #     clip_norm=1.0)

        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        self.decay_step = decay_step
        self.learning_rate = learning_rate
        self.train_op = train_op


class IPOMDPNetPolicy:
    """
    Policy wrapper for QMDPNet. Implements two functions: reset and eval.
    """
    def __init__(self, network, sess, params):
        self.network = network
        self.params = params
        self.sess = sess

        self.belief_state = None
        self.env_img = None
        self.goal_img = None

        assert self.network.batch_size == 1 and self.network.step_size == 1

    def reset(self, env_img, goal_img, belief_state):
        # TODO
        """

        :param belief_state:
        :param env_img:
        :param goal_img:
        :return:
        """
        grid_n = self.network.params.grid_n
        grid_m = self.network.params.grid_m

        self.belief_state = belief_state.reshape([1, self.params.num_state])
        self.env_img = env_img.reshape([1, grid_n, grid_m])
        self.goal_img = goal_img.reshape([1, grid_n, grid_m])

        self.sess.run(tf.assign(self.network.belief, self.belief_state))

    def eval(self, last_act, last_obs):
        # TODO
        """

        :param last_act:
        :param last_obs:
        :return:
        """
        is_start = np.array([0])
        last_act = np.reshape(last_act, [1, 1])
        last_obs = np.reshape(last_obs, [1, 1, self.network.params.obs_len])

        # input data. do not neet weight and label for prediction
        data = [self.env_img, self.goal_img, self.belief_state,
                is_start, last_act, last_obs]
        feed_dict = {self.network.placeholders[i]: data[i]
                     for i in range(len(self.network.placeholders)-2)}

        # evaluate QMDPNet
        logits, _ = self.sess.run([self.network.logits,
                                   self.network.update_belief_op],
                                  feed_dict=feed_dict)
        act = np.argmax(logits.flatten())

        return act


class PlannerNet:
    @staticmethod
    def f_reward(grid_map, goal, num_action):
        """

        :param grid_map:
        :param goal:
        :param num_action:
        :return:
        """
        theta = tf.stack([grid_map, goal], axis=3)
        r = nn_utils.conv_layers(
            input_data=theta,
            conv_params=np.array([[3, 200, 'relu'],
                                  [3, 150, 'lin'],
                                  [1, num_action, 'lin']]),
            names="R_conv")
        return r

    @staticmethod
    def value_iteration(grid_map, goal, params):
        """
        Builds neural network implementing value iteration. this is the first
        part of planner module. Fixed through time.
        inputs: grid_map (batch x N x N) and goal(batch)
        returns: Q_K, and optionally: R, list of Q_i
        """
        # Obtain immediate reward based on reward function.
        r = PlannerNet.f_reward(grid_map=grid_map,
                                goal=goal,
                                num_action=params.num_action)

        # Get transition model T'.
        # It represents the transition model in the filter, but the weights
        # are not shared.
        kernel = FilterNet.f_trans_func(num_action=params.num_action)
        # Initialize the value image.
        val = tf.zeros(grid_map.get_shape().as_list() + [1])
        q_val = None

        # repeat value iteration K times
        for i in range(params.K):
            # Apply transition and sum
            q_val = tf.nn.conv2d(val, kernel, [1, 1, 1, 1], padding='SAME')
            # print("Q value shape:", q_val.shape)
            q_val = q_val + r
            val = tf.reduce_max(input_tensor=q_val,
                                axis=[3],
                                keep_dims=True)

        return q_val, val, r

    @staticmethod
    def f_pi(q_val, num_action):
        action_pred = nn_utils.fc_layers(
            input_data=q_val,
            fc_params=np.array([[num_action, 'lin']]),
            names="pi_fc")
        return action_pred

    @staticmethod
    def policy(q_val, belief, params):
        """
        second part of planner module
        :param q_val: input Q_K after value iteration
        :param belief: belief at current step
        :param params: params
        :return: action_pred, vector with num_action elements, each has the
        """
        # Weight q_val by the belief
        b_tiled = tf.tile(tf.expand_dims(belief, 3),
                          [1, 1, 1, params.num_action])
        q = tf.multiply(q_val, b_tiled)
        # Sum over states
        q = tf.reduce_sum(input_tensor=q,
                          axis=[1, 2],
                          keep_dims=False)

        # low-level policy, f_pi
        action_pred = PlannerNet.f_pi(q_val=q,
                                      num_action=params.num_action)

        return action_pred


class FilterNet:
    @staticmethod
    def f_obs_func(grid_map):
        """
        This implements f_Z, outputs an observation model (Z). Fixed through
        time.
        inputs: map (NxN array)
        returns: Z
        """
        # CNN: theta -> Z
        grid_map = tf.expand_dims(grid_map, -1)
        obs_func = nn_utils.conv_layers(
            input_data=grid_map,
            conv_params=np.array([[3, 200, 'lin'],
                                  [3, 100, 'lin'],
                                  [1, 17, 'sig']]),
            names='Z_conv')

        # Normalize over observations
        normalized_sum = tf.reduce_sum(
            input_tensor=obs_func,
            axis=[3],
            keep_dims=True)
        # Avoid division by zero.
        obs_func = tf.div(obs_func, normalized_sum + 1e-8)

        return obs_func

    @staticmethod
    def f_act(action, num_action):
        """
        This method is for soft indexing actions.
        :param action:
        :param num_action:
        :return:
        """
        w_act = tf.one_hot(action, num_action)
        return w_act

    @staticmethod
    def f_obs(local_obs):
        """
        This method is for soft indexing observations.
        :param local_obs:
        :return:
        """
        w_obs = nn_utils.fc_layers(
            input_data=local_obs,
            fc_params=np.array([[17, 'tanh'], [17, 'smax']]),
            names="O_fc")
        return w_obs

    @staticmethod
    def f_trans_func(num_action):
        # Get transition kernel
        initializer = tf.truncated_normal_initializer(
            mean=1.0/9.0,
            stddev=1.0/90.0,
            dtype=tf.float32)
        kernel = tf.get_variable(
            name='w_T_conv',
            shape=[3 * 3, num_action],
            initializer=initializer,
            dtype=tf.float32)

        # Enforce proper probability distribution by softmax
        # (i.e. values must sum to one)
        kernel = tf.nn.softmax(logits=kernel, axis=0)
        kernel = tf.reshape(kernel, [3, 3, 1, num_action], name='T_w')

        return kernel

    @staticmethod
    def belief_update(obs_func, belief, action, local_obs, params):
        """
        Belief update in the filter module with pre-computed Z.
        :param obs_func:
        :param params:
        :param belief: belief (b_i), [batch, N, M, 1]
        :param action: action input (a_i)
        :param local_obs: observation input (o_i)
        :return: updated belief b_(i+1)
        """
        # step 1: update belief with transition
        # get transition kernel (T)
        kernel = FilterNet.f_trans_func(num_action=params.num_action)

        # Using one fully-connected layer to map belief from num_state
        # to shape_grid_map
        belief_state2grid = nn_utils.fc_layers(
            input_data=belief,
            fc_params=np.array([[20, 'lin'],
                                [params.grid_n*params.grid_m, 'lin']]),
            names='fc_state2grid')
        b_img = tf.reshape(tensor=belief_state2grid,
                           shape=[belief.get_shape().as_list()[0],  # batch_size
                                  params.grid_n,
                                  params.grid_m,
                                  1],
                           name='b_img')

        # apply convolution which corresponds to the transition function in
        # an MDP (f_trans_func)
        b_prime = tf.nn.conv2d(b_img, kernel, [1, 1, 1, 1], padding='SAME')

        # index into the appropriate channel of b_prime
        w_act = FilterNet.f_act(action, params.num_action)
        w_act = w_act[:, None, None]
        # soft indexing
        b_prime_a = tf.reduce_sum(
            input_tensor=tf.multiply(b_prime, w_act),
            axis=[3],
            keep_dims=False)

        # TODO there was this line. Does it make a difference with softmax?
        # b_prime_a = tf.abs(b_prime_a)

        # step 2: Update belief with observation
        # Get observation probabilities for the observation input by
        # soft indexing
        w_obs = FilterNet.f_obs(local_obs)
        w_obs = w_obs[:, None, None]

        # Soft indexing
        obs_prob = tf.reduce_sum(
            input_tensor=tf.multiply(obs_func, w_obs),
            axis=[3],
            keep_dims=False)

        b_next_img = tf.multiply(b_prime_a, obs_prob)
        b_grid2state = tf.reshape(
            tensor=b_next_img,
            shape=[b_next_img.get_shape().as_list()[0],
                   params.grid_n*params.grid_m],
            name='fc_grid2state')
        b_next = nn_utils.fc_layers(
            input_data=b_grid2state,
            fc_params=np.array([[10, 'lin'],
                                [params.num_state, 'lin']]),
            names='b_next')

        # step 3: normalize over the state space
        # add small number to avoid division by zero
        b_next_img = tf.div(
            x=b_next_img,
            y=tf.reduce_sum(input_tensor=b_next_img,
                            axis=[1, 2],
                            keep_dims=True) + 1e-8)
        b_next = tf.div(
            x=b_next,
            y=tf.reduce_sum(input_tensor=b_next,
                            axis=[1],
                            keep_dims=True) + 1e-8)

        return b_next_img, b_next
