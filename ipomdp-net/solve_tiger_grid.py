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
        num_frame = self.params.num_frame  # TODO: figure out if there should be such a variable
        num_particles = self.params.num_particles
        obs_len = self.params.obs_len
        step_size = self.step_size
        batch_size = self.batch_size

        placeholders = list()

        placeholders.append(tf.placeholder(
            dtype=tf.float32,
            shape=(batch_size, grid_n, grid_m),
            name='grid_map'))

        placeholders.append(tf.placeholder(
            dtype=tf.float32,
            shape=(batch_size, grid_n, grid_m),
            name='goal'))

        placeholders.append(tf.placeholder(
            dtype=tf.float32,
            shape=(batch_size, num_state),
            name='b0'))

        placeholders.append(tf.placeholder(
            dtype=tf.float32,
            shape=(batch_size, num_state*num_frame),  # ???
            name='ib0'))

        placeholders.append(tf.placeholder(
            dtype=tf.float32,
            shape=(batch_size, num_particles),
            name='particles'))

        placeholders.append(tf.placeholder(
            dtype=tf.float32,
            shape=(batch_size,),
            name='is_start'))

        placeholders.append(tf.placeholder(
            dtype=tf.int32,
            shape=(step_size, batch_size),
            name='act'))

        placeholders.append(tf.placeholder(
            dtype=tf.float32,
            shape=(step_size, batch_size, obs_len),
            name='local_obs'))

        placeholders.append(tf.placeholder(
            dtype=tf.float32,
            shape=(step_size, batch_size),
            name='weights'))

        placeholders.append(tf.placeholder(
            dtype=tf.int32,
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

        grid_map, goal, b0, ib0, particles, is_start, act_in, \
        obs_in, weight, act_label = self.placeholders

        # Type conversions.
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
            obs_func_self = FilterNet.f_obs_func(grid_map)
            obs_func_others = Level0FilterNet.f_obs_func(grid_map)

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
                b = FilterNet.interactive_belief_update(
                    particles=particles,
                    phys_belief_self=b,
                    action=act_in[step],
                    obs_func_self=obs_func_self,
                    obs_func_others=obs_func_others,
                    local_obs=obs_in[step],
                    grid_map=grid_map,
                    goal=goal,
                    params=self.params)

            # planner
            with tf.variable_scope("planner") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                action_pred = PlannerNet.policy(
                    q_val=q_val,
                    belief=b,
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


class Level0FilterNet:
    """

    """
    @staticmethod
    def f_obs_func(grid_map):
        """

        :param grid_map:
        :return: obs_func (observation function of the modeled agent)
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
    def f_trans_func(num_action):
        # Get transition kernel.
        initializer = tf.truncated_normal_initializer(
            mean=1.0/9.0,
            stddev=1.0/90.0,
            dtype=tf.float32)
        kernel = tf.get_variable(
            name='w_T_conv_lv0',
            shape=[3 * 3, num_action],
            initializer=initializer,
            dtype=tf.float32)

        # Enforce proper probability distribution by softmax
        # (i.e. values must sum to one)
        kernel = tf.nn.softmax(logits=kernel, axis=0)
        kernel = tf.reshape(kernel, [3, 3, 1, num_action], name='T_w')

        return kernel


class Level0PlannerNet:
    """

    """
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
        r = Level0PlannerNet.f_reward(
            grid_map=grid_map,
            goal=goal,
            num_action=params.num_action)

        # Get transition model T'.
        # It represents the transition model in the filter, but the weights
        # are not shared.
        kernel = Level0FilterNet.f_trans_func(params.num_action)
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
        q = tf.reduce_sum(
            input_tensor=q,
            axis=[1, 2],
            keep_dims=False)

        # low-level policy, f_pi
        action_pred = PlannerNet.f_pi(
            q_val=q,
            num_action=params.num_action)

        return action_pred


class FilterNet:
    """
    The network accounting for state estimation of the Interactive POMDP
    through time.
    """
    @staticmethod
    def f_obs_func(grid_map):
        """
        This method implements f_obs_func, outputs an observation model (Z).
        Fixed through time.
        :param grid_map:
        :return: obs_func (observation function of the modeling agent)
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
            shape=[3 * 3, num_action**2],
            initializer=initializer,
            dtype=tf.float32)

        # Enforce proper probability distribution by softmax
        # (i.e. values must sum to one)
        kernel = tf.nn.softmax(logits=kernel, axis=0)
        kernel = tf.reshape(kernel, [3, 3, 1, num_action**2], name='T_w')

        return kernel

    @staticmethod
    def physical_belief_update(belief, action_self, action_others, params):
        """

        :param belief: the modeling agent's belief over physical states at t-1
        :param action_self: the action of the modeling agent at t-1
        :param action_others: the action of the modeled agent at t-1
        :param params:
        :return: b_prime_a
        """
        # Get transition kernel (T_i)
        kernel = FilterNet.f_trans_func(params.num_action**2)

        # Using one fully-connected layer to map belief from num_state
        # to shape_grid_map
        belief_state2grid = nn_utils.fc_layers(
            input_data=belief,
            fc_params=np.array(
                [[20, 'lin'], [params.grid_n * params.grid_m, 'lin']]),
            names='fc_state2grid')
        b_img = tf.reshape(
            tensor=belief_state2grid,
            shape=[belief.get_shape().as_list()[0],  # batch_size
                   params.grid_n,
                   params.grid_m,
                   1],
            name='b_img')

        # Apply convolution corresponding to the transition function in
        # an MDP (f_trans_func).
        b_prime = tf.nn.conv2d(b_img, kernel, [1, 1, 1, 1], padding='SAME')

        # index into the appropriate channel of b_prime
        joint_action = (action_self, action_others)
        w_act = FilterNet.f_act(
            action=joint_action,
            num_action=params.num_actions**2)  # size of joint action space
        w_act = w_act[:, None, None]

        # Soft indexing
        b_prime_a = tf.reduce_sum(
            input_tensor=tf.multiply(b_prime, w_act),
            axis=[3],
            keepdims=False)

        return b_prime_a

    @staticmethod
    def others_model_update(particle_models, action, obs_func, params):
        """
        Particle propagation of the other agent's model.
        In this version, each particle belief is propagated separately, where
        each belief is convolved with the kernel by itself. There are N beliefs.
        :param particle_models: models factorized from particle interactive states
        :param action:
        :param obs_func:
        :param params:
        :return: particle_models
        """
        # 1) Propagation of particle frames.
        particle_frames = particle_models[:, :, params.num_state:]
        particle_frames = tf.tile(
            input=particle_frames,
            multiples=[1, params.num_obs, 1])

        # 2) Propagation of particle beliefs of the other agent.
        particle_others_beliefs = particle_models[:, :, :params.num_state]
        kernel = Level0FilterNet.f_trans_func(params.num_action)
        beliefs = tf.unstack(
            value=particle_others_beliefs,
            num=params.num_particles,
            axis=1)
        for particle_i in range(len(beliefs)):
            # Using one fully-connected layer to map belief from num_state
            # to shape_grid_map
            belief_state2grid = nn_utils.fc_layers(
                input_data=beliefs[particle_i],
                fc_params=np.array(
                    [[20, 'lin'], [params.grid_n * params.grid_m, 'lin']]),
                names='fc_state2grid')
            b_img = tf.reshape(
                tensor=belief_state2grid,
                shape=[beliefs[particle_i].get_shape().as_list()[0],  # batch
                       params.grid_n, params.grid_m, 1],
                name='b_img')

            # Apply convolution corresponding to the transition function in
            # an MDP (f_trans_func).
            b_prime = tf.nn.conv2d(b_img, kernel, [1, 1, 1, 1], padding='SAME')

            # Index into the channel of b_prime corresponding to previous action
            w_act = FilterNet.f_act(
                action=action,
                num_action=params.num_actions)  # size of joint action space
            w_act = w_act[:, None, None]
            # Soft indexing.
            b_prime_a = tf.reduce_sum(
                input_tensor=tf.multiply(b_prime, w_act),
                axis=[3],
                keepdims=False)

            # Correct belief with all j's available observation.
            # Without soft-indexing with j's observation, there are |O_j|
            # output belief images.
            b_next_imgs = tf.multiply(b_prime_a, obs_func)
            print("Shape of output belief: ", b_next_imgs.get_shape().as_list())

            b_grid2state = tf.reshape(
                tensor=b_next_imgs,
                shape=[b_next_imgs.get_shape().as_list()[1],  # batch_size (?)
                       params.num_obs*params.grid_n*params.grid_m],
                name='fc_grid2state')
            beliefs[particle_i] = nn_utils.fc_layers(
                input_data=b_grid2state,
                fc_params=np.array(
                    [[10, 'lin'], [params.num_state*params.num_obs], 'lin']),
                names='b_next')
            beliefs[particle_i] = tf.reshape(
                tensor=beliefs[particle_i],
                shape=[beliefs[particle_i].get_shape().as_list()[0],
                       params.num_obs,
                       params.num_state],
                name='b_per_o')
            beliefs[particle_i] = tf.div(  # normalize to make each particle belief sum to 1.
                x=beliefs[particle_i],
                y=tf.reduce_sum(
                    input_tensor=beliefs[particle_i],
                    axis=[2],
                    keepdims=True) + 1e-8)

        particle_others_beliefs = tf.stack(values=beliefs, axis=1)
        print("Shape of output beliefs: ",
              particle_others_beliefs.get_shape().as_list())
        particle_others_beliefs = tf.reshape(
            tensor=particle_others_beliefs,
            shape=(particle_others_beliefs.get_shape().as_list()[0],
                   params.num_particles*params.num_obs,
                   params.num_state))
        print("Shape of output beliefs after reshape: ",
              particle_others_beliefs.get_shape().as_list())

        # 3) Integrate frames and beliefs to rebuild models.
        particle_models = tf.concat(
            values=[particle_others_beliefs, particle_frames],
            axis=-1)

        return particle_models

    @staticmethod
    def others_model_update_2(particle_models, action, obs_func, params):
        """
        Particle propagation of the other agent's model.
        In this version, the N particle beliefs are regarded as one belief
        image, where the dimension of channels is set to N. The depth of the
        kernel is also duplicated to N. Then we apply depth-wise 2D convolution.
        :param particle_models: models factorized from particle interactive states
        :param action:
        :param obs_func:
        :param params:
        :return: particle_models
        """
        # 1) Propagation of particle frames.
        particle_frames = particle_models[:, :, params.num_state:]
        particle_frames = tf.tile(
            input=particle_frames,
            multiples=[1, params.num_obs, 1])

        # 2) Propagation of particle beliefs of the other agent.
        particle_others_beliefs = particle_models[:, :, :params.num_state]
        particle_others_beliefs = tf.reshape(
            tensor=particle_others_beliefs,
            shape=(particle_others_beliefs.get_shape().as_list()[0],
                   params.num_particles*params.num_state),  # (batch, N * |S|)
            name='fc_input_bj')
        beliefs_state2grid = nn_utils.fc_layers(
            input_data=particle_others_beliefs,
            fc_params=np.array(
                [[params.num_particles*20, 'lin'],
                 [params.num_particles*params.grid_n*params.grid_m]]),
            names='fc_state2grid')
        particle_b_imgs = tf.reshape(
            tensor=beliefs_state2grid,
            shape=(beliefs_state2grid.get_shape().as_list()[0],
                   params.grid_n,
                   params.grid_m,
                   params.num_particles),
            name='b_img_n_channel')  # N 1-channel imgs -> 1 N-channel img

        kernel = Level0FilterNet.f_trans_func(params.num_action)
        kernel = tf.tile(  # duplicate the kernel N times along 'in_channels'
            input=kernel,
            multiples=[1, 1, params.num_particles, 1],
            name='dw_kernel')

        b_imgs_prime = tf.nn.depthwise_conv2d(
            input=particle_b_imgs,
            filter=kernel,
            strides=[1, 1, 1, 1],
            padding='SAME')

        w_act_j = FilterNet.f_act(
            action=action,
            num_action=params.num_action)
        w_act_j = w_act_j[:, None, None]
        b_imgs_prime_a = tf.reduce_sum(
            input_tensor=tf.multiply(b_imgs_prime, w_act_j),
            axis=[3],
            keepdims=False)

        # Correct belief with all j's available observation.
        # Without soft-indexing with j's observation, there are |O_j|
        # output belief images.
        b_next_imgs = tf.multiply(b_imgs_prime_a, obs_func)  # (batch_size, ?, ?, |S|)
        print("Shape of output belief: ", b_next_imgs.get_shape().as_list())

        beliefs_grid2state = tf.reshape(
            tensor=b_next_imgs,
            shape=(b_next_imgs.get_shape().as_list()[0],
                   params.num_particles*params.num_obs*
                   params.grid_n*params.grid_m),
            name='fc_input_b_imgs_j')
        particle_others_beliefs = nn_utils.fc_layers(
            input_data=beliefs_grid2state,
            fc_params=np.array(
                [[params.num_particles*params.num_obs*20, 'lin'],
                 [params.num_particles*params.num_obs*params.num_state, 'lin']]),
            names='fc_grid2state')
        particle_others_beliefs = tf.reshape(
            tensor=particle_others_beliefs,
            shape=(particle_others_beliefs.get_shape().as_list(),
                   params.num_particles*params.num_obs,
                   params.num_state),
            name='particle_bs')

        # Normalization: make each particle belief sum to 1.
        particle_others_beliefs = tf.div(
            x=particle_others_beliefs,
            y=tf.reduce_sum(
                input_tensor=particle_others_beliefs,
                axis=[2],
                keepdims=True) + 1e-8)

        # 3) Integrate frames and beliefs to rebuild models.
        particle_models = tf.concat(
            values=[particle_others_beliefs, particle_frames],
            axis=-1)

        return particle_models

    @staticmethod
    def interactive_belief_update(particles,
                                  action,
                                  obs_func_self,
                                  obs_func_others,
                                  local_obs,
                                  grid_map,
                                  goal,
                                  params):
        """
        Interactive belief update
        :param particles: [istate, weight] with shape (batch_size, num_particles)
        :param action: i's action at time step t-1
        :param obs_func_self: i's observation function, getting from grid_map
        :param obs_func_others: j's observation function, getting from grid_map (?)
        :param local_obs: i's observation at time step t
        :param grid_map: the given environmental setting of the domain
        :param goal: goal states which both agent aim to reach
        :param params:
        :return: particles, phys_belief_self, ibelief
        """
        # Initialize variables to be used in interactive belief update.
        batch_size = params.batch_size
        num_particles = params.num_particles

        particle_istates = particles[:, :, :-1]
        particle_phys_states = particle_istates[:, :, :params.num_state]
        particle_others_models = particle_istates[:, :, params.num_state:]
        particle_weights = particles[:, :, -1]

        # Step 0: Preparation.
        # Predict the other agent's action by simulating its value function.
        particle_others_beliefs = particle_others_models[:, :, :params.num_state]
        b_others_mean = tf.reduce_mean(  # TODO: not correct
            input_tensor=particle_others_beliefs,
            axis=0,
            keepdims=False)
        q_val_others_pred = Level0PlannerNet.value_iteration(
            grid_map=grid_map,
            goal=goal,
            params=params)
        a_others_pred = Level0PlannerNet.policy(
            q_val=q_val_others_pred,
            belief=b_others_mean,
            params=params)

        # Step 1: Propagation.
        # 1) Propagate physical particle states.
        #    belief_t-1 -> belief_t -> particle_states_t
        # i. Belief update via convolution.
        phys_belief_self = 
        phys_belief_self = FilterNet.physical_belief_update(
            belief=phys_belief_self,
            action_self=action,
            action_others=a_others_pred,
            params=params)
        # ii. Re-sampling particles according to updated belief.
        log_belief = tf.log(phys_belief_self)
        particle_physical_states = tf.multinomial(
            logits=log_belief,
            num_samples=params.num_particles*params.num_obs)  # (N*|O_j|)

        # 2) Propagate particle models of others.
        particle_others_models = FilterNet.others_model_update(
            particle_models=particle_others_models,
            action=a_others_pred,
            obs_func=obs_func_others,
            params=params)

        # 3) Integrate physical states and the other agent's models to be
        #    interactive states.
        particle_istates = tf.concat(
            values=[particle_physical_states, particle_others_models],
            axis=-1)

        # Step 2: Weighting.
        # 1) Previous weights are weighted by j's observation function
        #    considering all possible observations of j.
        # ----------------------------------------------------------------
        # TODO: Figure out the correct way of getting weights from Oj.
        weight_by_oj = tf.multiply(phys_belief_self, obs_func_others)
        # ----------------------------------------------------------------
        # 2) Updated weights are weighted by i's observation function
        #    given i's local observation.
        w_obs = FilterNet.f_obs(local_obs)
        w_obs = w_obs[:, None, None]
        weight_by_oi = tf.reduce_sum(
            input_tensor=tf.multiply(obs_func_self, w_obs),
            axis=[3],
            keepdims=False)
        weights_updated = tf.multiply(particle_weights, weight_by_oj)
        weights_updated = tf.multiply(weights_updated, weight_by_oi)
        # Normalize the updated particle weights.
        weights_updated = tf.div(
            x=weights_updated,
            y=tf.reduce_sum(
                input_tensor=weights_updated,
                axis=0,
                keepdims=False))

        # Step 3: Re-sampling.
        log_weights = tf.log(weights_updated)
        indices = tf.cast(tf.multinomial(
            logits=log_weights,
            num_samples=num_particles))
        helper = tf.range(
            start=0,
            limit=batch_size*num_particles,
            delta=num_particles,
            dtype=tf.int32
        )
        indices = indices + tf.expand_dims(helper, axis=1)
        # TODO: Figure out the exact shape of each interactive state.
        # ----------------------------------------------
        particle_istates = tf.reshape(
            tensor=particle_istates,
            shape=(batch_size*num_particles,))
        # ----------------------------------------------
        particle_istates = tf.gather(
            params=particle_istates,
            indices=indices,
            axis=0)
        particle_weights = tf.constant(
            tf.div(1.0, num_particles),
            shape=(batch_size, num_particles),
            dtype=tf.float32)
        particle_weights = tf.expand_dims(
            input=particle_weights,
            axis=-1)

        # Pack the particle states and particle weights in pairs as particles.
        particles = tf.concat(
            values=[particle_istates, particle_weights],
            axis=-1)

        return particles, phys_belief_self, ibelief
