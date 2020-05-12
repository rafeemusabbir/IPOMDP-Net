from tensorpack import graph_builder
import tensorflow as tf
import numpy as np

from ipomdpnet import IPOMDPNet
import nn_utils


class IPOMDPNetTigerGrid(IPOMDPNet):
    """
    Class implementing a IPOMDP-Net for the grid navigation domain
    """
    def build_placeholders(self):
        """
        Creates placeholders for all inputs in self.placeholders
        """
        grid_n = self.params.grid_n
        grid_m = self.params.grid_m
        num_l0_phys_state = self.params.num_l0_phys_state
        num_inter_phys_state = self.params.num_inter_phys_state
        step_size = self.step_size
        batch_size = self.batch_size

        placeholders = list()

        placeholders.append(tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=(batch_size, grid_n, grid_m),
            name='grid_map'))

        placeholders.append(tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=(batch_size, grid_n, grid_m),
            name='terminal_map'))

        placeholders.append(tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=(batch_size, num_l0_phys_state),
            name='b0_other'))

        placeholders.append(tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=(batch_size, num_inter_phys_state),
            name='b0_self'))

        placeholders.append(tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=(batch_size,),
            name='is_start'))

        placeholders.append(tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=(step_size, batch_size),
            name='act'))

        placeholders.append(tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=(step_size, batch_size),
            name='local_obs'))

        placeholders.append(tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=(step_size, batch_size),
            name='weights'))

        placeholders.append(tf.compat.v1.placeholder(
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
        num_l0_phys_state = self.params.num_l0_phys_state
        num_inter_phys_state = self.params.num_inter_phys_state
        num_particles = self.params.num_particle_pf
        batch_size = self.batch_size

        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()

        self.build_placeholders()

        grid_map, terminal_map, b0_other, b0_self, is_start, act_in, \
        obs_in, weight, act_label = self.placeholders

        # Type conversions.
        b0_other_istate = tf.tile(
            b0_other[:, None], multiples=[1, num_inter_phys_state, 1])
        b0_phys_istate = tf.tile(
            tf.range(num_inter_phys_state, dtype=tf.float32)[None, :, None],
            multiples=[batch_size, 1, 1])
        init_istate_space = tf.concat([b0_phys_istate, b0_other_istate], axis=-1)
        is_start = tf.reshape(
            tensor=is_start,
            shape=[self.batch_size] + [1] * (b0_self.get_shape().ndims - 1))

        # Initialize the series of predicted actions
        outputs = list()

        # pre-compute context, fixed through time
        with tf.compat.v1.variable_scope("planner"):
            q_val_self, _, _ = PlannerNet.value_iteration(
                grid_map=grid_map,
                terminal_map=terminal_map,
                params=self.params,
                conv_layer_names='vi_conv_self')
        with tf.compat.v1.variable_scope("filter"):
            obs_func_self = FilterNet.f_obs_func(
                grid_map=grid_map,
                terminal_map=terminal_map,
                params=self.params,
                conv_layer_names='obs_conv_self')
            obs_func_others = FilterNet.f_obs_func(
                grid_map=grid_map,
                terminal_map=terminal_map,
                params=self.params,
                conv_layer_names='obs_conv_other')

        # Create variable for hidden belief
        # (equivalent to the hidden state of an RNN)
        self.belief_self = tf.Variable(
            tf.zeros(shape=b0_self.get_shape().as_list(), dtype=tf.float32),
            trainable=False,
            name="hidden_belief_self")

        # Figure out current b. b = b0 if is_start else blast
        b_self = (b0_self * is_start) + (self.belief_self * (1 - is_start))

        prob_dist = tf.math.log(b_self)
        indices = tf.cast(tf.random.categorical(
            logits=prob_dist, num_samples=num_particles), dtype=tf.int32)
        helper = tf.range(
            0, batch_size * num_particles, delta=num_particles, dtype=tf.int32)
        indices += tf.expand_dims(helper, axis=1)
        particle_istates = tf.reshape(
            init_istate_space,
            shape=(batch_size * num_inter_phys_state, num_l0_phys_state + 1))
        init_particles = tf.gather(particle_istates, indices=indices, axis=0)

        self.particles = tf.Variable(
            tf.zeros(shape=init_particles.get_shape().as_list(),
                     dtype=tf.float32),
            trainable=False,
            name='intermediate_particles')

        # Figure out current particles. Similar to the process of belief.
        particles = (init_particles * is_start) + (self.particles * (1 - is_start))

        for step in range(self.step_size):
            # filter
            with tf.compat.v1.variable_scope("filter") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                particles = FilterNet.interactive_belief_update(
                    particles=particles,
                    action=act_in[step],
                    obs_func_self=obs_func_self,
                    obs_func_others=obs_func_others,
                    local_obs=obs_in[step],
                    grid_map=grid_map,
                    terminal_map=terminal_map,
                    batch_size=self.batch_size,
                    params=self.params)
                particle_phys_states = particles[:, :, 0]
                vec_phys_state = tf.one_hot(
                    tf.cast(particle_phys_states, dtype=tf.int32),
                    num_inter_phys_state)
                b_self = tf.reduce_sum(vec_phys_state, axis=1)
                b_self = tf.math.divide(  # normalization
                    b_self, tf.reduce_sum(b_self, axis=1, keepdims=True))

            # planner
            with tf.compat.v1.variable_scope("planner") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                action_pred = PlannerNet.policy(
                    q_val=q_val_self,
                    belief=b_self,
                    params=self.params)
                outputs.append(action_pred)

        # Create op that updates the beliefs.
        self.update_belief_self_op = self.belief_self.assign(b_self)
        self.update_particles_op = self.particles.assign(particles)

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
        learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate=init_learning_rate,
            global_step=decay_step,
            decay_steps=1,
            decay_rate=0.9,
            staircase=True,
            name="learning_rate")

        trainable_variables = tf.compat.v1.trainable_variables()

        optimizer = tf.compat.v1.train.RMSPropOptimizer(
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
            use_norm=tf.compat.v1.global_norm(grads))
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

        self.belief_self = None
        self.env_img = None
        self.terminal_img = None

        assert self.network.batch_size == 1 and self.network.step_size == 1

    def reset(self, env_img, terminal_map, belief_self):
        # TODO: do what?
        """

        :param env_img:
        :param terminal_map:
        :param belief_self:
        :return:
        """
        grid_n = self.network.params.grid_n
        grid_m = self.network.params.grid_m

        self.belief_self = belief_self.reshape(
            [1, self.params.num_inter_phys_state])
        self.env_img = env_img.reshape([1, grid_n, grid_m])
        self.terminal_img = terminal_map.reshape([1, grid_n, grid_m])

        self.sess.run(
            tf.compat.v1.assign(self.network.belief_self, self.belief_self))

    def eval(self, last_act, last_obs):
        """

        :param last_act:
        :param last_obs:
        :return:
        """
        is_start = np.array([0])
        last_act = np.reshape(last_act, [1, 1])
        last_obs = np.reshape(last_obs, [1, 1])

        # input data. do not need weight and label for prediction
        data = [self.env_img, self.terminal_img, self.belief_self,
                is_start, last_act, last_obs]
        feed_dict = {self.network.placeholders[i]: data[i]
                     for i in range(len(self.network.placeholders)-2)}

        # Evaluate IPOMDP-Net
        logits, _, _ = self.sess.run(
            [self.network.logits,
             self.network.update_belief_self_op,
             self.network.update_particles_op],
            feed_dict=feed_dict)
        act = np.argmax(logits.flatten())

        return act


class Level0PlannerNet:
    """
    Value function for the modeled agent when the nested levels bottom down to
    zero. This module is analogous to the Planner module of QMDP-net, not
    completely the same, though. This module is used for the modeling agent's
    prediction to the modeled agent's action. Hence, the value iteration and
    the policy determined by it are in second-person vision. The reward function
    and transition function of the modeled agent are not the ground true ones.
    Instead, they are learned by the modeling agent.
    """
    @staticmethod
    def f_reward(grid_map, terminal_map, params, names):
        """

        :param grid_map:
        :param terminal_map:
        :param params:
        :param names:
        :return:
        """
        num_joint_action = params.num_joint_action
        theta = tf.stack([grid_map, terminal_map], axis=3)
        r = nn_utils.conv_layers(
            input_data=theta,
            conv_params=np.array(
                [[3, 200, 'relu'],
                 [3, 150, 'lin'],
                 [1, num_joint_action, 'lin']]),
            names="l0_R_conv_" + names)
        return r

    @staticmethod
    def value_iteration(grid_map, terminal_map, params, names):
        """
        Builds neural network implementing value iteration. this is the first
        part of planner module. Fixed through time.
        :param grid_map: (batch x N x N) map images representing environments
        :param terminal_map: (batch x N x N) map images showing terminal cells
        :param params:
        :param names:
        :return:
        """
        num_l0_phys_state = params.num_l0_phys_state
        num_joint_action = params.num_joint_action
        num_recur = params.K
        # Obtain immediate reward based on reward function.
        r = Level0PlannerNet.f_reward(
            grid_map=grid_map,
            terminal_map=terminal_map,
            params=params,
            names=names)

        # Get transition model T'.
        # It represents the transition model in the filter, but the weights
        # are not shared.
        kernel = FilterNet.f_trans_func(
            num_joint_action, names='l0vi_' + names)
        # Initialize the value image.
        val = tf.zeros(grid_map.get_shape().as_list() + [1])
        q_val = None

        # repeat value iteration K times
        for i in range(num_recur):
            # Apply transition and sum
            q_val = tf.nn.conv2d(val, kernel, [1, 1, 1, 1], padding='SAME')
            # print("Q value shape:", q_val.shape)
            q_val = q_val + r
            val = tf.reduce_max(
                input_tensor=q_val,
                axis=[3],
                keepdims=True)
        q_val = nn_utils.scale_change_nd2lin(
            nd_tensor=q_val,
            fc_params=np.array([[num_l0_phys_state * num_joint_action, 'lin']]),
            names='img2linstate' + names)
        q_val = tf.reshape(
            q_val,
            shape=(q_val.get_shape().as_list()[0], num_l0_phys_state, num_joint_action))
        val = tf.reduce_max(
            input_tensor=q_val,
            axis=[2],
            keepdims=True)

        return q_val, val, r

    @staticmethod
    def f_pi(q_val, num_action, names):
        """

        :param q_val:
        :param num_action:
        :param names:
        :return:
        """
        action_pred = nn_utils.fc_layers(
            input_data=q_val,
            fc_params=np.array([[num_action, 'lin']]),
            names="pi_fc_" + names)
        return action_pred

    @staticmethod
    def policy(q_val, belief, params, names):
        """
        The second part of planner module
        :param q_val: input Q_K after value iteration
        :param belief: belief at current step
        :param params: params
        :param names:
        :return: action_pred, vector with num_action elements, each has the
        """
        num_action = params.num_action
        num_joint_action = params.num_joint_action
        # Weight q_val by the belief
        b_tiled = tf.tile(tf.expand_dims(belief, 2), [1, 1, num_joint_action])
        q = tf.multiply(q_val, b_tiled)
        # Sum over states
        q = tf.reduce_sum(
            input_tensor=q,
            axis=[1],
            keepdims=False)

        q = nn_utils.fc_layers(
            input_data=q,
            fc_params=np.array([[num_action, 'lin']]),
            names='joint2self' + names)

        # low-level policy, f_pi
        action_pred = Level0PlannerNet.f_pi(
            q_val=q,
            num_action=num_action,
            names=names)

        return action_pred


class FilterNet:
    """
    The network accounting for state estimation of the Interactive POMDP
    through time.
    """
    @staticmethod
    def f_obs_func(grid_map, terminal_map, params, conv_layer_names):
        """
        This method implements f_obs_func, outputs an observation model (Z).
        Fixed through time.
        :param grid_map:
        :param terminal_map:
        :param params:
        :param conv_layer_names:
        :return: tensor, obs_func (observation function of the modeling agent)
        """
        num_joint_action = params.num_joint_action
        num_obs = params.num_obs

        # CNN: theta -> Z
        theta = tf.stack([grid_map, terminal_map], axis=3)
        obs_func = nn_utils.conv_layers(
            input_data=theta,
            conv_params=np.array(
                [[3, 200, 'lin'],
                 [3, 100, 'lin'],
                 [1, num_joint_action * num_obs, 'sig']]),
            names=conv_layer_names)
        obs_func = tf.reshape(  # TODO: double check the dimensions of obs_func
            tensor=obs_func,
            shape=(obs_func.get_shape().as_list()[0],
                   obs_func.get_shape().as_list()[1],
                   obs_func.get_shape().as_list()[2],
                   num_joint_action,
                   num_obs))

        # Normalize over observations
        normalized_sum = tf.reduce_sum(
            input_tensor=obs_func,
            axis=[4],
            keepdims=True)
        # Avoid division by zero.
        obs_func = tf.math.divide(obs_func, normalized_sum + 1e-8)

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
    def f_obs(local_obs, names):
        """
        This method is for soft indexing observations.
        :param local_obs:
        :return:
        """
        w_obs = nn_utils.fc_layers(
            input_data=local_obs,
            fc_params=np.array([[5, 'tanh'], [5, 'smax']]),
            names="O_fc_" + names)
        return w_obs

    @staticmethod
    def f_trans_func(num_action, names):
        # Get transition kernel
        initializer = tf.compat.v1.truncated_normal_initializer(
            mean=1.0/9.0,
            stddev=1.0/90.0,
            dtype=tf.float32)
        kernel = tf.compat.v1.get_variable(
            shape=[3 * 3, num_action],
            initializer=initializer,
            dtype=tf.float32,
            name='w_trans_conv' + names)

        # Enforce proper probability distribution by softmax
        # (i.e. values must sum to one)
        kernel = tf.nn.softmax(logits=kernel, axis=0)
        kernel = tf.reshape(kernel, [3, 3, 1, num_action], name='T_w')

        return kernel

    @staticmethod
    def physical_belief_predict(belief, action_self, action_other, params, names):
        """
        The prediction phase in belief update over the physical states,
        equivalent to propagation in I-PF.
        :param belief: the modeling agent's belief over physical states at t-1
        :param action_self: the action of the modeling agent at t-1
        :param action_other: the action of the modeled agent at t-1
        :param params:
        :return: b_prime_a
        """
        num_action = params.num_action
        num_joint_action = params.num_joint_action
        grid_n = params.grid_n
        grid_m = params.grid_m

        # Get transition kernel (T_i)
        kernel = FilterNet.f_trans_func(
            num_joint_action, names='b_phys_self_update_' + names)

        # Using one fully-connected layer to map belief from num_state
        # to shape_grid_map
        belief_state2grid = nn_utils.fc_layers(
            input_data=belief,
            fc_params=np.array([[grid_n * grid_m, 'lin']]),
            names='fc_state2grid_' + str(names))
        b_img = tf.reshape(
            tensor=belief_state2grid,
            shape=[belief_state2grid.get_shape().as_list()[0], grid_n, grid_m, 1],
            name='b_img')

        # Apply convolution corresponding to the transition function in
        # an MDP (f_trans_func).
        b_prime = tf.nn.conv2d(b_img, kernel, [1, 1, 1, 1], padding='SAME')

        # index into the appropriate channel of b_prime
        a_joint = tf.concat([action_self, action_other], axis=-1)
        a_joint_space = tf.tile(
            tf.reshape(
                tf.range(num_action * num_action),
                shape=(num_action, num_action))[None],
            multiples=[action_self.get_shape().as_list()[0], 1, 1])
        a_joint_lin = tf.gather_nd(
            params=a_joint_space, indices=a_joint, batch_dims=1)
        a_joint_lin = tf.cast(a_joint_lin, dtype=tf.int32)
        vec_act = FilterNet.f_act(
            action=a_joint_lin,
            num_action=num_joint_action)  # size of joint action space
        vec_act = vec_act[:, None, None]

        # Soft indexing
        b_prime_a = tf.reduce_sum(
            input_tensor=tf.multiply(b_prime, vec_act),
            axis=[3],
            keepdims=False)

        return b_prime_a

    @staticmethod
    def others_model_update(particle_models, action, obs_func, params, names):
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
        grid_n = params.grid_n,
        grid_m = params.grid_m
        num_particles = params.num_particle_pf
        num_l0_phys_state = params.num_l0_phys_state
        num_joint_action = params.num_joint_action
        num_obs = params.num_obs

        # 1) Propagation of particle frames.
        # particle_frames = particle_models[:, :, num_phys_state:]
        # particle_frames = tf.tile(
        #     input=particle_frames,
        #     multiples=[1, num_obs, 1])

        # 2) Propagation of particle beliefs of the other agent.
        particle_others_beliefs = particle_models[:]
        kernel = FilterNet.f_trans_func(num_joint_action, names=names)
        beliefs = tf.unstack(  # returns a list of tensors unstacked along particle size
            value=particle_others_beliefs,
            num=num_particles,
            axis=1)
        for particle_i in range(len(beliefs)):
            # Using one fully-connected layer to map belief from num_state
            # to the shape of grid_map
            belief_state2grid = nn_utils.fc_layers(
                input_data=beliefs[particle_i],
                fc_params=np.array([[grid_n * grid_m, 'lin']]),
                names='fc_state2grid')
            b_img = tf.reshape(
                tensor=belief_state2grid,
                shape=[belief_state2grid.get_shape().as_list()[0], grid_n, grid_m, 1],
                name='b_img')

            # Apply convolution corresponding to the transition function in
            # an MDP (f_trans_func).
            b_prime = tf.nn.conv2d(b_img, kernel, [1, 1, 1, 1], padding='SAME')

            # Index into the channel of b_prime corresponding to previous action
            w_act = FilterNet.f_act(
                action=action,
                num_action=num_joint_action)
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
                shape=[b_next_imgs.get_shape().as_list()[0], num_obs * grid_n * grid_m],
                name='grid2state')
            beliefs[particle_i] = nn_utils.fc_layers(
                input_data=b_grid2state,
                fc_params=np.array([[num_l0_phys_state * num_obs], 'lin']),
                names='b_next')
            beliefs[particle_i] = tf.reshape(
                tensor=beliefs[particle_i],
                shape=[beliefs[particle_i].get_shape().as_list()[0], num_obs, num_l0_phys_state],
                name='b_per_o')
            beliefs[particle_i] = tf.math.divide(
                # normalize to make each particle belief sum to 1.
                x=beliefs[particle_i],
                y=tf.reduce_sum(
                    input_tensor=beliefs[particle_i],
                    axis=[2],
                    keepdims=True) + 1e-8)

        particle_others_beliefs = tf.stack(values=beliefs, axis=1)
        particle_others_beliefs = tf.reshape(
            tensor=particle_others_beliefs,
            shape=(particle_others_beliefs.get_shape().as_list()[0],
                   num_particles * num_obs,
                   num_l0_phys_state))

        # 3) Integrate frames and beliefs to rebuild models.
        particle_models = particle_others_beliefs[:]

        return particle_models

    @staticmethod
    def others_model_update_2(
            particle_models,
            action_self,
            particle_arr_act_other,
            obs_func, params,
            batch_size,
            names):
        """
        Particle propagation of the other agent's model.
        In this version, the N particle beliefs are regarded as one belief
        image, where the dimension of channels is set to N. The depth of the
        kernel is also duplicated to N. Then we apply depth-wise 2D convolution.
        :param particle_models: models factorized from particle interactive states
        :param action_self:
        :param particle_arr_act_other:
        :param obs_func:
        :param params:
        :param batch_size:
        :param names:
        :return: particle_models
        """
        grid_n = params.grid_n
        grid_m = params.grid_m
        obs_space = params.obs_space
        num_particles = params.num_particle_pf
        num_l0_phys_state = params.num_l0_phys_state
        num_action = params.num_action
        num_joint_action = params.num_joint_action
        num_obs = params.num_obs
        
        # 1) Propagation of particle frames.
        # particle_frames = particle_models[:, :, num_phys_state:]
        # particle_frames = tf.tile(
        #     input=particle_frames,
        #     multiples=[1, num_obs, 1])

        # 2) Propagation of particle beliefs of the other agent.
        particle_others_beliefs = particle_models[:]
        particle_others_beliefs = tf.reshape(
            tensor=particle_others_beliefs,
            shape=(particle_others_beliefs.get_shape().as_list()[0],
                   num_particles * num_l0_phys_state),  # (batch, N * |S|)
            name='fc_input_bj')
        beliefs_state2grid = nn_utils.fc_layers(
            input_data=particle_others_beliefs,
            fc_params=np.array(
                [[num_particles * grid_n * grid_m, 'lin']]),
            names='fc_state2grid')
        particle_b_imgs = tf.reshape(
            tensor=beliefs_state2grid,
            shape=(beliefs_state2grid.get_shape().as_list()[0],
                   grid_n, grid_m, num_particles),
            name='b_img_n_channel')  # N 1-channel imgs -> 1 N-channel img

        kernel = FilterNet.f_trans_func(
            num_joint_action, names='b_other_update_' + names)
        kernel = tf.tile(  # duplicate the kernel N times along 'in_channels'
            input=kernel,
            multiples=[1, 1, num_particles, 1],
            name='dw_kernel')

        b_imgs_prime = tf.nn.depthwise_conv2d(
            input=particle_b_imgs,
            filter=kernel,
            strides=[1, 1, 1, 1],
            padding='SAME')
        b_imgs_prime = tf.reshape(
            b_imgs_prime,
            shape=(b_imgs_prime.get_shape().as_list()[0],
                   grid_n, grid_m, num_particles, num_joint_action))

        vec_particle_arr_act = list()

        for a_other in particle_arr_act_other:
            a_joint = tf.concat([a_other, action_self], axis=-1)
            a_joint_space = tf.tile(
                tf.reshape(
                    tf.range(num_action * num_action),
                    shape=(num_action, num_action))[None],
                multiples=[action_self.get_shape().as_list()[0], 1, 1])
            a_joint_lin = tf.gather_nd(
                params=a_joint_space, indices=a_joint, batch_dims=1)
            a_joint_lin = tf.cast(a_joint_lin, dtype=tf.int32)
            vec_act = FilterNet.f_act(
                action=a_joint_lin,
                num_action=num_joint_action)
            vec_particle_arr_act.append(vec_act)
        vec_particle_arr_act = tf.stack(vec_particle_arr_act, axis=1)
        vec_particle_arr_act = tf.reshape(
            vec_particle_arr_act,
            shape=(vec_particle_arr_act.get_shape().as_list()[0],
                   1, 1, num_particles, num_joint_action))
        b_imgs_prime_a = tf.reduce_sum(
            input_tensor=tf.multiply(b_imgs_prime, vec_particle_arr_act),
            axis=[4],
            keepdims=False)

        # Correct belief with all j's available observation.
        # Without soft-indexing with j's observation, there are |O_j|
        # output belief images.
        vec_all_obs = list()
        i = 0
        for obs in obs_space:
            obs = tf.reshape(
                tf.multiply(tf.ones(batch_size), obs), shape=(batch_size, 1))
            vec_obs = FilterNet.f_obs(obs, names=str(names) + "_" + str(i))
            vec_all_obs.append(vec_obs)
            i += 1
        vec_all_obs = tf.stack(vec_all_obs, axis=1)[:, None, None, None]
        obs_func = tf.expand_dims(obs_func, axis=4)
        obs_func_o = tf.reduce_sum(
            input_tensor=tf.multiply(obs_func, vec_all_obs),
            axis=[5],
            keepdims=False)
        obs_func_o = tf.tile(
            obs_func_o[:, :, :, None], multiples=[1, 1, 1, num_particles, 1, 1])
        vec_particle_arr_act = tf.expand_dims(vec_particle_arr_act, axis=-1)
        obs_func_o_a = tf.reduce_sum(
            input_tensor=tf.multiply(obs_func_o, vec_particle_arr_act),
            axis=[4],
            keepdims=False)
        b_imgs_prime_a = tf.reshape(
            tf.tile(  # duplicate number beliefs from N to N * |O_j|
                input=b_imgs_prime_a,
                multiples=[1, 1, 1, num_obs]),
            shape=(batch_size, grid_n, grid_m, num_particles, num_obs))

        b_next_imgs = tf.multiply(b_imgs_prime_a, obs_func_o_a)  # correction

        particle_others_beliefs = nn_utils.scale_change_nd2lin(
            nd_tensor=b_next_imgs,
            fc_params=np.array(
                [[num_particles * num_obs * num_l0_phys_state, 'lin']]),
            names='b_other_grid2state_' + str(names))
        particle_others_beliefs = tf.reshape(
            tensor=particle_others_beliefs,
            shape=(batch_size,
                   num_particles * num_obs,
                   num_l0_phys_state),
            name='particle_bs')

        # Normalization: make each particle belief sum to 1.
        particle_others_beliefs = tf.math.divide(
            x=particle_others_beliefs,
            y=tf.reduce_sum(
                input_tensor=particle_others_beliefs,
                axis=[2],
                keepdims=True) + 1e-8)

        # 3) Integrate frames and beliefs to rebuild models.
        # If frames are not considered currently, the other's models are exactly
        # its beliefs.
        particle_models = particle_others_beliefs[:]

        return particle_models, obs_func_o_a

    @staticmethod
    def interactive_belief_update(
            particles,
            action,
            obs_func_self,
            obs_func_others,
            local_obs,
            grid_map,
            terminal_map,
            batch_size,
            params):
        """
        Interactive belief update
        :param particles:
        :param action: i's action at time step t-1
        :param obs_func_self: i's observation function, getting from grid_map
        :param obs_func_others: j's observation function, getting from grid_map
        :param local_obs: i's observation at time step t
        :param grid_map: the given environmental setting of the domain
        :param terminal_map: map showing where the door-cells are located
        :param params:
        :return: particles, phys_belief_self, ibelief
        """
        # Initialize variables to be used in interactive belief update.
        num_particle = params.num_particle_pf
        num_l0_phys_state = params.num_l0_phys_state
        num_inter_phys_state = params.num_inter_phys_state
        num_action = params.num_action
        num_joint_action = params.num_joint_action
        num_obs = params.num_obs
        action = action[:, None]

        particles = tf.reshape(
            particles, shape=(batch_size, num_particle, num_l0_phys_state + 1))
        particle_phys_states = particles[:, :, 0]
        particle_belief_other = particles[:, :, 1:]

        # Step 0: Preparation.
        # Predict the other agent's action by simulating its value function.
        particle_acts_other_pred = list()
        for particle_i in range(particles.get_shape().as_list()[1]):
            q_val_others_pred, _, _ = Level0PlannerNet.value_iteration(
                grid_map=grid_map,
                terminal_map=terminal_map,
                params=params,
                names=str(particle_i))
            a_other_pred = Level0PlannerNet.policy(
                q_val=q_val_others_pred,
                belief=particle_belief_other[:, particle_i],
                params=params,
                names=str(particle_i))
            a_other_pred = tf.reduce_sum(
                tf.multiply(a_other_pred, tf.range(num_action, dtype=tf.float32)),
                axis=[1], keepdims=False)
            a_other_pred = tf.cast(a_other_pred, dtype=tf.int32)
            a_other_pred = a_other_pred[:, None]
            particle_acts_other_pred.append(a_other_pred)
        assert len(particle_acts_other_pred) == num_particle

        # Step 1: Propagation.
        # 1) Propagate physical particle states.
        #    belief_t-1 -> belief_t -> particle_states_t
        # i. Belief update via convolution.
        # phys_belief_self = np.zeros(
        #     (particle_phys_states.get_shape().as_list()[0], num_inter_phys_state))
        # for batch_i in range(particle_phys_states.get_shape().as_list()[0]):
        #     unique_phys_states, _ = tf.unique(particle_phys_states[batch_i])
        #     print(unique_phys_states.get_shape().as_list())
        #     for i in range(unique_phys_states.get_shape().as_list()[0]):
        #         for particle in particle_phys_states[batch_i]:
        #             if tf.math.equal(particle, unique_phys_states[i]):
        #                 phys_belief_self[batch_i, unique_phys_states[i]] += 1
        #     phys_belief_self[batch_i] /= np.sum(phys_belief_self[batch_i]) + 1e-8
        # phys_belief_self = tf.convert_to_tensor(
        #     value=phys_belief_self,
        #     dtype=tf.float32)
        vec_phys_state = tf.one_hot(
            tf.cast(particle_phys_states, dtype=tf.int32),
            num_inter_phys_state)
        phys_belief_self = tf.reduce_sum(vec_phys_state, axis=1)
        phys_belief_self = tf.math.divide(  # normalization
            phys_belief_self,
            tf.reduce_sum(phys_belief_self, axis=1, keepdims=True))

        particle_phys_belief_self = list()
        names = 0
        for a_other in particle_acts_other_pred:
            phys_belief_img_self = FilterNet.physical_belief_predict(
                belief=phys_belief_self,
                action_self=action,
                action_other=a_other,
                params=params,
                names=str(names))
            phys_belief_self_prime = nn_utils.scale_change_nd2lin(
                nd_tensor=phys_belief_img_self,
                fc_params=np.array([[num_inter_phys_state, 'lin']]),
                names='belief_phys_self_img2state_' + str(names))
            normalized_sum_prime = tf.reduce_sum(
                phys_belief_self_prime, axis=[1], keepdims=True)
            phys_belief_self_prime = tf.math.divide(
                phys_belief_self_prime, normalized_sum_prime)
            names += 1
        # ii. Re-sampling particles according to updated belief.
            log_belief = tf.math.log(phys_belief_self_prime)
            indices = tf.cast(
                x=tf.random.categorical(
                    logits=log_belief,
                    num_samples=num_obs),  # (N * |O_j|), where for-loop over N
                dtype=tf.int32)
            helper = tf.range(
                start=0,
                limit=batch_size,
                delta=1,
                dtype=tf.int32)
            indices = indices + tf.expand_dims(helper, axis=1)
            particle_phys_states_prime = tf.gather(
                params=tf.reshape(
                    tf.tile(
                        tf.range(num_inter_phys_state)[None],
                        multiples=[batch_size, 1]),
                    shape=(batch_size * num_inter_phys_state,)),
                indices=indices,
                axis=0)

            particle_phys_belief_self.append(particle_phys_states_prime)
        particle_phys_belief_self = tf.reshape(
            tf.cast(
                tf.stack(particle_phys_belief_self, axis=1), dtype=tf.float32),
            shape=(batch_size, num_particle * num_obs, 1))

        # 2) Propagate particle models of others.
        particle_belief_other, weights_by_oj = FilterNet.others_model_update_2(
            particle_models=particle_belief_other,
            action_self=action,
            particle_arr_act_other=particle_acts_other_pred,
            obs_func=obs_func_others,
            params=params,
            batch_size=batch_size,
            names=str(names))
        # print("Shape of i's belief over j's physical beliefs:",
        #       particle_belief_other.get_shape().as_list())

        # 3) Integrate physical states and the other agent's models to be
        #    interactive states.
        particle_weights = list()
        particle_istates = tf.concat(
            values=[particle_phys_belief_self, particle_belief_other],
            axis=-1)

        # Step 2: Weighting.
        local_obs = tf.reshape(local_obs, shape=(batch_size, 1))
        v_obs_self = FilterNet.f_obs(local_obs, names='self')
        v_obs_self = v_obs_self[:, None, None, None]
        weights_by_oi_a = tf.reduce_sum(
            input_tensor=tf.multiply(obs_func_self, v_obs_self),
            axis=[4],
            keepdims=False)

        for a_other in particle_acts_other_pred:
            a_joint = tf.concat([action, a_other], axis=-1)
            a_joint_space = tf.tile(
                tf.reshape(
                    tf.range(num_action * num_action),
                    shape=(num_action, num_action))[None],
                multiples=[batch_size, 1, 1])
            a_joint_lin = tf.gather_nd(
                params=a_joint_space, indices=a_joint, batch_dims=1)
            a_joint_lin = tf.cast(a_joint_lin, dtype=tf.int32)
            v_act_self = FilterNet.f_act(
                action=tf.reshape(a_joint_lin, shape=(batch_size,)),
                num_action=num_joint_action)
            v_act_self = v_act_self[:, None, None]
            weights_by_oi = tf.reduce_sum(
                input_tensor=tf.multiply(weights_by_oi_a, v_act_self),
                axis=[3],
                keepdims=False)
            particle_weights.append(weights_by_oi)
        particle_weights = tf.stack(particle_weights, axis=-1)
        particle_weights = tf.tile(
            particle_weights, multiples=[1, 1, 1, num_obs])
        weights_by_oj = tf.reshape(
            weights_by_oj,
            shape=(batch_size,
                   weights_by_oj.get_shape().as_list()[1],
                   weights_by_oj.get_shape().as_list()[2],
                   num_particle * num_obs))
        weights_updated = tf.multiply(particle_weights, weights_by_oj)
        weights_updated = nn_utils.scale_change_nd2lin(
            nd_tensor=weights_updated,
            fc_params=np.array([[num_particle * num_obs, 'lin']]),
            names='w_img2scalar')
        # Normalize the updated particle weights.
        weights_updated = tf.math.divide(
            weights_updated, tf.reduce_sum(
                input_tensor=weights_updated,
                axis=[-1],
                keepdims=True) + 1e-8)
        # weights_updated = tf.reshape(
        #     weights_updated, shape=(batch_size, num_particle * num_obs))

        # Step 3: Down-sampling.
        log_weights = tf.math.log(weights_updated)
        indices = tf.cast(
            tf.random.categorical(
                logits=log_weights,
                num_samples=num_particle),
            dtype=tf.int32)
        helper = tf.range(
            start=0,
            limit=batch_size * num_particle,
            delta=num_particle,
            dtype=tf.int32)
        indices = indices + tf.expand_dims(helper, axis=1)
        # ----------------------------------------------
        # TODO: Figure out the exact shape of each interactive state.
        particle_istates = tf.reshape(
            tensor=particle_istates,
            shape=(batch_size * num_particle * num_obs, num_l0_phys_state + 1))
        # ----------------------------------------------
        particle_istates = tf.gather(
            params=particle_istates,
            indices=indices,
            axis=0)

        # --------------CHOOSE ONE OUT OF THE TWO OPTIONS-------------
        # 1) The ibelief contains both istates and their corresponding
        # probabilistic values in its structure, which is like
        # [[istate_0, prob_0], [istate 1, prob_1], ...[istate_N, prob_N]]
        # ibelief = [[] for _ in range(batch_size)]
        # particle_istates_np = particle_istates.numpy()
        # for batch_i in range(particle_istates.shape[0]):
        #     unique_istates = np.unique(
        #         ar=particle_istates_np[batch_i],
        #         axis=0)
        #     for i in range(unique_istates.shape[0]):
        #         ibelief[batch_i].append(np.append(unique_istates[i], 0))
        #         for particle in particle_istates[batch_i]:
        #             if particle.tolist() == unique_istates[i].tolist():
        #                 ibelief[batch_i][i][-1] += 1
        #     ibelief[batch_i] = np.array(ibelief[batch_i])
        #     ibelief[batch_i][:, -1] /= np.sum(ibelief[batch_i][:, -1]) + 1e-8
        # ibelief = tf.convert_to_tensor(
        #     value=ibelief,
        #     dtype=tf.float32)

        # 2) The ibelief contains only probabilistic values. The summation of
        # probabilities over the other agent's models is left to a fully-
        # connected layer in the planner module.
        # ibelief = ibelief[:, :, -1]
        # --------------------------------------------------------------

        return particle_istates


class PlannerNet:
    @staticmethod
    def f_reward(grid_map, terminal_map, params, conv_layer_names):
        """
        Rewards for the multi-agent tiger-grid navigation task depend on the
        goals, obstacles and the beliefs of modeled agents over physical states.
        :param grid_map: the input map for tiger-grid domain represented by 0,1.
        :param terminal_map: the map show cells where the doors are set.
        :param params: parameters from commands and the domain setting.
        :return: r -- reward function of the modeling agent (i in 2-agent setting)
        """
        theta = tf.stack([grid_map, terminal_map], axis=3)
        num_joint_action = params.num_joint_action
        r = nn_utils.conv_layers(
            input_data=theta,
            conv_params=np.array([
                [3, 200, 'relu'],
                [3, 150, 'lin'],
                [1, num_joint_action, 'lin']]),
            names="R_conv" + conv_layer_names)
        return r

    @staticmethod
    def value_iteration(grid_map, terminal_map, params, conv_layer_names):
        """
        Builds neural network implementing value iteration. this is the first
        part of planner module. Fixed through time.
        inputs: grid_map (batch x N x N) and goal(batch)
        returns: Q_K, and optionally: R, list of Q_i
        """
        num_inter_phys_state = params.num_inter_phys_state
        num_joint_action = params.num_joint_action
        num_recur = params.K

        r = PlannerNet.f_reward(
            grid_map=grid_map,
            terminal_map=terminal_map,
            params=params,
            conv_layer_names=conv_layer_names)

        # Get transition model T'.
        # It represents the transition model in the filter, but the weights
        # are not shared.
        kernel = FilterNet.f_trans_func(num_joint_action, names='_inter_vi')
        # Initialize the value image.
        val = tf.zeros(grid_map.get_shape().as_list() + [1])
        q_val = None

        # repeat value iteration K times
        for i in range(num_recur):
            # Apply transition and sum
            q_val = tf.nn.conv2d(val, kernel, [1, 1, 1, 1], padding='SAME')
            # print("Q value shape:", q_val.shape)
            q_val = q_val + r
            val = tf.reduce_max(
                input_tensor=q_val,
                axis=[3],
                keepdims=True)

        q_val = nn_utils.scale_change_nd2lin(
            nd_tensor=q_val,
            fc_params=np.array(
                [[20 * num_joint_action, 'lin'],
                 [num_inter_phys_state * num_joint_action, 'lin']]),
            names='img2linstate')
        q_val = tf.reshape(
            q_val, shape=(
                q_val.get_shape().as_list()[0],
                num_inter_phys_state,
                num_joint_action))
        val = tf.reduce_max(
            input_tensor=q_val,
            axis=[2],
            keepdims=True)

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
        :return: action_pred, vector with num_action elements
        """
        # batch_size = params.batch_size
        num_action = params.num_action
        num_joint_action = params.num_joint_action

        # if belief.ndim == 2:
        #     belief_phys = nn_utils.fc_layers(
        #         input_data=belief,
        #         fc_params=np.array([[params.num_inter_phys_state, 'lin']]),
        #         names='inter2phys_' + names)
        # elif belief.ndim == 3:
        #     belief_phys = np.zeros((batch_size, params.num_inter_phys_state))
        #     belief_np = belief.numpy()
        #     for batch_i in range(batch_size):
        #         for particle_i in range(belief.get_shape().as_list()[1]):
        #             phys_state = int(belief_np[batch_i, particle_i, 0])
        #             prob = belief_np[batch_i, particle_i, -1]
        #             belief_phys[batch_i, phys_state] += prob
        #     belief_phys = tf.convert_to_tensor(belief_phys)
        # else:
        #     assert False

        # Weight q_val by the belief
        b_tiled = tf.tile(
            tf.expand_dims(belief, 2), [1, 1, num_joint_action])
        q = tf.multiply(q_val, b_tiled)
        # Sum over states
        q = tf.reduce_sum(input_tensor=q,
                          axis=[1],
                          keepdims=False)
        q = nn_utils.fc_layers(
            input_data=q,
            fc_params=np.array([[num_action, 'lin']]),
            names='joint2self')

        # low-level policy, f_pi
        action_pred = PlannerNet.f_pi(q_val=q, num_action=num_action)

        return action_pred


def tf_unique_2d(x):
    x_shape = tf.shape(x)  # (3,2)
    x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
    x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

    x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
    x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
    cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
    # reshaping cond to match x1_2 & x2_2
    cond = tf.reshape(cond, [x_shape[0], x_shape[0]])
    cond_shape = tf.shape(cond)
    # convertin condition boolean to int
    cond_cast = tf.cast(cond, tf.int32)
    # replicating condition tensor into all 0's
    cond_zeros = tf.zeros(cond_shape, tf.int32)

    # CREATING RANGE TENSOR
    r = tf.range(x_shape[0])
    r = tf.add(tf.tile(r, [x_shape[0]]), 1)
    r = tf.reshape(r, [x_shape[0], x_shape[0]])

    # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default)
    # so when we take min it wont get selected & in end we will only take
    # values <max(indx).
    f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
    f2 = tf.ones(cond_shape, tf.int32)
    # if false make it max_index+1 else keep it 1
    cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)

    # multiply range with new int boolean mask
    r_cond_mul = tf.multiply(r, cond_cast2)
    r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
    r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
    r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

    # get actual values from unique indexes
    op = tf.gather(x, r_cond_mul4)

    return op