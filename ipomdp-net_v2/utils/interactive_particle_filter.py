import numpy as np


class IParticleFilter:
    def __init__(self, trans_func, reward_func, obs_func, params):
        self.trans_func = trans_func
        self.obs_func = obs_func
        self.reward_func = reward_func

        self.agent_loc_space = params.agent_loc_space
        self.door_loc_space = params.door_loc_space

        self.phys_state_space = params.phys_state_space
        self.act_space = params.action_space
        self.joint_act_space = params.joint_action_space
        self.obs_space = params.obs_space

        self.num_phys_state = params.num_phys_state
        self.num_action = params.num_action
        self.num_joint_action = params.num_joint_action
        self.num_obs = params.num_obs

        self.num_particle = params.num_particle_pf
        self.discount = params.discount

        self.params = params

        self.count = 0

    def particle_sampling(self, dist, particle_size):
        """
        This method samples particles from the distribution given by the belief.
        :param dist: the probabilistic distribution given by the input belief
        :param particle_size:
        :return: particles -- samples from the probabilistic distribution
        """
        states = np.array([x[0] for x in dist])
        probs = np.array([x[1] for x in dist])
        indices = np.arange(dist.shape[0])

        particles = states[np.random.choice(
            indices, size=particle_size, p=probs)]

        return particles

    def approx2exact(self, particles):
        num_uniq_particle = len(set(particles))
        belief = np.zeros(shape=(num_uniq_particle, 2), dtype='object')

        i = 0  # the index for assigning istates and corresponding probabilities
        for particle in particles:
            if particle in belief[:, 0]:
                belief[np.where(belief[:, 0] == particle), 1] += 1
            else:
                belief[i] = np.array([particle, 1])
                i += 1

        assert np.sum(belief[:, 1]) == len(particles)
        belief[:, 1] /= len(particles)  # normalization

        return belief

    def l0_particle_filtering(self, l0belief, act, obs):
        pr_act_other = 1 / self.num_action
        b_next = np.zeros(l0belief.shape)
        for state_next in self.phys_state_space:
            if np.ndim(state_next):
                state_next = joint_to_lin(
                    state_next, (self.agent_loc_space, self.door_loc_space))
            b_prime = 0
            for state in self.phys_state_space:
                if np.ndim(state):
                    state = joint_to_lin(
                        state, (self.agent_loc_space, self.door_loc_space))
                pr_trans = 0
                for act_other in self.act_space:
                    joint_act = [act, act_other]
                    joint_act_lin = joint_to_lin(
                        joint_act, (self.act_space, self.act_space))
                    pr_trans += np.multiply(
                        self.trans_func[joint_act_lin, state, state_next],
                        pr_act_other)
                b_prime += np.multiply(pr_trans, l0belief[state])

            pr_obs = 0
            for act_other in self.act_space:
                joint_act = [act, act_other]
                joint_act_lin = joint_to_lin(
                    joint_act, (self.act_space, self.act_space))
                pr_obs += np.multiply(
                    self.obs_func[joint_act_lin, state_next, obs], pr_act_other)

            b_next[state_next] = np.multiply(pr_obs, b_prime)

        return b_next

    def i_particle_filtering(self, ibelief, act, obs, level=1):
        assert level > 0

        b_tmp = np.zeros(
            (self.num_particle * self.num_obs, 2), dtype='object')
        istates = self.particle_sampling(
            dist=ibelief, particle_size=self.num_particle)
        count = 0

        # Importance sampling
        for istate in istates:
            phys_state = int(istate[0])
            others_model = istate[1:]
            if np.ndim(phys_state):
                phys_state = joint_to_lin(
                    phys_state, (self.agent_loc_space, self.door_loc_space))
            if level > 1:
                act_other = self.qmdp_policy(ibelief)[0][0]
            else:
                b = others_model[:self.num_phys_state]
                act_other = np.random.choice(self.l0_policy(b))

            joint_act = [act, act_other]
            joint_act_lin = joint_to_lin(
                joint_act, (self.act_space, self.act_space))
            print(phys_state)
            phys_state_next = self.phys_state_space[np.random.choice(
                np.arange(self.num_phys_state),
                p=self.trans_func[joint_act_lin, phys_state, :])]
            assert not np.ndim(phys_state_next)

            for obs_other in range(self.num_obs):
                if level == 1:
                    b_other, frame_other = others_model
                    b_other_next = self.l0_particle_filtering(
                        l0belief=b_other,
                        act=act_other,
                        obs=obs_other)
                    others_model_next = [b_other_next, frame_other]
                    istate_next = [phys_state_next, others_model_next]
                else:
                    b_other, frame_other = others_model
                    b_other_next = self.i_particle_filtering(
                        ibelief=b_other,
                        act=act_other,
                        obs=obs_other,
                        level=level-1)
                    others_model_next = [b_other_next, frame_other]
                    istate_next = [phys_state_next, others_model_next]

                particle_weight = self.obs_func[
                    act_other, phys_state_next, obs_other]
                particle_weight *= self.obs_func[act, phys_state_next, obs]
                b_tmp[count] = np.array([istate_next, particle_weight])
                count += 1

        # Normalize particle weights to make them sum to 1.
        b_tmp[:, 1] /= np.sum(b_tmp[:, 1])

        # Selection (down-sampling)
        b_next_approx = np.random.choice(
            b_tmp[:, 0], size=self.num_particle, p=b_tmp[:, 1])
        b_next = self.approx2exact(b_next_approx)

        return b_next

    def l0_val_func(self, belief, num_iter=100):
        print("num_iter_left:", num_iter)
        q_vals = np.zeros(self.act_space.shape)
        if num_iter > 0:
            for a_self in self.act_space:
                st_rwd = 0
                for s in self.phys_state_space:
                    s = joint_to_lin(
                        s, (self.agent_loc_space, self.door_loc_space))
                    for a_other in self.act_space:
                        a_joint = [a_self, a_other]
                        a_joint_lin = joint_to_lin(
                            a_joint, (self.act_space, self.act_space))
                        st_rwd += np.multiply(
                            belief[s], self.reward_func[a_joint_lin, s])
                lt_rwd = 0
                for o in self.obs_space:
                    pr_obs = 0
                    for s in self.phys_state_space:
                        s = joint_to_lin(
                            s, (self.agent_loc_space, self.door_loc_space))
                        for a_other in self.act_space:
                            a_joint = [a_self, a_other]
                            a_joint_lin = joint_to_lin(
                                a_joint, (self.act_space, self.act_space))
                            pr_obs += np.multiply(
                                belief[s], self.obs_func[a_joint_lin, s, o])
                    b_next = self.l0_particle_filtering(
                        l0belief=belief, act=a_self, obs=o)
                    self.count += 1
                    print("VI executed count:", self.count)
                    u_next, _ = self.l0_val_func(b_next, num_iter=num_iter - 1)
                    lt_rwd += pr_obs * u_next
                q_val = st_rwd + lt_rwd * self.discount
                q_vals[a_self] = q_val
            val = np.max(q_vals)
        else:
            for a_self in self.act_space:
                st_rwd = 0
                for s in self.phys_state_space:
                    s = joint_to_lin(
                        s, (self.agent_loc_space, self.door_loc_space))
                    for a_other in self.act_space:
                        a_joint = [a_self, a_other]
                        a_joint_lin = joint_to_lin(
                            a_joint, (self.act_space, self.act_space))
                        # print("belief over state %d: %f" % (s, belief[s]))
                        # print("reward of taking %d: %f" %
                        #       (a_self, self.reward_func[a_joint_lin, s]))
                        st_rwd += belief[s] * self.reward_func[a_joint_lin, s]
                q_val = st_rwd
                q_vals[a_self] = q_val
            val = np.max(q_vals)

        # print("value:", val)

        return val, q_vals

    def l0_policy(self, b):
        val, q_vals = self.l0_val_func(b, 2)
        opt_act = np.argwhere(q_vals == val)
        opt_act = np.array([x[0] for x in opt_act if np.ndim(x)])
        return opt_act

    def ma_mdp_vi(self, num_iter=100):
        q_vals_joint = np.zeros(
            (self.num_phys_state, self.num_action, self.num_action))

        if num_iter > 0:
            for s in self.phys_state_space:
                s = joint_to_lin(
                    s, (self.agent_loc_space, self.door_loc_space))
                for a_self in self.act_space:
                    for a_other in self.act_space:
                        a_joint = joint_to_lin(
                            [a_self, a_other], (self.act_space, self.act_space))
                        st_rwd = self.reward_func[a_joint, s]
                        lt_rwd = 0
                        for s_next in self.phys_state_space:
                            s_next = joint_to_lin(
                                s_next, (self.agent_loc_space,
                                         self.door_loc_space))
                            pr_trans = self.trans_func[a_joint, s, s_next]
                            lt_rwd += np.multiply(
                                pr_trans, self.ma_mdp_vi(num_iter - 1))
                        q_val = st_rwd + lt_rwd
                        q_vals_joint[s, a_self, a_other] = q_val
        else:
            for s in self.phys_state_space:
                s = joint_to_lin(
                    s, (self.agent_loc_space, self.door_loc_space))
                for a_self in self.act_space:
                    for a_other in self.act_space:
                        a_joint = joint_to_lin(
                            [a_self, a_other], (self.act_space, self.act_space))
                        st_rwd = self.reward_func[a_joint, s]
                        q_val = st_rwd
                        q_vals_joint[s, a_self, a_other] = q_val

        q_vals_self = np.sum(q_vals_joint, axis=2)
        vals = np.max(q_vals_self, axis=1)

        return vals

    def qmdp_policy(self, b_inter):
        num_act_self = self.params.num_action
        num_act_joint = self.params.num_joint_action
        num_phys_state = self.params.num_phys_state

        vals = self.ma_mdp_vi()

        b_phys = np.zeros(self.phys_state_space.shape)
        s_inter = b_inter[:, 0]
        s_phys = list()
        for s in s_inter:
            s_phys.append(s[0])
        phys_uniq = set(s_phys)
        for s in phys_uniq:
            for i in range(s_inter.shape[0]):
                if s == s_inter[i][0]:
                    b_phys[s] += b_inter[i, 1]

        q_vals_self = np.zeros((num_act_self, num_phys_state))
        q_vals_joint = np.zeros((num_act_joint, num_phys_state))
        for a_self in self.act_space:
            for a_other in self.act_space:
                a_joint = [a_self, a_other]
                a_joint = joint_to_lin(
                    a_joint, (self.act_space, self.act_space))
                st_rwd = self.reward_func[a_joint, :]
                lt_rwd = np.zeros(self.phys_state_space.shape)
                for s_next in self.phys_state_space:
                    s_next = joint_to_lin(
                        s_next, (self.agent_loc_space, self.door_loc_space))
                    lt_rwd += np.multiply(
                        vals[s_next], self.trans_func[a_joint, :, s_next])
                q_vals_joint[a_joint, :] = st_rwd + lt_rwd
                q_vals_joint[a_joint, :] = np.multiply(
                    q_vals_joint[a_joint, :], b_phys)

            idx = joint_to_lin(
                [a_self, self.act_space[0]], (self.act_space, self.act_space))
            q_vals_self[a_self] = np.mean(
                q_vals_joint[idx:idx+num_act_self, :], axis=0)

        q_vals_a = np.sum(q_vals_self, axis=1)

        return np.argwhere(q_vals_a == np.max(q_vals_a))[0][0]  # argmax_a(q(a))


def compute_lin_index(arr_indices, arr_space_sizes):
    """

    :param arr_indices:
    :param arr_space_sizes:
    :return:
    """
    lin_idx = None

    if len(arr_indices) > 2:
        lin_idx = np.multiply(
            arr_indices[0], np.prod(arr_space_sizes[1:])) \
                  + compute_lin_index(arr_indices[1:], arr_space_sizes[1:])
    elif len(arr_indices) == 2:
        lin_idx = np.multiply(
            arr_indices[0], arr_space_sizes[1]) + arr_indices[1]
    elif len(arr_indices) == 1:
        lin_idx = arr_indices[0]
    else:
        assert (len(arr_indices) >= 1), "No element exists in the input array."

    return lin_idx


def joint_to_lin(joint_coord, joint_space):
    joint_coord = np.array(joint_coord)
    joint_space = np.array(joint_space)

    assert (joint_coord.shape[0] == joint_space.shape[0]), \
        "The joint coordinate and the joint space must be in the same dimension"

    idx_coord_in_space = [np.argwhere(
        joint_coord[i] == joint_space[i])[0][0]
                          for i in range(joint_coord.shape[0])]
    space_sizes = [len(space) for space in joint_space]

    lin_idx = 0
    lin_idx += compute_lin_index(
        arr_indices=idx_coord_in_space,
        arr_space_sizes=space_sizes)

    return lin_idx


def lin_to_joint(lin_idx, joint_space):
    joint_space = np.array(joint_space)
    space_sizes = [len(space) for space in joint_space]

    i = 1
    numerator = lin_idx
    denominator = space_sizes[i]
    remainder = np.mod(numerator, denominator)

    joint_coord = list()
    while i + 1 < joint_space.shape[0]:
        joint_coord.append(remainder)
        numerator = np.floor_divide(numerator, denominator)
        i += 1
        denominator = space_sizes[i]
    joint_coord.append(np.mod(numerator, denominator))
    joint_coord.append(np.floor_divide(numerator, denominator))
    joint_coord = joint_coord[::-1]

    for i in range(len(joint_coord)):
        joint_coord[i] = joint_space[i][joint_coord[i]]

    return joint_coord
