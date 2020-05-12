import numpy as np
import scipy.sparse
import mdptoolbox


class IParticleFilter:
    def __init__(self, params):
        self.l0_trans_func = None
        self.l0_obs_func = None
        self.l0_reward_func = None
        self.inter_trans_func = None
        self.inter_obs_func = None
        self.inter_reward_func = None

        self.sub_q_val = None
        self.sub_val = None
        self.ob_q_val = None
        self.ob_val = None

        self.agent_loc_space = params.agent_loc_space
        self.door_loc_space = params.door_loc_space

        self.l0_phys_state_space = params.l0_phys_state_space
        self.inter_phys_state_space = params.inter_phys_state_space
        self.act_space = params.action_space
        self.joint_act_space = params.joint_action_space
        self.obs_space = params.obs_space

        self.num_l0_phys_state = params.num_l0_phys_state
        self.num_inter_phys_state = params.num_inter_phys_state
        self.num_action = params.num_action
        self.num_joint_action = params.num_joint_action
        self.num_obs = params.num_obs

        self.num_particle = params.num_particle_pf
        self.discount = params.discount

        self.agent_level = params.agent_level

        self.params = params

        self.count = 0

        self.issparse = False

    def solve_qmdp(self):
        self.solve_ob_agent()
        self.solve_sub_agent()

    def solve_sub_agent(self):
        self.compute_val(agent_level=self.agent_level, is_self=True)
        self.compute_q_val(agent_level=self.agent_level, is_self=True)

    def solve_ob_agent(self):
        self.compute_val(agent_level=self.agent_level - 1, is_self=False)
        self.compute_q_val(agent_level=self.agent_level - 1, is_self=False)

    def compute_val(self, agent_level, is_self, max_iter=1e4):
        if not agent_level:
            vi = mdptoolbox.mdp.ValueIteration(
                transitions=self.l0_trans_func,
                reward=self.l0_reward_func,
                discount=self.discount,
                max_iter=max_iter,
                skip_check=True)
        else:
            vi = mdptoolbox.mdp.ValueIteration(
                transitions=self.inter_trans_func,
                reward=self.inter_reward_func,
                discount=self.discount,
                max_iter=max_iter,
                skip_check=True)

        vi.run()
        assert vi.iter < max_iter

        if is_self:
            self.sub_val = vi.V
        else:
            self.ob_val = vi.V

    def compute_q_val(self, agent_level, is_self):
        self.sub_q_val = np.zeros(
            (self.num_inter_phys_state, self.num_joint_action))
        if not agent_level:
            self.ob_q_val = np.zeros(
                (self.num_l0_phys_state, self.num_joint_action))
            for act_joint_lin in range(self.num_joint_action):
                if self.issparse:
                    reward = np.array(
                        self.l0_trans_func[act_joint_lin].multiply(
                        self.l0_reward_func[act_joint_lin]).sum(1)).squeeze()
                else:
                    reward = self.l0_trans_func[act_joint_lin].multiply(
                        self.l0_reward_func[act_joint_lin]).sum(1)
                self.ob_q_val[:, act_joint_lin] = \
                    reward + self.discount *\
                    self.l0_trans_func[act_joint_lin].dot(self.ob_val)
        else:
            if is_self:
                for act_joint_lin in range(self.num_joint_action):
                    if self.issparse:
                        reward = np.array(
                            self.inter_trans_func[act_joint_lin].multiply(
                            self.inter_reward_func[act_joint_lin]).sum(1)).squeeze()
                    else:
                        reward = self.inter_trans_func[act_joint_lin].multiply(
                            self.inter_reward_func[act_joint_lin]).sum(1)
                    self.sub_q_val[:, act_joint_lin] = \
                        reward + self.discount *\
                        self.inter_trans_func[act_joint_lin].dot(self.sub_val)
            else:
                self.ob_q_val = np.zeros(
                    (self.num_inter_phys_state, self.num_joint_action))
                for act_joint_lin in range(self.num_joint_action):
                    if self.issparse:
                        reward = np.array(
                            self.inter_trans_func[act_joint_lin].multiply(
                            self.inter_reward_func[act_joint_lin]).sum(1)).squeeze()
                    else:
                        reward = self.inter_trans_func[act_joint_lin].multiply(
                        self.inter_reward_func[act_joint_lin]).sum(1)
                    self.ob_q_val[:, act_joint_lin] = \
                        reward + self.discount *\
                        self.inter_trans_func[act_joint_lin].dot(self.ob_val)

    def qmdp_policy(self, belief, agent_level, is_self):
        belief = np.array(belief)
        if not agent_level:
            q_val_sum = np.dot(belief, self.ob_q_val)
            q_val_sum_self = list()
            for a_self in self.act_space:
                a_joint = [[a_self, a_other] for a_other in self.act_space]
                a_joint_lin = [joint_to_lin(
                    a, (self.act_space, self.act_space)) for a in a_joint]
                q_val_sum_self.append(np.mean(q_val_sum[a_joint_lin]))
            q_val_sum_self = np.array(q_val_sum_self)
            act_self = q_val_sum_self.argmax()
        else:
            if is_self:
                q_val = self.sub_q_val
            else:
                q_val = self.ob_q_val

            phys_belief = np.zeros(self.num_inter_phys_state)
            for i in range(belief.shape[0]):
                istate, prob = belief[i]
                s = int(istate[0])
                phys_belief[s] += prob

            # phys_belief /= phys_belief.sum()
            q_val_sum = np.dot(phys_belief, q_val)
            q_val_sum_self = list()
            for a_self in self.act_space:
                a_joint = [[a_self, a_other] for a_other in self.act_space]
                a_joint_lin = [joint_to_lin(
                    a, (self.act_space, self.act_space)) for a in a_joint]
                q_val_sum_self.append(np.mean(q_val_sum[a_joint_lin]))
            q_val_sum_self = np.array(q_val_sum_self)
            act_self = q_val_sum_self.argmax()

        return act_self

    @staticmethod
    def particle_sampling(dist, particle_size):
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

    @staticmethod
    def approx2exact(particles):
        particles_o2f = list()
        for i in range(particles.shape[0]):
            particles_o2f.append(np.append(particles[i][0], particles[i][1]))
        unique_particles = np.unique(particles_o2f, axis=0)
        num_uniq_particle = len(unique_particles)
        belief = np.zeros(shape=(num_uniq_particle, 2), dtype='object')

        for i in range(unique_particles.shape[0]):
            particle = unique_particles[i].tolist()
            count = 0
            for pt in particles_o2f:
                pt = pt.tolist()
                if pt == particle:
                    count += 1
            belief[i] = np.array([particle, count])

        # assert np.sum(belief[:, 1]) == len(particles), \
        #     "total count of particles is %d" % np.sum(belief[:, 1])
        try:
            belief[:, 1] /= np.sum(belief[:, 1])  # normalization
        except ZeroDivisionError:
            belief[:, 1] /= np.sum(belief[:, 1]) + 1e-10
            print(particles)
            print("-" * 10)
            print(particles_o2f)
            print("-" * 10)
            print(unique_particles)

        return belief

    def l0_particle_filtering(self, l0belief, act, obs):
        pr_act_other = 1 / self.num_action
        b_next = np.zeros(l0belief.shape)
        for state_next in self.l0_phys_state_space:
            if np.ndim(state_next):
                state_next = joint_to_lin(
                    state_next, (self.agent_loc_space, self.door_loc_space))
            b_prime = 0
            for state in self.l0_phys_state_space:
                if np.ndim(state):
                    state = joint_to_lin(
                        state, (self.agent_loc_space, self.door_loc_space))
                pr_trans = 0
                for act_other in self.act_space:
                    joint_act = [act, act_other]
                    joint_act_lin = joint_to_lin(
                        joint_act, (self.act_space, self.act_space))
                    pr_trans += np.multiply(
                        self.l0_trans_func[joint_act_lin][state, state_next],
                        pr_act_other)
                b_prime += np.multiply(pr_trans, l0belief[state])

            pr_obs = 0
            for act_other in self.act_space:
                joint_act = [act, act_other]
                joint_act_lin = joint_to_lin(
                    joint_act, (self.act_space, self.act_space))
                pr_obs += np.multiply(
                    self.l0_obs_func[joint_act_lin][state_next, obs],
                    pr_act_other)

            # print("state_next:", state_next)
            # print("act:", act)
            # print("obs:", obs)
            # print("b_prime:", b_prime)
            # print("pr_obs:", pr_obs)
            b_next[state_next] = np.multiply(pr_obs, b_prime)

        # print("b_next", b_next)
        b_next = np.divide(b_next, b_next.sum())

        return b_next

    def i_particle_filtering(self, ibelief,
                             act, obs,
                             agent_level=1,
                             first_step=False):
        assert agent_level > 0

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

            if agent_level > 1:
                phys_belief_other = others_model[:self.num_inter_phys_state]
            else:
                phys_belief_other = others_model[:self.num_l0_phys_state]
            if first_step:
                act_other = self.params.init_action
            else:
                act_other = self.qmdp_policy(
                phys_belief_other, agent_level=agent_level - 1, is_self=False)

            joint_act = [act, act_other]
            joint_act_lin = joint_to_lin(
                joint_act, (self.act_space, self.act_space))
            phys_state_next_lin = np.random.choice(
                np.arange(self.num_inter_phys_state),
                p=self.inter_trans_func[joint_act_lin][phys_state, :].toarray()[0])
            assert not np.ndim(phys_state_next_lin)

            for obs_other in self.obs_space:
                if agent_level == 1:
                    b_other = others_model.copy()
                    # print("b_other:")
                    # print(b_other)
                    b_other_next = self.l0_particle_filtering(
                        l0belief=b_other,
                        act=act_other,
                        obs=obs_other)
                    # print("b_other_next:")
                    # print(b_other_next)
                    others_model_next = b_other_next.copy()
                    istate_next = [phys_state_next_lin, others_model_next]
                    # print("istate_next:")
                    # print(istate_next)
                    phys_state_next = lin_to_joint(
                        phys_state_next_lin,
                        (self.agent_loc_space,
                         self.agent_loc_space,
                         self.door_loc_space))
                    l0_phys_state_next = np.array(
                        [phys_state_next[1], phys_state_next[2]])
                    l0_phys_state_next_lin = joint_to_lin(
                        l0_phys_state_next,
                        (self.agent_loc_space,
                         self.door_loc_space))
                    particle_weight = self.l0_obs_func[
                        joint_act_lin][l0_phys_state_next_lin, obs_other]
                    particle_weight *= self.inter_obs_func[
                        joint_act_lin][phys_state_next_lin, obs]
                    # print("particle_weight:")
                    # print(particle_weight)
                else:
                    b_other = others_model.copy()
                    b_other_next = self.i_particle_filtering(
                        ibelief=b_other,
                        act=act_other,
                        obs=obs_other,
                        agent_level=agent_level-1)
                    istate_next = [phys_state_next_lin, b_other_next]
                    particle_weight = self.inter_obs_func[
                        joint_act_lin, phys_state_next_lin, obs_other]
                    particle_weight *= self.inter_obs_func[
                        joint_act_lin, phys_state_next_lin, obs]

                b_tmp[count] = np.array([istate_next, particle_weight])
                count += 1

        # Normalize particle weights to make them sum to 1.
        # print(b_tmp)
        b_tmp[:, 1] = np.divide(b_tmp[:, 1], np.sum(b_tmp[:, 1]))

        # Selection (down-sampling)
        b_next_approx = np.random.choice(
            b_tmp[:, 0],
            size=self.num_particle,
            p=np.array(b_tmp[:, 1], dtype='f'))
        b_next = self.approx2exact(b_next_approx)

        return b_next

    def process_trans_func(self, l0_trans_func, inter_trans_func):
        self.l0_trans_func = [l0_trans_func[a].copy()
                              for a in range(self.num_joint_action)]
        self.inter_trans_func = [inter_trans_func[a].copy()
                                 for a in range(self.num_joint_action)]

    def process_reward_func(self, l0_reward_func, inter_reward_func):
        self.l0_reward_func = [l0_reward_func[a].copy()
                               for a in range(self.num_joint_action)]
        self.inter_reward_func = [inter_reward_func[a].copy()
                                  for a in range(self.num_joint_action)]

    def process_obs_func(self, l0_obs_func, inter_obs_func):
        self.l0_obs_func = [l0_obs_func[a].copy()
                            for a in range(self.num_joint_action)]
        self.inter_obs_func = [inter_obs_func[a].copy()
                               for a in range(self.num_joint_action)]

    def transfer_all_sparse(self):
        self.l0_trans_func = self.transfer_sparse(self.l0_trans_func)
        self.l0_reward_func = self.transfer_sparse(self.l0_reward_func)
        self.l0_obs_func = self.transfer_sparse(self.l0_obs_func)

        self.inter_trans_func = self.transfer_sparse(self.inter_trans_func)
        self.inter_reward_func = self.transfer_sparse(self.inter_reward_func)
        self.inter_obs_func = self.transfer_sparse(self.inter_obs_func)

        self.issparse = True

    def transfer_sparse(self, matrix):
        return [scipy.sparse.csr_matrix(matrix[a])
                for a in range(self.num_joint_action)]


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

    return int(lin_idx)


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
    num_joint = joint_space.shape[0]
    space_size = [space.shape[0] for space in joint_space][::-1]

    space_indices = list()
    numerator = lin_idx
    i = 0
    while i < len(space_size) - 1:
        denominator = space_size[i]
        space_indices.append(np.mod(numerator, denominator))
        numerator = np.floor_divide(numerator, denominator)
        i += 1
    space_indices.append(numerator)
    space_indices = space_indices[::-1]

    joint_coord = np.zeros(num_joint, dtype='i')
    for i in range(num_joint):
        joint_coord[i] = joint_space[i][space_indices[i]]

    return joint_coord
