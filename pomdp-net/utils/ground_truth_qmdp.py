import numpy as np, scipy.sparse
import mdptoolbox

try:
    import ipdb as pdb
except Exception:
    import pdb


class QMDP:
    def __init__(self, params=None):
        self.T = None
        self.R = None
        self.Z = None
        self.Q = None
        self.V = None
        self.b0 = None

        self.issparse = False
        self.Zdim = 3

        if params:
            self.num_state = params['num_state']
            self.num_action = params['num_action']
            self.num_obs = params['num_obs']
            self.discount = params["discount"]

    def solve(self):
        self.compute_v(max_iter=10000)
        self.compute_q()

    def compute_v(self, max_iter=10000):
        try:
            # skip_check is not supported in the official release,
            # while it gives significant speed-up
            vi = mdptoolbox.mdp.ValueIteration(transitions=self.T,
                                               reward=self.R,
                                               discount=self.discount,
                                               max_iter=max_iter,
                                               skip_check=True)
        except:
            # try without skip_check
            vi = mdptoolbox.mdp.ValueIteration(self.T, self.R, self.discount,
                                               max_iter=max_iter)
        vi.run()
        assert vi.iter < max_iter
        self.V = vi.V

    def compute_q(self):
        self.Q = np.zeros([self.num_state, self.num_action], 'f')
        for act in range(self.num_action):
            if self.issparse:
                Ra_s = np.array(
                    self.T[act].multiply(self.R[act]).sum(1)).squeeze()  # (s)
            else:
                Ra_s = np.multiply(self.T[act], self.R[act]).sum(1)

            self.Q[:, act] = Ra_s + self.discount * self.T[act].dot(self.V)

    def qmdp_action(self, b, random_actions=False):
        """
        random_actions: select randomly from actions with near equal values.
        Lowest index by default
        """
        Qsum = b.dot(self.Q)

        if random_actions:
            equal_actions = np.isclose(Qsum, Qsum.max())
            act = np.random.choice(
                [i for i in range(self.num_action) if equal_actions[i]], 1)[0]
        else:
            act = Qsum.argmax()

        return act

    def transition(self, state, action):
        snext = self.sparse_choice(self.T[action][state], 1)[0]
        reward = self.R[action][state, snext]

        return snext, reward

    def sparse_choice(self, probs, count, **kwargs):
        if self.issparse:
            if probs.shape[1] == 1:
                vals, _, p = scipy.sparse.find(probs)
            else:
                assert probs.shape[0] == 1
                _, vals, p = scipy.sparse.find(probs)
        else:
            vals = len(probs)
            p = probs

        return np.random.choice(vals, count, p=p, **kwargs)

    def random_obs(self, state, act):
        """
        Sample an observation
        :param state: state after taking the action
        :param act: last aciton
        :return: observation
        """
        if self.Zdim == 3:
            pobs = self.Z[act][state]
        else:
            pobs = self.Z[state]
        # sample weighted with p_obs
        obs = self.sparse_choice(pobs, 1)[0]
        return obs

    def random_obs_over_belief(self, bprime, act):
        """
        Random observation given a belief
        :param bprime: belief after taking an action (updated)
        :param act: last action
        :return: observation
        """
        if self.issparse and not scipy.sparse.issparse(bprime):
            bprime = scipy.sparse.csr_matrix(bprime)

        if self.Zdim == 3:
            pobs_given_s = self.Z[act]
        else:
            pobs_given_s = self.Z

        pobs = bprime.dot(pobs_given_s)

        # normalize
        pobs = pobs / pobs.sum()
        # sample weighted with p_obs
        obs = self.sparse_choice(pobs, 1)[0]

        return obs

    def propagate_act(self, b, act):
        """
        Propagate belief when taking an action
        :param b:  belief
        :param act: action
        :return: updated belief
        """
        if self.issparse:
            if scipy.sparse.issparse(b):
                bsp = b
            else:
                bsp = scipy.sparse.csr_matrix(b)
            bprime = bsp.dot(self.T[act])
        else:
            bprime = np.zeros([self.num_state], dtype='f')
            for i in range(self.num_state):
                bprime += np.multiply(self.T[act][i], b[i])

        bprime = bprime / bprime.sum()

        return bprime

    def propagate_obs(self, b, act, obs):
        """
        Propagate belief with an observation
        :param b:  belief
        :param act: last action that produced the observation
        :param obs: observation
        :return: updated belief
        """
        if self.issparse and not scipy.sparse.issparse(b):
            b = scipy.sparse.csr_matrix(b)

        if self.Zdim == 3:
            bnext = b.multiply(
                self.Z[act][:, obs].transpose())  # elementwise multiply
        else:
            bnext = b.multiply(
                self.Z[:, obs].transpose())  # elementwise multiply
        bnext = bnext / bnext.sum()

        return bnext

    def belief_update(self, b, act, state_after_trans=None):
        """
        Update belief with action. Sample an observation for the current state.
        If state is not specified observation is sampled according to the belief
        :param state_after_trans:
        :param b: belief
        :param act: action
        :params state_after_transition: state after executing the action
        Return: bprime (belief after taking action), observation, belief after
        taking action and receiving observation
        """
        bprime = self.propagate_act(b, act)

        # sample observation
        if state_after_trans is None:
            obs = self.random_obs_over_belief(bprime, act)
        else:
            obs = self.random_obs(state_after_trans, act)

        # update beleif with observation
        bnext = self.propagate_obs(bprime, act, obs)

        return bprime, obs, bnext

    def set_terminals(self, goal_states, reward=None):
        """
        goal_states is a list of linear terminal state coordinates
        """
        goal_states = np.array(goal_states, 'i')

        # set reward, only where transition is possible
        if reward is not None:
            for x in range(self.num_action):
                non0ind = (self.T[x][:, goal_states]).nonzero()
                # reward for reaching goal from anywhere
                self.R[x][non0ind[0], goal_states[non0ind[1]]] = reward

        # make goal terminal: can't move away and all actions reward 0
        for x in range(self.num_action):
            self.T[x][goal_states, :] = 0.0
            self.T[x][goal_states, goal_states] = 1.0
            self.R[x][goal_states, goal_states] = 0

    def process_rwd_func(self, reward_func):
        if isinstance(reward_func, list):
            self.R = [reward_func[a].copy() for a in range(self.num_action)]

        elif reward_func.ndim == 3:
            self.R = reward_func.copy()

        elif reward_func.ndim == 2:
            assert self.T is not None
            reward_func = np.array(reward_func)
            self.R = [scipy.sparse.lil_matrix((self.num_state, self.num_state))
                      for _ in range(self.num_action)]
            for act in range(self.num_action):
                non0 = self.T[act].nonzero()
                self.R[act][non0[0], non0[1]] = reward_func[non0[0], act]
        else:
            assert False

    def process_trans_func(self, trans_func):
        if isinstance(trans_func, list):
            self.T = [trans_func[a].copy() for a in range(self.num_action)]
        elif trans_func.ndim == 3:
            self.T = trans_func.copy()
        elif trans_func.ndim == 2:
            # (state, action) -> state
            self.T = [scipy.sparse.lil_matrix((self.num_state, self.num_state))
                      for _ in range(self.num_action)]
            for act in range(self.num_action):
                self.T[act][
                    np.arange(self.num_state),
                    trans_func[:, act].astype('i')
                ] = 1.0
        else:
            assert False

    def process_obs_func(self, obs_func):
        self.Zdim = 3
        if isinstance(obs_func, list):
            self.Z = [obs_func[a].copy() for a in range(self.num_action)]
        elif obs_func.ndim == 3:
            # Normalize to avoid representation issues when loaded from file.
            self.Z = obs_func / obs_func.sum(axis=2, keepdims=True)
        elif obs_func.ndim == 2:
            self.Z = scipy.sparse.csr_matrix(obs_func)
            self.Zdim = 2
        elif obs_func.ndim == 1:
            self.Zdim = 2
            self.Z = scipy.sparse.lil_matrix((self.num_state, self.num_obs))
            self.Z[np.arange(self.num_state), obs_func.astype('i')] = 1.0
            self.Z = scipy.sparse.csr_matrix(self.Z)

        else:
            assert False

    def transfer_all_sparse(self):
        self.T = self.transfer_sparse(self.T)
        self.R = self.transfer_sparse(self.R)
        if self.Zdim == 3:
            self.Z = self.transfer_sparse(self.Z)
        self.issparse = True

    def transfer_sparse(self, mat):
        return [scipy.sparse.csr_matrix(mat[a]) for a in range(self.num_action)]
