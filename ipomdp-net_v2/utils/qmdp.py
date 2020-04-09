import numpy as np, scipy.sparse
# import mdptoolbox
from pomdpy.solvers.solver import Solver
from pomdpy.solvers.alpha_vector import AlphaVector
from scipy.optimize import linprog
from itertools import product

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
            self.num_state = len(params['state_space'])
            self.num_action = len(params['action_space'])
            self.num_obs = len(params['obs_space'])
            self.init_discount = params["init_discount"]
            self.game_len = params['game_len']
            self.horizon = params['game_len']

    def one_step_value_iteration(self, belief, last_act, state):
        # immediate_reward = self.R.dot(self.R)
        immediate_reward = self.R[:, state]
        print("immediate_reward:", immediate_reward)
        print("Horizon:", self.horizon)
        if self.horizon == 0:
            self.horizon = self.game_len
            return immediate_reward

        _, belief = self.belief_update(b=belief,
                                       act=last_act,
                                       state_after_trans=state)
        print("belief:", belief)

        long_term_reward = None
        long_term_rewards = list()
        for act in range(self.num_action):
            if self.issparse:
                raise NotImplementedError
            else:
                print("action:", act)
                state_after_trans, _ = self.transition(state=state, action=act)
                print("state_after_trans:", state_after_trans)
                self.horizon -= 1

                value_next_horizon = \
                    self.one_step_value_iteration(
                        belief=belief,
                        last_act=act,
                        state=state_after_trans)
                long_term_reward = np.multiply(self.T[act, state_after_trans],
                                               value_next_horizon[act]).sum(axis=0)
                print("long term value for %d:" % act, long_term_reward)
                long_term_reward = belief.dot(long_term_reward)
                long_term_rewards.append(long_term_reward)
                print("----------------------")
        long_term_rewards = np.array(long_term_rewards)
        long_term_rewards = long_term_rewards.dot(self.Z)
        long_term_rewards *= self.init_discount

        value = (immediate_reward + long_term_reward).max()
        print("Value after one step VI:", value)

        return value

    # def compute_Q(self):
    #     self.Q = np.zeros([self.num_action, self.num_state], 'f')
    #     for act in range(self.num_action):
    #         if self.issparse:
    #             Ra_s = np.array(np.multiply(self.T[act], self.R[act]).
    #                             sum(axis=0)).squeeze()  # (s)
    #         else:
    #             Ra_s = np.multiply(self.T[act], self.R[act]).sum(axis=0)
    #
    #         self.Q[act] = Ra_s + self.init_discount * self.T[act].dot(self.V)
    #     print("Q:", self.Q)

    def get_action_wrt_value(self,
                             belief,
                             last_act,
                             state,
                             random_actions=False):
        """
        random_actions: select randomly from actions with near equal values.
        Lowest index by default
        :return: act
        """
        exp_val_sum = self.one_step_value_iteration(belief=belief,
                                                    last_act=last_act,
                                                    state=state)
        print("Expected sum of value:", exp_val_sum)

        if random_actions:
            equal_actions = np.isclose(exp_val_sum, exp_val_sum.max())
            act = np.random.choice(
                [i for i in range(self.num_action) if equal_actions[i]], 1)[0]

        act = exp_val_sum.argmax()

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
            bnext = np.multiply(b, self.Z[act][:, obs].transpose())
        else:
            bnext = np.multiply(b, self.Z[:, obs].transpose())
        bnext = bnext / bnext.sum()

        return bnext

    def belief_update(self, b, act, state_after_trans=None):
        """ Update belief with action. Sample an observation for the current state.
        If state is not specified observation is sampled according to the belief.
        :param b: belief
        :param act: action
        :params state_after_transition: state after executing the action
        Return: bprime (belief after taking action), observation, belief after taking action and receiving observation
        """
        bprime = self.propagate_act(b, act)

        # sample observation
        if not state_after_trans:
            obs = self.random_obs_over_belief(bprime, act)
        else:
            obs = self.random_obs(state_after_trans, act)

        # update beleif with observation
        bnext = self.propagate_obs(bprime, act, obs)

        return obs, bnext

    def processR(self, input_rwd_func):
        if isinstance(input_rwd_func, list):
            self.R = [input_rwd_func[a].copy() for a in range(self.num_action)]

        elif input_rwd_func.ndim == 3:
            self.R = input_rwd_func.copy()

        elif input_rwd_func.ndim == 2:
            self.R = np.zeros([self.num_action, self.num_state, self.num_state])
            for act in range(self.num_action):
                # self.R[act][non0[0], non0[1]] = input_rwd_func[act, non0[0]]
                for state in range(self.num_state):
                    self.R[act, state] = input_rwd_func[act, state]
        else:
            assert False

    def processT(self, input_trans_func):
        if isinstance(input_trans_func, list):
            self.T = [input_trans_func[a].copy() for a in
                      range(self.num_action)]
        elif input_trans_func.ndim == 3:
            self.T = input_trans_func.copy()
        elif input_trans_func.ndim == 2:
            # (state, action) -> state
            # self.T = np.zeros([self.num_action, self.num_state, self.num_state], 'f')
            self.T = [scipy.sparse.lil_matrix((self.num_state, self.num_state))
                      for a in range(self.num_action)]
            for act in range(self.num_action):
                self.T[act][
                    np.arange(self.num_state), input_trans_func[:, act].astype(
                        'i')] = 1.0
        else:
            assert False

    def processZ(self, input_obs_func):
        self.Zdim = 3
        if isinstance(input_obs_func, list):
            self.Z = [input_obs_func[a].copy() for a in range(self.num_action)]
        elif input_obs_func.ndim == 3:
            # normalize to avoid representation issues when loaded from file
            self.Z = input_obs_func / input_obs_func.sum(axis=2, keepdims=True)
        elif input_obs_func.ndim == 2:
            self.Z = scipy.sparse.csr_matrix(input_obs_func)
            self.Zdim = 2
        elif input_obs_func.ndim == 1:
            self.Zdim = 2
            self.Z = scipy.sparse.lil_matrix((self.num_state, self.num_obs))
            self.Z[np.arange(self.num_state), input_obs_func.astype('i')] = 1.0
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
