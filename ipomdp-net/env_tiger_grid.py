import argparse
import random
import os

import numpy as np
import scipy.sparse
import _pickle as pickle
import tables

from utils.dotdict import dotdict
from utils.ground_truth_qmdp import QMDP

try:
    import ipdb as pdb
except Exception:
    import pdb


FREESTATE = 0.0
OBSTACLE = 1.0


class TigerGridBase(object):
    def __init__(self, params):
        """
        Initialize domain simulator
        :param params: domain descriptor dotdict
        """
        self.params = params

        self.N = params.grid_n
        self.M = params.grid_m
        self.grid_shape = [self.N, self.M]

        self.init_states = params.init_states
        self.goal_states = params.goal_states
        self.negative_states = params.negative_states

        self.states_wall_above_left = params.states_wall_above_left
        self.states_wall_above_right = params.states_wall_above_right
        self.states_wall_below_left = params.states_wall_below_left
        self.states_wall_below_right = params.states_wall_below_right
        self.states_wall_above_only = params.states_wall_above_only
        self.states_wall_below_only = params.states_wall_below_only
        self.states_wall_left_only = params.states_wall_left_only
        self.states_wall_right_only = params.states_wall_right_only

        self.observe_directions = params.observe_directions

        self.num_action = params.num_action
        self.num_obs = params.num_obs
        self.obs_len = len(self.observe_directions) + 1  # 4 directions + goal
        self.num_state = (self.N * self.M - 1) * 4

        self.trans_func = None
        self.reward_func = None
        self.obs_func = None

        self.grid = None
        self.graph = None

    def simulate_policy(self, policy, grid, b0, init_state, goal_states,
                        first_action=None):
        params = self.params
        max_traj_len = params.traj_limit

        if not first_action:
            first_action = params.init_action

        self.grid = grid

        self.gen_pomdp()
        qmdp = self.get_qmdp(goal_states)

        state = init_state
        reward_sum = 0.0  # accumulated reward

        failed = False
        step_i = 0

        # initialize policy
        env_img = grid[None]
        goal_img = self.process_goals(goal_states)
        b0_img = self.process_beliefs(b0)
        policy.reset(env_img, goal_img, b0_img)

        act = first_action
        obs = None

        while True:
            # finish if state is terminal, i.e. we reached a goal state
            if all([np.isclose(qmdp.T[x][state, state], 1.0)
                    for x in range(params.num_action)]):
                assert state in goal_states
                break

            # stop if trajectory limit reached
            if step_i >= max_traj_len:
                failed = True
                break

            # choose next action
            # if it is the initial step, stay will always be selected;
            # otherwise the action is selected based on the learned policy
            if step_i:
                if obs >= 16:
                    act = first_action
                else:
                    obs_without_goal = self.obs_lin_to_bin(obs)
                    if list(obs_without_goal) == [0, 1, 0, 1] or \
                            list(obs_without_goal) == [1, 0, 1, 0]:
                        obs_with_goal = np.array([0, 0, 0, 0, 1])
                    else:
                        obs_with_goal = np.append(obs_without_goal, 0)
                    act = policy.eval(act, obs_with_goal)

            # simulate action
            state, r = qmdp.transition(state, act)
            obs = qmdp.random_obs(state, act)

            # Update expected/accumulated reward.
            reward_sum += r

            step_i += 1

        traj_len = step_i

        return (not failed), traj_len, reward_sum

    def generate_trajectories(self, db, num_traj):
        params = self.params
        max_traj_len = params.traj_limit

        for traj_i in range(num_traj):
            # generate a QMDP object, initial belief, initial state and
            # goal state, also generates a random grid for the first iteration
            qmdp, b0, init_state, goal_states = self.random_instance(
                generate_grid=(traj_i == 0))

            qmdp.solve()

            state = init_state
            goal_state = None
            b = b0.copy()  # linear belief
            reward_sum = 0.0  # accumulated reward

            # Trajectory of beliefs: Includes start and goal.
            beliefs = list()
            # Trajectory of states: Includes start and goal.
            states = list()
            # Trajectory of actions:
            # First action is always stay. Excludes action after reaching goal.
            actions = list()
            # Trajectory of rewards
            rewards = list()
            # Trajectory of observations:
            # Includes observation at start but excludes observation after
            # reaching goal.
            observations = list()

            failed = False
            step_i = 0

            while True:
                beliefs.append(b)
                states.append(state)

                # finish if state is terminal, i.e. we reached a goal state
                if all([np.isclose(qmdp.T[x][state, state], 1.0)
                        for x in range(params.num_action)]):
                    assert state in goal_states
                    goal_state = state
                    break

                # stop if trajectory limit reached
                if step_i >= max_traj_len:
                    failed = True
                    break

                # choose action
                if step_i == 0:
                    # dummy first action
                    act = params.init_action
                else:
                    act = qmdp.qmdp_action(b)

                #  Simulate action
                state, r = qmdp.transition(state, act)
                bprime, obs, b = qmdp.belief_update(
                    b=b,
                    act=act,
                    state_after_trans=state)

                actions.append(act)
                rewards.append(r)
                observations.append(obs)

                reward_sum += r

                # # count collisions
                # if np.isclose(r, params.R_obst):
                #     collisions += 1

                step_i += 1

            # add to database
            if not failed:
                db.root.valids.append([len(db.root.samples)])
            else:
                goal_state = random.choice(self.goal_states)

            traj_len = step_i

            # step: state (linear), action, observation (linear)
            step = np.stack([states[:traj_len],
                             actions[:traj_len],
                             observations[:traj_len]], axis=1)

            # print("--State Trajectory--")
            # for i in range(traj_len):
            #     print(states[i])
            # print("-" * 30)
            # print("--Belief Trajectory--")
            # for y in range(traj_len):
            #     print("Step:", y)
            #     print(beliefs[y])
            # print("-" * 30)
            # print("--Action Trajectory--")
            # for j in range(traj_len):
            #     print(actions[j])
            # print("-" * 30)
            # print("--Reward Trajectory--")
            # for k in range(traj_len):
            #     print(rewards[k])
            # print("-" * 30)
            # print("--Observation Trajectory--")
            # for x in range(traj_len):
            #     print(observations[x])

            # sample: env_id, goal_state, step_id, traj_len, failed
            # length includes both start and goal (so one step path is length 2)
            sample = np.array(
                [len(db.root.envs), goal_state, len(db.root.steps),
                 traj_len, failed], 'i')

            db.root.samples.append(sample[None])
            db.root.bs.append(np.array(beliefs[:1]))
            db.root.expRs.append([reward_sum])
            db.root.steps.append(step)

        # add environment only after adding all trajectories
        db.root.envs.append(self.grid[None])

    def random_instance(self, generate_grid=True):
        """
        Generate a random problem instance for a grid.
        Picks a random initial belief, initial state and goal states.
        :param generate_grid: generate a new grid and POMDP model if True,
        otherwise use self.grid
        :return:
        """
        while True:
            if generate_grid:
                self.grid = self.gen_grid(self.params.grid_n,
                                          self.params.grid_m)
                # Generate POMDP model: self.T, self.Z, self.R
                self.gen_pomdp()

            while True:
                # sample initial belief, start, goal
                # b0: initial belief uniformly distributed over 24 states
                # b0_2: initial belief distributed over 2 fixed states
                b0, b0_2, init_state, init_state_2, goal_states = \
                    self.gen_start_and_goal()
                # print("Initial belief:")
                # print(b0)
                # print("Initial state:", init_state)
                # print("Goal state:", goal_states)
                if b0 is None:
                    assert generate_grid
                    break  # regenerate obstacles

                # reject if start == goal
                if init_state in goal_states:
                    continue

                # Create qmdp: makes soft copies from self.T{R,Z}simple
                # it will also convert to csr sparse, and set qmdp.issparse=True
                qmdp = self.get_qmdp(goal_states)

                return qmdp, b0, init_state, goal_states

    def gen_pomdp(self):
        # construct all POMDP model(R, T, Z)
        self.obs_func = self.build_obs_func()

        # for act in range(len(self.obs_func)):
        #     np.savetxt('./data/obs_func.txt', self.obs_func[act])

        self.trans_func, trans_func_most_likely, self.reward_func = \
            self.build_trans_reward_funcs()

        # f = open(file='./data/trans_reward.txt', mode='w')
        # f.write("TRANSITION FUNCTION\n\n")
        # for act in range(len(self.trans_func)):
        #     f.write("State transition for action %d\n" % act)
        #     for i in range(np.array(self.trans_func[act].todense()).shape[0]):
        #         for j in range(
        #                 np.array(self.trans_func[act].todense()).shape[1]):
        #             f.write(
        #                 str(np.array(self.trans_func[act].todense())[i, j]))
        #             f.write(" ")
        #         f.write("\n")
        #     f.write("\n\n")
        #
        #
        # f.write("-" * 30)
        # f.write("\n\n")
        # f.write("REWARD FUNCTION\n\n")
        # for act in range(len(self.reward_func)):
        #     f.write("Reward for action %d\n" % act)
        #     for i in range(
        #             np.array(self.reward_func[act].todense()).shape[0]):
        #         for j in range(
        #                 np.array(self.reward_func[act].todense()).shape[1]):
        #             f.write(
        #                 str(np.array(self.reward_func[act].todense())[i, j]))
        #             f.write(" ")
        #         f.write("\n")
        #     f.write("\n\n")
        #
        # f.close()

    def build_obs_func(self):

        obs_func = np.zeros(
            [self.num_action, self.num_state, self.num_obs], dtype='f')

        for i in range(self.N):
            for j in range(self.M):
                grid_coord = np.array([i, j])
                grid = self.state_bin_to_lin(grid_coord)
                state = None

                if grid == 7:
                    continue
                elif grid > 7:
                    for k in range(4):
                        state = (grid-1)*4 + k

                        # Construct observations
                        obs = np.zeros(
                            [self.obs_len])  # 1 or 0 in four directions
                        if state in self.states_wall_above_left:
                            obs[0] = obs[3] = 1
                        elif state in self.states_wall_above_right:
                            obs[0] = obs[1] = 1
                        elif state in self.states_wall_below_left:
                            obs[2] = obs[3] = 1
                        elif state in self.states_wall_below_right:
                            obs[1] = obs[2] = 1
                        elif state in self.states_wall_above_only:
                            obs[0] = 1
                        elif state in self.states_wall_right_only:
                            obs[1] = 1
                        elif state in self.states_wall_below_only:
                            obs[2] = 1
                        elif state in self.states_wall_left_only:
                            obs[3] = 1
                        elif state in self.goal_states:
                            obs[-1] = 1
                            obs_func[:, state, -1] = 1.0
                            continue
                        else:
                            assert 36 > state >= 0
                            print("Exception at 337:", state)

                        true_obs = np.array(obs)

                        for obs_i in range(self.num_obs-1):
                            prob = 1.0
                            cur_obs = self.obs_lin_to_bin(obs_i)
                            for direction in range(self.obs_len-1):
                                if true_obs[direction] == 1:
                                    if true_obs[direction] == cur_obs[direction]:
                                        prob *= 0.9
                                    else:
                                        prob *= 0.1
                                else:
                                    if true_obs[direction] == cur_obs[direction]:
                                        prob *= 0.95
                                    else:
                                        prob *= 0.05
                            obs_func[:, state, obs_i] = prob

                        # Sanity check
                        assert np.isclose(1.0, obs_func[0, state, :].sum())

                else:
                    for k in range(4):
                        state = grid*4 + k

                        # Construct observations
                        obs = np.zeros(
                            [self.obs_len])  # 1 or 0 in four directions
                        if state in self.states_wall_above_left:
                            obs[0] = obs[3] = 1
                        elif state in self.states_wall_above_right:
                            obs[0] = obs[1] = 1
                        elif state in self.states_wall_below_left:
                            obs[2] = obs[3] = 1
                        elif state in self.states_wall_below_right:
                            obs[1] = obs[2] = 1
                        elif state in self.states_wall_above_only:
                            obs[0] = 1
                        elif state in self.states_wall_right_only:
                            obs[1] = 1
                        elif state in self.states_wall_below_only:
                            obs[2] = 1
                        elif state in self.states_wall_left_only:
                            obs[3] = 1
                        elif state in self.goal_states:
                            obs[-1] = 1
                            obs_func[:, state, -1] = 1.0
                            continue
                        else:
                            assert 36 > state >= 0
                            print("Exception at 385:", state)

                        true_obs = np.array(obs)
                        # print("True observations:")
                        # print(true_obs)

                        for obs_i in range(self.num_obs - 1):
                            prob = 1.0
                            cur_obs = self.obs_lin_to_bin(obs_i)
                            # print(obs_i, cur_obs)
                            for direction in range(self.obs_len - 1):
                                if true_obs[direction] == 1:
                                    if true_obs[direction] == cur_obs[direction]:
                                        prob *= 0.9
                                    else:
                                        prob *= 0.1
                                else:
                                    if true_obs[direction] == cur_obs[direction]:
                                        prob *= 0.95
                                    else:
                                        prob *= 0.05
                            #     print("direction:", direction, prob)
                            # print("prob:", prob)
                            obs_func[:, state, obs_i] = prob

                        # Sanity check
                        assert np.isclose(1.0, obs_func[0, state, :].sum())

        return obs_func

    def build_trans_reward_funcs(self):
        """
        Build transition (trans_func) and reward (reward_func) model for a grid.
        :return: transition model trans_func, reward model reward_func, and
        maximum likely transitions trans_func_most_likely
        """

        # Probability of transition with a0 from s1 to s2
        trans_func = [
            scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
            for _ in range(self.num_action)
        ]

        # Scalar reward of transition with a0 from s1 to s2 (only depend on s2)
        reward_func = [
            scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
            for _ in range(self.num_action)
        ]
        # goal will be defined as a terminal state, all actions remain in goal
        # with 0 reward

        # maximum likely versions
        # Tml[s, a] -> next state
        trans_func_most_likely = np.zeros(
            [self.num_state, self.num_action], 'i')
        # Rml[s, a, s'] -> reward in s'
        # reward_func_most_likely = np.zeros(
        #     [self.num_state, self.num_action], 'f')

        for i in range(self.N):
            for j in range(self.M):
                grid_coord = np.array([i, j])
                grid = self.state_bin_to_lin(grid_coord)

                if grid == 7:
                    continue
                elif grid > 7:
                    for k in range(4):
                        state = (grid-1)*4 + k
                        if state in self.goal_states:
                            trans_func_most_likely[state, :] = \
                                random.choice([24, 28])
                            trans_func[:, state, 24] = 0.5
                            trans_func[:, state, 28] = 0.5
                        for act in range(self.num_action):
                            trans_func, trans_func_most_likely = \
                                self.trans_func_val_assign_wrt_act(
                                    state=state,
                                    act=act,
                                    trans_func=trans_func,
                                    trans_func_most_likely=trans_func_most_likely
                                )
                else:
                    for k in range(4):
                        state = grid*4 + k
                        for act in range(self.num_action):
                            trans_func, trans_func_most_likely = \
                                self.trans_func_val_assign_wrt_act(
                                    state=state,
                                    act=act,
                                    trans_func=trans_func,
                                    trans_func_most_likely=trans_func_most_likely
                                )

        # Fill in values for reward function, indexed by [a, s, s_]
        for i in range(self.num_action):
            for j in range(self.num_state):
                for k in range(self.num_state):
                    if k in self.goal_states:
                        reward_func[i][j, k] = 1.0
                    elif k in self.negative_states:
                        reward_func[i][j, k] = -1.0
                    else:
                        continue

        return trans_func, trans_func_most_likely, reward_func

    def trans_func_val_assign_wrt_act(self,
                                      state,
                                      act,
                                      trans_func,
                                      trans_func_most_likely):
        if act == 0:
            trans_func_most_likely[state, act] = state
            trans_func[act][state, state] = 1.0

        elif act == 1:  # Move forward
            if state in self.states_wall_above_left:
                trans_func_most_likely[state, act] = state
                trans_func[act][state, state] = 0.9
                if state == 0:
                    trans_func[act][state, state + 5] = 0.05
                    trans_func[act][state, state + 20] = 0.025
                    trans_func[act][state, state + 22] = 0.025
                elif state == 17:
                    trans_func[act][state, state + 17] = 0.05
                    trans_func[act][state, state - 4] = 0.025
                    trans_func[act][state, state - 2] = 0.025
                elif state == 23:
                    trans_func[act][state, state - 23] = 0.05
                    trans_func[act][state, state + 4] = 0.025
                    trans_func[act][state, state + 2] = 0.025
                elif state == 26:
                    trans_func[act][state, state - 3] = 0.05
                    trans_func[act][state, state - 20] = 0.025
                    trans_func[act][state, state - 22] = 0.025
                elif state == 31:
                    trans_func[act][state, state - 19] = 0.05
                    trans_func[act][state, state + 4] = 0.025
                    trans_func[act][state, state + 2] = 0.025
                else:
                    trans_func[act][state, state - 3] = 0.05
                    trans_func[act][state, state - 16] = 0.025
                    trans_func[act][state, state - 18] = 0.025
            elif state in self.states_wall_above_right:
                trans_func_most_likely[state, act] = state
                trans_func[act][state, state] = 0.9
                if state == 3:
                    trans_func[act][state, state + 19] = 0.05
                    trans_func[act][state, state + 4] = 0.025
                    trans_func[act][state, state + 2] = 0.025
                elif state == 16:
                    trans_func[act][state, state - 1] = 0.05
                    trans_func[act][state, state + 16] = 0.025
                    trans_func[act][state, state + 18] = 0.025
                elif state == 22:
                    trans_func[act][state, state + 3] = 0.05
                    trans_func[act][state, state - 20] = 0.025
                    trans_func[act][state, state - 22] = 0.025
                elif state == 25:
                    trans_func[act][state, state - 21] = 0.05
                    trans_func[act][state, state - 4] = 0.025
                    trans_func[act][state, state - 2] = 0.025
                elif state == 30:
                    trans_func[act][state, state + 3] = 0.05
                    trans_func[act][state, state - 16] = 0.025
                    trans_func[act][state, state - 18] = 0.025
                else:
                    trans_func[act][state, state - 17] = 0.05
                    trans_func[act][state, state - 4] = 0.025
                    trans_func[act][state, state - 2] = 0.025
            elif state in self.states_wall_below_left:
                trans_func[act][state, state] = 0.15
                if state == 1:
                    trans_func_most_likely[state, act] = state + 4
                    trans_func[act][state, state + 4] = 0.8
                    trans_func[act][state, state + 21] = 0.05
                elif state == 18:
                    trans_func_most_likely[state, act] = state + 16
                    trans_func[act][state, state + 16] = 0.8
                    trans_func[act][state, state - 3] = 0.05
                elif state == 20:
                    trans_func_most_likely[state, act] = state - 20
                    trans_func[act][state, state - 20] = 0.8
                    trans_func[act][state, state + 3] = 0.05
                elif state == 27:
                    trans_func_most_likely[state, act] = state - 4
                    trans_func[act][state, state - 4] = 0.8
                    trans_func[act][state, state - 23] = 0.05
                elif state == 28:
                    trans_func_most_likely[state, act] = state - 16
                    trans_func[act][state, state - 16] = 0.8
                    trans_func[act][state, state + 5] = 0.05
                else:
                    trans_func_most_likely[state, act] = state - 4
                    trans_func[act][state, state - 4] = 0.8
                    trans_func[act][state, state - 19] = 0.05
            elif state in self.states_wall_below_right:
                trans_func[act][state, state] = 0.15
                if state == 2:
                    trans_func_most_likely[state][act] = state + 20
                    trans_func[act][state, state + 20] = 0.8
                    trans_func[act][state, state + 3] = 0.05
                elif state == 19:
                    trans_func_most_likely[state][act] = state - 4
                    trans_func[act][state, state - 4] = 0.8
                    trans_func[act][state, state + 15] = 0.05
                elif state == 21:
                    trans_func_most_likely[state][act] = state + 4
                    trans_func[act][state, state + 4] = 0.8
                    trans_func[act][state, state - 21] = 0.05
                elif state == 24:
                    trans_func_most_likely[state][act] = state - 20
                    trans_func[act][state, state - 20] = 0.8
                    trans_func[act][state, state - 1] = 0.05
                elif state == 29:
                    trans_func_most_likely[state][act] = state + 4
                    trans_func[act][state, state + 4] = 0.8
                    trans_func[act][state, state - 17] = 0.05
                else:
                    trans_func_most_likely[state][act] = state + 16
                    trans_func[act][state, state - 16] = 0.8
                    trans_func[act][state, state - 1] = 0.05
            elif state in self.states_wall_above_only:
                trans_func_most_likely[state, act] = state
                trans_func[act][state, state] = 0.85
                if state == 4:
                    trans_func[act][state, state - 1] = 0.05
                    trans_func[act][state, state + 5] = 0.05
                    trans_func[act][state, state + 20] = 0.025
                    trans_func[act][state, state + 22] = 0.025
                else:
                    trans_func[act][state, state - 1] = 0.05
                    trans_func[act][state, state + 5] = 0.05
                    trans_func[act][state, state + 16] = 0.025
                    trans_func[act][state, state + 18] = 0.025
            elif state in self.states_wall_below_only:
                trans_func[act][state, state] = 0.1
                if state == 6:
                    trans_func_most_likely[state, act] = state + 20
                    trans_func[act][state, state + 20] = 0.8
                    trans_func[act][state, state + 3] = 0.05
                    trans_func[act][state, state - 3] = 0.05
                else:
                    trans_func_most_likely[state, act] = state + 16
                    trans_func[act][state, state + 16] = 0.8
                    trans_func[act][state, state + 3] = 0.05
                    trans_func[act][state, state - 3] = 0.05
            elif state in self.states_wall_left_only:
                trans_func_most_likely[state][act] = state + 4
                trans_func[act][state, state + 4] = 0.8
                trans_func[act][state, state - 4] = 0.025
                trans_func[act][state, state - 2] = 0.025
                trans_func[act][state, state] = 0.1
                if state == 5:
                    trans_func[act][state, state + 21] = 0.05
                else:
                    trans_func[act][state, state + 17] = 0.05
            elif state in self.states_wall_right_only:
                trans_func_most_likely[state, act] = state - 4
                trans_func[act][state, state - 4] = 0.8
                trans_func[act][state, state + 4] = 0.025
                trans_func[act][state, state + 2] = 0.025
                trans_func[act][state, state] = 0.1
                if state == 7:
                    trans_func[act][state, state + 19] = 0.05
                else:
                    trans_func[act][state, state + 15] = 0.05
            elif state in self.goal_states:
                trans_func_most_likely[state, act] = 24
                trans_func[act][state, 24] = 0.5
                trans_func[act][state, 28] = 0.5
            else:
                assert 36 > state >= 0
                print("Exception at 653:", state)

        elif act == 3:  # Turn around
            trans_func[act][state, state] = 0.1
            if state % 4 < 2:
                trans_func_most_likely[state, act] = state + 2
                trans_func[act][state, state + 2] = 0.6
                if state % 4:
                    trans_func[act][state, state + 1] = 0.15
                    trans_func[act][state, state - 1] = 0.15
                else:
                    trans_func[act][state, state + 1] = 0.15
                    trans_func[act][state, state + 3] = 0.15
            else:
                trans_func_most_likely[state, act] = state - 2
                trans_func[act][state, state - 2] = 0.6
                if state % 2:
                    trans_func[act][state, state - 3] = 0.15
                    trans_func[act][state, state - 1] = 0.15
                else:
                    trans_func[act][state, state + 1] = 0.15
                    trans_func[act][state, state - 1] = 0.15

        elif act == 2:  # Turn right
            trans_func[act][state, state] = 0.1
            if state % 4 < 3:
                trans_func_most_likely[state, act] = state + 1
                trans_func[act][state, state + 1] = 0.7
                if state % 4 < 2:
                    trans_func[act][state, state + 2] = 0.1
                    if state % 2:
                        trans_func[act][state, state - 1] = 0.1
                    else:
                        trans_func[act][state, state + 3] = 0.1
                else:
                    trans_func[act][state, state - 2] = 0.1
                    trans_func[act][state, state - 1] = 0.1
            else:
                trans_func_most_likely[state, act] = state - 3
                trans_func[act][state, state - 3] = 0.7
                trans_func[act][state, state - 2] = 0.1
                trans_func[act][state, state - 1] = 0.1

        else:  # Turn left
            trans_func[act][state, state] = 0.1
            if not state % 4:
                trans_func_most_likely[state, act] = state + 3
                trans_func[act][state, state + 3] = 0.7
                trans_func[act][state, state + 2] = 0.1
                trans_func[act][state, state + 1] = 0.1

            else:
                trans_func_most_likely[state, act] = state - 1
                trans_func[act][state, state - 1] = 0.7
                if state % 4 < 2:
                    trans_func[act][state, state + 2] = 0.1
                    trans_func[act][state, state + 1] = 0.1
                else:
                    trans_func[act][state, state - 2] = 0.1
                    if state % 2:
                        trans_func[act][state, state - 3] = 0.1
                    else:
                        trans_func[act][state, state + 1] = 0.1

        return trans_func, trans_func_most_likely

    def gen_start_and_goal(self):
        # x = np.random.randint(5)
        idx_b0 = np.random.choice(a=self.init_states,
                                  size=2,  # there are two initial states
                                  replace=False)  # avoid replicated state
        b0 = np.zeros([self.num_state])
        b0[idx_b0] = 1.0 / len(idx_b0)

        idx_b0_2 = np.array([28, 32])
        b0_2 = np.zeros([self.num_state])
        b0_2[idx_b0_2] = 1.0/len(idx_b0_2)

        # Sample initial state from initial belief.
        # In tiger-grid domain, the initial state is uniformly sampled
        # from 24 and 28.
        init_state = np.random.choice(self.num_state, p=b0)
        init_state_2 = np.random.choice(self.num_state, p=b0_2)

        # Sample goal state
        # In fixed tiger-grid domain, the goal state is fixed, which can be
        # any one within [8, 9, 10, 11].
        # Practically, the goal grid should be uniformly distributed in all
        # available grids except "-1 grids"
        goal_states = self.goal_states

        return b0, b0_2, init_state, init_state_2, goal_states

    def get_qmdp(self, goal_states):
        qmdp = QMDP(self.params)

        qmdp.process_trans_func(self.trans_func)  # this will make a hard copy
        qmdp.process_rwd_func(self.reward_func)
        qmdp.process_obs_func(self.obs_func)

        qmdp.set_terminals(goal_states=goal_states,
                           reward=self.params.reward_goal)

        qmdp.transfer_all_sparse()
        return qmdp

    @staticmethod
    def sample_free_state(grid_map):
        """
        Return the coordinates of a random free state from the 2D input map
        """
        while True:
            coord = [random.randrange(grid_map.shape[0]),
                     random.randrange(grid_map.shape[1])]
            if grid_map[coord[0], coord[1], 0] == FREESTATE:
                return coord

    @staticmethod
    def out_of_bounds(grid_map, coord):
        return (coord[0] < 0
                or coord[0] >= grid_map.shape[0]
                or coord[1] < 0
                or coord[1] >= grid_map.shape[1])

    @staticmethod
    def apply_move(coord_in, move):
        coord = coord_in.copy()
        coord[:2] += move[:2]
        return coord

    def check_free(self, coord):
        return (not TigerGridBase.out_of_bounds(self.grid, coord)
                and self.grid[coord[0], coord[1]] != OBSTACLE)

    @staticmethod
    def gen_grid(grid_n, grid_m):
        # Initialize grid-world.
        grid = np.zeros([grid_n, grid_m])

        # Set obstacles.
        grid[1, grid_m//2] = OBSTACLE

        return grid

    def obs_lin_to_bin(self, obs_lin):
        obs = np.array(np.unravel_index(obs_lin, [2, 2, 2, 2]), 'i')
        if obs.ndim > 2:
            raise NotImplementedError
        elif obs.ndim > 1:
            obs = np.transpose(obs, [1, 0])
        return obs

    def obs_bin_to_lin(self, obs_bin):
        return np.ravel_multi_index(obs_bin, [2, 2, 2, 2])

    def state_lin_to_bin(self, state_lin):
        return np.unravel_index(state_lin, self.grid_shape)

    def state_bin_to_lin(self, state_coord):
        return np.ravel_multi_index(state_coord, self.grid_shape)

    @staticmethod
    def create_db(filename, params, total_env_count=None, traj_per_env=None):
        """
        :param filename: file name for database
        :param params: dotdict describing the domain
        :param total_env_count: total number of environments in the dataset
        (helps to preallocate space)
        :param traj_per_env: number of trajectories per environment
        """
        grid_n = params.grid_n
        grid_m = params.grid_m
        num_state = 36
        if total_env_count is not None and traj_per_env is not None:
            total_traj_count = total_env_count * traj_per_env
        else:
            total_traj_count = 0

        if os.path.isfile(filename):
            print(filename + " already exist, opening.")
            return tables.open_file(filename, mode='a')

        db = tables.open_file(filename, mode='w')

        db.create_earray(where=db.root,
                         name='envs',
                         atom=tables.Int32Atom(),
                         shape=(0, grid_n, grid_m),
                         expectedrows=total_env_count)

        db.create_earray(where=db.root,
                         name='expRs',
                         atom=tables.Float32Atom(),
                         shape=(0, ),
                         expectedrows=total_traj_count)

        db.create_earray(where=db.root,
                         name='valids',
                         atom=tables.Int32Atom(),
                         shape=(0, ),
                         expectedrows=total_traj_count)

        db.create_earray(where=db.root,
                         name='bs',
                         atom=tables.Float32Atom(),
                         shape=(0, num_state),
                         expectedrows=total_traj_count)

        db.create_earray(where=db.root,
                         name='steps',
                         atom=tables.Int32Atom(),
                         shape=(0, 3),  # state,  action, observation
                         expectedrows=total_traj_count * 10)  # rough estimate

        db.create_earray(where=db.root,
                         name='samples',
                         atom=tables.Int32Atom(),
                         shape=(0, 5),  # env_id, goal_state, step_id, traj_length, failed
                         expectedrows=total_traj_count)
        return db

    def process_goals(self, goal_state):
        """
        :param goal_state: linear goal state
        :return: goal image, same size as grid
        """
        if goal_state.shape[0] >= 1:  # batch size
            goal_grid = goal_state.copy()
            goal_grid[np.nonzero(goal_state < 7)] //= 4
            goal_grid[np.nonzero(goal_state >= 7)] = \
                goal_grid[np.nonzero(goal_state >= 7)]//4 + 1

        else:
            print("Error: batch size < 1.")
            assert False

        goal_img = np.zeros([goal_grid.shape[0], self.N, self.M], 'i')
        goal_idx = np.unravel_index(goal_grid, [self.N, self.M])

        goal_img[
            np.arange(goal_grid.shape[0]),
            goal_idx[0],
            goal_idx[1]
        ] = 1

        return goal_img

    def process_beliefs(self, linear_belief):
        """
        :param linear_belief: belief in linear space
        :return: belief reshaped to grid size
        """
        batch = (linear_belief.shape[0] if linear_belief.ndim > 1 else 1)
        b = linear_belief.reshape(
            [batch, self.params.num_state]
        )
        if b.dtype != np.float:
            return b.astype('f')

        return b


def generate_grid_data(path,
                       grid_n=2,
                       grid_m=5,
                       num_env=10000,
                       traj_per_env=5):
    """
    :param path: path for data file. use separate folders for training and
    test data
    :param grid_n: grid rows
    :param grid_m: grid columns
    :param num_env: number of environments in the data set (grids)
    :param traj_per_env: number of trajectories per environment
    (different initial state, goal, initial belief)
    """

    params = dotdict({
        'grid_n': grid_n,
        'grid_m': grid_m,

        # Initial states
        'init_states': [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 24,
                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        'init_2_states': [[24, 28], [23, 29], [22, 30], [21, 31], [20, 32]],
        'goal_states': [8, 9, 10, 11],
        'negative_states': [0, 1, 2, 3, 16, 17, 18, 19],

        # Classify states w.r.t feature of distribution of walls' directions
        'states_wall_above_left': [0, 17, 23, 26, 31, 34],
        'states_wall_above_right': [3, 16, 22, 25, 30, 33],
        'states_wall_below_left': [1, 18, 20, 27, 28, 35],
        'states_wall_below_right': [2, 19, 21, 24, 29, 32],
        'states_wall_above_only': [4, 12],
        'states_wall_below_only': [6, 14],
        'states_wall_left_only': [5, 13],
        'states_wall_right_only': [7, 15],

        'reward_goal': 1,
        'reward_pit': -1,
        'reward_others': 0,

        'discount': 0.9,
        # 'prob_succ_trans': prob_succ_trans,
        # 'prob_succ_obs': prob_succ_obs,

        'num_action': 5,
        'init_action': 0,

        'num_obs': 17,
        'observe_directions': [[0, 1], [1, 0], [0, -1], [-1, 0]],
        })

    params['obs_len'] = len(params['observe_directions']) + 1
    params['num_state'] = 36
    params['traj_limit'] = 4 * (params['grid_n'] + params['grid_m'])

    # save params
    if not os.path.isdir(path):
        os.mkdir(path)
    pickle.dump(dict(params), open(path + "/params.pickle", 'wb'), -1)

    # randomize seeds, set to previous value to determine random numbers
    np.random.seed()
    random.seed()

    # grid domain object
    domain = TigerGridBase(params)

    # make database file
    db = TigerGridBase.create_db(filename=path+"data.hdf5",
                                 params=params,
                                 total_env_count=num_env,
                                 traj_per_env=traj_per_env)

    for env_i in range(num_env):
        print("Generating env %d with %d trajectories "
              % (env_i, traj_per_env))
        domain.generate_trajectories(db, num_traj=traj_per_env)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description='Generate grid environments')
    parser.add_argument(
        '--path',
        type=str,
        help='Directory for data sets')
    parser.add_argument(
        '--train',
        type=int,
        default=10000,
        help='Number of training environments')
    parser.add_argument(
        '--test',
        type=int,
        default=500,
        help='Number of test environments')
    parser.add_argument(
        '--N',
        type=int,
        default=2,
        help='Grid size height')
    parser.add_argument(
        '--M',
        type=int,
        default=5,
        help='Grid size width'
    )
    parser.add_argument(
        '--train_trajs',
        type=int,
        default=5,
        help='Number of trajectories per environment in the training set. '
             '1 by default.')
    parser.add_argument(
        '--test_trajs',
        type=int,
        default=1,
        help='Number of trajectories per environment in the test set. '
             '1 by default.')

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        os.mkdir(args.path)

    # training data
    generate_grid_data(args.path + '/train/', grid_n=args.N, grid_m=args.M,
                       num_env=args.train,
                       traj_per_env=args.train_trajs)

    # test data
    generate_grid_data(args.path + '/test/', grid_n=args.N, grid_m=args.M,
                       num_env=args.test,
                       traj_per_env=args.test_trajs)


# default
if __name__ == "__main__":
    main()
