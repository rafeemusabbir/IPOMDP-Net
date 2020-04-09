import argparse
import random
import os

import numpy as np
import _pickle as pickle
import tables

from utils.dotdict import dotdict

try:
    import ipdb as pdb
except Exception:
    import pdb

from utils.interactive_particle_filter import IParticleFilter, \
    joint_to_lin, lin_to_joint


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
        self.num_agent = params.num_agent
        self.num_door = params.num_door
        self.num_tiger = params.num_tiger
        self.num_gold = params.num_gold

        self.agent_loc_space = params.agent_loc_space
        self.agent_birth_space = params.agent_birth_space
        self.door_loc_space = params.door_loc_space

        self.phys_state_space = params.phys_state_space
        self.action_space = params.action_space
        self.obs_space = params.obs_space

        self.observe_directions = params.observe_directions

        self.num_action = params.num_action
        self.num_joint_action = params.num_joint_action
        self.num_obs = params.num_obs

        self.trans_func = None
        self.reward_func = None
        self.obs_func = None

        self.pr_trans_intent = params.pr_trans_intent
        self.pr_obs_dir_corr = params.pr_obs_dir_corr
        self.pr_obs_door_corr = params.pr_obs_door_corr

        self.moves = params.moves

        self.grid = None
        self.gold_cell = None
        self.tiger_cell = None

    # def simulate_policy(self, policy, grid, b0, init_state,
    #                     init_common_state, first_action=None):
    #     params = self.params
    #     max_traj_len = params.traj_limit
    #
    #     if not first_action:
    #         first_action = params.init_action
    #
    #     self.grid = grid
    #
    #     self.gen_ipomdp(init_state[0])
    #
    #     state = init_state
    #     reward_sum = 0.0  # accumulated reward
    #
    #     failed = False
    #     step_i = 0
    #
    #     # initialize policy
    #     env_img = grid[None]
    #     b0_img = self.process_beliefs(b0)
    #     policy.reset(env_img, b0_img)
    #
    #     act = first_action
    #     obs = None
    #
    #     while True:
    #         # finish if state is terminal, i.e. we reached a goal state
    #         if all([np.isclose(qmdp.T[x][state, state], 1.0)
    #                 for x in range(params.num_action)]):
    #             assert state in goal_states
    #             break
    #
    #         # stop if trajectory limit reached
    #         if step_i >= max_traj_len:
    #             failed = True
    #             break
    #
    #         # choose next action
    #         # if it is the initial step, stay will always be selected;
    #         # otherwise the action is selected based on the learned policy
    #         if step_i:
    #             if obs >= 16:
    #                 act = first_action
    #             else:
    #                 obs_without_goal = self.obs_lin_to_bin(obs)
    #                 if list(obs_without_goal) == [0, 1, 0, 1] or \
    #                         list(obs_without_goal) == [1, 0, 1, 0]:
    #                     obs_with_goal = np.array([0, 0, 0, 0, 1])
    #                 else:
    #                     obs_with_goal = np.append(obs_without_goal, 0)
    #                 act = policy.eval(act, obs_with_goal)
    #
    #         # simulate action
    #         state, r = qmdp.transition(state, act)
    #         obs = qmdp.random_obs(state, act)
    #
    #         # Update expected/accumulated reward.
    #         reward_sum += r
    #
    #         step_i += 1
    #
    #     traj_len = step_i
    #
    #     return (not failed), traj_len, reward_sum

    def generate_trajectories(self, db, num_traj):
        params = self.params
        max_traj_len = params.traj_limit

        for traj_i in range(num_traj):
            # generate a QMDP object, initial belief, initial state and
            # goal state, also generates a random grid for the first iteration
            i_pf, b0, ib0, \
            init_state_self, \
            init_state_others = self.random_instance((traj_i == 0))

            state_self = init_state_self
            state_self_lin = state_self.copy()
            phys_state_self = state_self[:2]
            phys_state_self_lin = joint_to_lin(
                phys_state_self, (self.agent_loc_space, self.door_loc_space))
            state_self_lin[0] = phys_state_self_lin
            state_others = init_state_others
            state_others_lin = joint_to_lin(
                state_others, (self.agent_loc_space, self.door_loc_space))

            b_others = b0.copy()
            b_self = ib0.copy()
            reward_sum = 0.0  # accumulated reward

            # Trajectory of beliefs
            beliefs_self = [ib0]
            beliefs_others = [b0]
            # Trajectory of states
            states_self = [state_self]
            states_self_lin = [state_self_lin]
            states_others = [state_others]
            states_others_lin = [state_others_lin]
            # Trajectory of actions: first action is always stay.
            actions_self = list()
            actions_others = list()
            # Trajectory of rewards
            rewards = list()
            # Trajectory of observations
            observs_self = list()
            observs_others = list()

            failed = False
            step_i = 0

            while True:
                # stop if trajectory limit reached
                if step_i >= max_traj_len:
                    failed = True
                    break

                # choose action
                if step_i == 0:
                    # dummy first action
                    act_self = act_others = params.init_action
                else:
                    act_self = i_pf.qmdp_policy(b_self)
                    act_others = i_pf.l0_policy(b_others)

                actions_self.append(act_self)
                actions_others.append(act_others)
                act_joint = [act_self, act_others]

                #  Simulate action
                phys_state_prime_self, \
                phys_state_prime_self_lin, r = self.transit(
                    state=phys_state_self,
                    action=act_joint)
                state_prime_others, state_prime_others_lin, _ = self.transit(
                    state=state_others,
                    action=act_joint)

                rewards.append(r)
                reward_sum += r

                # Finish if either agent opens either door.
                cell_self = state_self[0]
                cell_others = state_others[0]

                if np.any(cell_self == np.array(
                        [self.gold_cell, self.tiger_cell])) and \
                        (act_self == self.action_space[-1]):
                    break

                if np.any(cell_others == np.array(
                        [self.gold_cell, self.tiger_cell])) and \
                        (act_others == self.action_space[-1]):
                    break

                phys_state_self = phys_state_prime_self.copy()
                phys_state_self_lin = phys_state_prime_self_lin
                state_others = state_prime_others
                state_others_lin = state_prime_others_lin
                states_others.append(state_others)
                states_others_lin.append(state_others_lin)

                obs_self = self.observe(
                    state=phys_state_self,
                    action=act_joint)
                obs_others = self.observe(
                    state=state_others,
                    action=act_joint)

                observs_self.append(obs_self)
                observs_others.append(obs_others)

                b_others = i_pf.l0_particle_filtering(
                    l0belief=b_others,
                    act=act_others,
                    obs=obs_others)
                state_self = np.concatenate([phys_state_self, b_others])
                state_self_lin = np.concatenate(
                    [[phys_state_self_lin], b_others])
                states_self.append(state_self)
                states_self_lin.append(state_self_lin)

                b_self = i_pf.i_particle_filtering(
                    ibelief=b_self,
                    act=act_self,
                    obs=obs_self)
                beliefs_self.append(b_self)
                beliefs_others.append(b_others)

                step_i += 1

            # add to database
            if not failed:
                db.root.valids.append([len(db.root.samples)])

            traj_len = step_i

            # step: state (linear), action, observation (linear)
            step = np.stack(
                [states_self[:traj_len],
                 actions_self[:traj_len],
                 observs_self[:traj_len]], axis=1)

            print("--THE STATE TRAJECTORY OF i--")
            for i in range(traj_len):
                print(states_self[i])
            print("-" * 30)
            print("--THE STATE TRAJECTORY OF j--")
            for i in range(traj_len):
                print(states_others[i])
            print("-" * 30)
            print("--THE BELIEF TRAJECTORY OF i--")
            for i in range(traj_len):
                print("Step:", i)
                print(beliefs_self[i])
            print("-" * 30)
            print("--THE BELIEF TRAJECTORY OF j--")
            for i in range(traj_len):
                print("Step:", i)
                print(beliefs_others[i])
            print("-" * 30)
            print("--THE ACTION TRAJECTORY OF i--")
            for i in range(traj_len):
                print(actions_self[i])
            print("-" * 30)
            print("--THE ACTION TRAJECTORY OF j--")
            for i in range(traj_len):
                print(actions_others[i])
            print("-" * 30)
            print("--THE REWARD TRAJECTORY OF i--")
            for i in range(traj_len):
                print(rewards[i])
            print("-" * 30)
            print("--THE OBSERVATION TRAJECTORY OF i--")
            for i in range(traj_len):
                print(observs_self[i])
            print("-" * 30)
            print("--THE OBSERVATION TRAJECTORY OF j--")
            for i in range(traj_len):
                print(observs_others[i])
            print("-" * 30)

            # sample: env_id, goal_state, step_id, traj_len, failed
            # length includes both start and goal (so one step path is length 2)
            sample = np.array(
                [len(db.root.envs), len(db.root.steps), traj_len, failed], 'i')

            db.root.samples.append(sample[None])
            db.root.bs.append(np.array(beliefs_self[:1]))  # only picks init_bs
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
        if generate_grid:
            self.grid = self.gen_grid(
                grid_n=self.params.grid_n, grid_m=self.params.grid_m)

            # The two door-cells are fixed for one environment over
            # different instances.
            door_cells = self.gen_door_cells()
            self.gold_cell, self.tiger_cell = door_cells

        # sample initial belief
        # b0: initial belief uniformly distributed over (4 * 4) states
        b0, ib0 = self.gen_init_belief()

        # Sample initial cells for the two agents.
        init_cells = self.gen_init_cells()
        init_self_cell, init_others_cell = init_cells
        init_phys_state_self = np.array([init_self_cell, self.gold_cell])
        init_inter_state_self = np.concatenate([init_phys_state_self, b0])
        init_phys_state_others = np.array([init_others_cell, self.gold_cell])

        print("THE INITIAL PHYSICAL STATE FOR j:")
        print(init_phys_state_others)
        print("THE INITIAL BELIEF FOR AGENT j:")
        print(b0)
        print("THE INITIAL PHYSICAL STATE FOR i:")
        print(init_phys_state_self)
        print("THE INITIAL INTERACTIVE STATE FOR i:")
        print(init_inter_state_self)
        print("THE INITIAL BELIEF FOR AGENT i:")
        print(ib0)

        # Generate POMDP model: self.T, self.Z, self.R
        self.gen_ipomdp(self.gold_cell)

        # Initiate an instance of I-PF
        i_pf = IParticleFilter(
            trans_func=self.trans_func,
            reward_func=self.reward_func,
            obs_func=self.obs_func,
            params=self.params)

        return i_pf, b0, ib0, init_inter_state_self, \
               init_phys_state_others

    def gen_ipomdp(self, gold_cell):
        # Construct all I-POMDP model (obs_func, trans_func, reward_func)
        self.obs_func = self.build_obs_func(gold_cell)
        self.trans_func, self.reward_func = \
            self.build_trans_reward_func(gold_cell)

    def build_obs_func(self, gold_cell):
        """
        Build the observation model (obs_func) for a grid.
        :param gold_cell: the cell (linear representation) with
        the gold door in it.
        :return: obs_func
        """
        gold_coord = self.cell_lin_to_bin(gold_cell)
        num_phys_state = self.phys_state_space.shape[0]

        obs_func = np.zeros(
            [self.num_joint_action, num_phys_state, self.num_obs], dtype='f')

        for i in range(self.N):
            for j in range(self.M):
                cell_coord = np.array([i, j])
                cell = self.cell_bin_to_lin(cell_coord)
                state = np.array([cell, gold_cell])
                state_lin = joint_to_lin(
                    state, (self.agent_loc_space, self.door_loc_space))

                for act_other in self.action_space:

                    # action: stay and listen
                    act_stay = self.action_space[0]
                    act_joint = [act_stay, act_other]
                    act_joint_lin = joint_to_lin(
                        act_joint, (self.action_space, self.action_space))

                    if i > gold_coord[0]:  # gold cell is to the north
                        obs_ml = self.obs_space[0]
                        obs_side = self.obs_space[1:4]
                        obs_func[act_joint_lin,
                                 state_lin,
                                 obs_ml] = self.pr_obs_dir_corr
                        obs_func[act_joint_lin, state_lin, obs_side] = \
                            (1 - self.pr_obs_dir_corr) / len(obs_side)

                        if j < gold_coord[1]:  # gold cell is in the northeast
                            obs_ml = self.obs_space[:2]
                            obs_side = self.obs_space[2:4]
                            obs_func[act_joint_lin, state_lin, obs_ml] = \
                                (1 - obs_func[act_joint_lin, state_lin, obs_side]
                                 * len(obs_side)) / len(obs_ml)
                        if j > gold_coord[1]:  # gold cell is in the northwest
                            obs_ml = np.array([self.obs_space[0],
                                               self.obs_space[3]])
                            obs_side = self.obs_space[1:3]
                            obs_func[act_joint_lin, state_lin, obs_ml] = \
                                (1 - obs_func[act_joint_lin, state_lin, obs_side]
                                 * len(obs_side)) / len(obs_ml)

                    elif i < gold_coord[0]:  # gold cell is in the south
                        obs_ml = self.obs_space[2]
                        obs_side = np.concatenate(
                            (self.obs_space[:2], self.obs_space[3:4]))
                        obs_func[act_joint_lin,
                                 state_lin,
                                 obs_ml] = self.pr_obs_dir_corr
                        obs_func[act_joint_lin, state_lin, obs_side] = \
                            (1 - self.pr_obs_dir_corr) / len(obs_side)

                        if j < gold_coord[1]:  # gold cell is in the southeast
                            obs_ml = self.obs_space[1:3]
                            obs_side = [self.obs_space[0], self.obs_space[3]]
                            obs_func[act_joint_lin, state_lin, obs_ml] = \
                                (1 - obs_func[act_joint_lin, state_lin, obs_side]
                                 * len(obs_side)) / len(obs_ml)

                        if j > gold_coord[1]:  # gold cell is in the southwest
                            obs_ml = self.obs_space[2:4]
                            obs_side = self.obs_space[:2]
                            obs_func[act_joint_lin, state_lin, obs_ml] = \
                                (1 - obs_func[act_joint_lin, state_lin, obs_side]
                                 * len(obs_side)) / len(obs_ml)

                    elif j < gold_coord[1]:  # gold cell is in the east
                        obs_ml = self.obs_space[1]
                        obs_side = np.concatenate(
                            (self.obs_space[:1], self.obs_space[2:4]))
                        obs_func[act_joint_lin,
                                 state_lin,
                                 obs_ml] = self.pr_obs_dir_corr
                        obs_func[act_joint_lin, state_lin, obs_side] = \
                            (1 - self.pr_obs_dir_corr) / len(obs_side)

                    elif j > gold_coord[1]:  # gold cell is in the west
                        obs_ml = self.obs_space[3]
                        obs_side = self.obs_space[:3]
                        obs_func[act_joint_lin,
                                 state_lin,
                                 obs_ml] = self.pr_obs_dir_corr
                        obs_func[act_joint_lin, state_lin, obs_side] = \
                            (1 - self.pr_obs_dir_corr) / len(obs_side)

                    else:
                        obs_ml = self.obs_space[4]
                        obs_func[act_joint_lin,
                                 state_lin,
                                 obs_ml] = self.pr_obs_door_corr

                    # action: move actions & door-opening
                    act_mob = self.action_space[1:]
                    act_joint = [[a, act_other] for a in act_mob]
                    act_joint_lin = [joint_to_lin(
                        aj, (self.action_space, self.action_space))
                        for aj in act_joint]
                    for obs in self.obs_space[:4]:
                        obs_func[act_joint_lin, state_lin, obs] = \
                            1 / np.shape(self.obs_space[:4])[0]

        print("OBSERVATION FUNCTION")
        print(obs_func)

        return obs_func

    def build_trans_reward_func(self, gold_cell):
        """
        Build transition (trans_func) and reward (reward_func) model for a grid.
        :return:
        trans_func: the transition function followed by level-0 (modeled) agent
        reward_func: the reward function followed by level-0 agent
        """
        num_phys_state = self.phys_state_space.shape[0]

        gold_coord = self.cell_lin_to_bin(gold_cell)
        tiger_coord = [gold_coord[0], self.M - 1 - gold_coord[1]]
        tiger_cell = self.cell_bin_to_lin(tiger_coord)

        # Probability of transition with a0 from s to s'
        trans_func = np.zeros(
            shape=(self.num_joint_action, num_phys_state, num_phys_state),
            dtype='f')

        # Scalar reward of transition with a0 from s1 to s2 (only depend on s2)
        reward_func = np.zeros(
            shape=(self.num_joint_action, num_phys_state), dtype='f')

        for i in range(self.N):
            for j in range(self.M):
                self_coord = np.array([i, j])
                self_cell = self.cell_bin_to_lin(self_coord)
                state = np.array([self_cell, gold_cell])
                state_lin = joint_to_lin(
                    state, (self.agent_loc_space, self.door_loc_space))

                for act_other in self.action_space:
                    # action: stay & listen
                    act_stay = self.action_space[0]
                    act_joint = [act_stay, act_other]
                    act_joint_lin = joint_to_lin(
                        act_joint, (self.action_space, self.action_space))
                    state_prime_lin = state_lin
                    trans_func[act_joint_lin, state_lin, state_prime_lin] = 1.0
                    reward_func[act_joint_lin,
                                state_lin] = self.params.reward_listen

                    # action: open the door
                    act_open = self.action_space[-1]
                    act_joint = [act_open, act_other]
                    act_joint_lin = joint_to_lin(
                        act_joint, (self.action_space, self.action_space))
                    if state[0] == gold_cell:  # the agent is in the gold cell
                        self_cell_next = self.params.agent_birth_space
                        state_prime = np.array(
                            [[x, gold_cell] for x in self_cell_next])
                        state_prime_lin = np.array([joint_to_lin(
                            s, (self.agent_loc_space, self.door_loc_space))
                            for s in state_prime])
                        trans_func[act_joint_lin,
                                   state_lin,
                                   state_prime_lin] = 1.0 / len(self_cell_next)
                        reward_func[act_joint_lin,
                                    state_lin] = self.params.reward_gold
                    elif state[0] == tiger_cell:
                        self_cell_next = self.params.agent_birth_space
                        state_prime = np.array(
                            [[x, gold_cell] for x in self_cell_next])
                        state_prime_lin = np.array([joint_to_lin(
                            s, (self.agent_loc_space, self.door_loc_space))
                            for s in state_prime])
                        trans_func[act_joint_lin,
                                   state_lin,
                                   state_prime_lin] = 1.0 / len(self_cell_next)
                        reward_func[act_joint_lin,
                                    state_lin] = self.params.reward_tiger
                    else:
                        state_prime_lin = state_lin
                        trans_func[act_joint_lin,
                                   state_lin,
                                   state_prime_lin] = 1.0
                        reward_func[act_joint_lin,
                                    state_lin] = self.params.reward_wrong_open

                    # action: move-N, move-E, move-S, move-W
                    for act in self.action_space[1:-1]:
                        act_joint = [act, act_other]
                        act_joint_lin = joint_to_lin(
                            act_joint, (self.action_space, self.action_space))
                        intent_coord = self.apply_move(
                            coord_in=self_coord, move=self.moves[act])
                        side_coords = list()
                        if self.is_outbound(intent_coord):
                            intent_coord = self_coord
                        intent_cell = self.cell_bin_to_lin(intent_coord)
                        state_prime = np.array([intent_cell, gold_cell])
                        state_prime_lin = joint_to_lin(
                            state_prime,
                            (self.agent_loc_space, self.door_loc_space))
                        trans_func[act_joint_lin,
                                   state_lin,
                                   state_prime_lin] = self.pr_trans_intent
                        if self.moves[act][0]:  # [1, 0] or [-1, 0]
                            side_moves = [x for x in self.moves if x[1]]
                            side_coords = \
                                np.array([self.apply_move(self_coord, y)
                                          for y in side_moves])
                        elif self.moves[act][1]:
                            side_moves = [x for x in self.moves if x[0]]
                            side_coords = \
                                np.array([self.apply_move(self_coord, y)
                                          for y in side_moves])
                        for coord in side_coords:
                            if self.is_outbound(coord):
                                coord = self_coord
                            side_cell = self.cell_bin_to_lin(coord)
                            if side_cell == self_cell:
                                state_prime_lin = state_lin
                                trans_func[act_joint_lin,
                                           state_lin,
                                           state_prime_lin] += \
                                    (1 - self.pr_trans_intent) / 2
                            else:
                                state_prime = np.array([side_cell, gold_cell])
                                state_prime_lin = joint_to_lin(
                                    state_prime,
                                    (self.agent_loc_space, self.door_loc_space))
                                trans_func[act_joint_lin,
                                           state_lin,
                                           state_prime_lin] = \
                                    (1 - self.pr_trans_intent) / 2

                        reward_func[act_joint_lin, :] = self.params.reward_move

        print("TRANSITION FUNCTION")
        print(trans_func)
        print("REWARD FUNCTION")
        print(reward_func)
        return trans_func, reward_func

    def transit(self, state, action):
        if np.ndim(state):
            state = joint_to_lin(
                state, (self.agent_loc_space, self.door_loc_space))
        if np.ndim(action):
            action = joint_to_lin(
                action, (self.agent_loc_space, self.door_loc_space))

        dist = self.trans_func[action, state, :]
        state_prime_lin = np.random.choice(
            np.arange(self.params.num_phys_state), p=dist)
        state_prime = lin_to_joint(
            state_prime_lin, (self.agent_loc_space, self.door_loc_space))
        reward = self.reward_func[action, state]

        return state_prime, state_prime_lin, reward

    def observe(self, state, action):
        if np.ndim(state):
            state = joint_to_lin(
                state, (self.agent_loc_space, self.door_loc_space))
        if np.ndim(action):
            action = joint_to_lin(
                action, (self.action_space, self.action_space))

        dist = self.obs_func[action, state, :]
        obs = np.random.choice(self.obs_space, p=dist)

        return obs

    def apply_move(self, coord_in, move):
        coord = coord_in.copy()
        coord += move

        return coord

    def gen_door_cells(self):
        gold_cell = np.random.choice(self.params.door_loc_space)
        gold_coord = self.cell_lin_to_bin(gold_cell)
        tiger_coord = [gold_coord[0], self.M - 1 - gold_coord[1]]
        tiger_cell = self.cell_bin_to_lin(tiger_coord)

        return gold_cell, tiger_cell

    def gen_init_cells(self):
        return np.random.choice(
            self.params.agent_birth_space, size=self.num_agent)

    def gen_init_belief(self):
        # level-0 initial belief over physical state space
        b0 = np.zeros(self.phys_state_space.shape[0])
        birth_indices = list()

        for i in self.agent_birth_space:
            for j in self.door_loc_space:
                birth_indices.append(joint_to_lin(
                    [i, j], (self.agent_loc_space, self.door_loc_space)))

        b0[birth_indices] = np.divide(1, len(birth_indices))

        # Initial belief over interactive state space
        ib0 = np.zeros((self.phys_state_space.shape[0], 2), dtype='object')

        for i in self.agent_loc_space:
            for j in self.door_loc_space:
                idx = joint_to_lin(
                    [i, j], (self.agent_loc_space, self.door_loc_space))
                ib0[idx][0] = np.concatenate([[idx], b0])

        for i in self.agent_birth_space:
            for j in self.door_loc_space:
                idx = joint_to_lin(
                    [i, j], (self.agent_loc_space, self.door_loc_space))
                ib0[idx][1] = b0[idx]

        return b0, ib0

    @staticmethod
    def gen_grid(grid_n, grid_m):
        if grid_m % 2:
            print("Currently the grid_m is required to be even.")
        assert not grid_m % 2
        # Initialize grid-world.
        grid = np.zeros([grid_n, grid_m])

        return grid

    def is_outbound(self, coord_in):
        assert len(coord_in) == 2
        if 0 <= coord_in[0] < self.N:
            if 0 <= coord_in[1] < self.M:
                return False

            return True

        return True

    def obs_lin_to_bin(self, obs_lin):
        obs = np.array(np.unravel_index(obs_lin, [2, 2, 2, 2]), 'i')
        if obs.ndim > 2:
            raise NotImplementedError
        elif obs.ndim > 1:
            obs = np.transpose(obs, [1, 0])
        return obs

    def obs_bin_to_lin(self, obs_bin):
        return np.ravel_multi_index(obs_bin, [2, 2, 2, 2])

    def cell_lin_to_bin(self, cell_lin):
        return np.unravel_index(cell_lin, self.grid_shape)

    def cell_bin_to_lin(self, cell_coord):
        return np.ravel_multi_index(cell_coord, self.grid_shape)

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
        num_phys_state = params.num_phys_state
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
                         shape=(0,),
                         expectedrows=total_traj_count)

        db.create_earray(where=db.root,
                         name='valids',
                         atom=tables.Int32Atom(),
                         shape=(0,),
                         expectedrows=total_traj_count)

        db.create_earray(where=db.root,
                         name='bs',
                         atom=tables.Float32Atom(),
                         shape=(0, num_phys_state),
                         expectedrows=total_traj_count)

        db.create_earray(where=db.root,
                         name='steps',
                         atom=tables.Int32Atom(),
                         shape=(0, 3),  # state,  action, observation
                         expectedrows=total_traj_count * 10)  # rough estimate

        db.create_earray(where=db.root,
                         name='samples',
                         atom=tables.Int32Atom(),
                         shape=(0, 5),
                         # env_id, goal_state, step_id, traj_length, failed
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
                goal_grid[np.nonzero(goal_state >= 7)] // 4 + 1

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
            [batch, self.params.num_phys_state]
        )
        if b.dtype != np.float:
            return b.astype('f')

        return b


def generate_grid_data(path,
                       grid_n,
                       grid_m,
                       num_gold,
                       num_tiger,
                       num_agent,
                       p_trans_intent,
                       p_obs_dir_corr,
                       p_obs_door_corr,
                       num_env=10000,
                       traj_per_env=5):
    """
    :param path: path for data file. use separate folders for training and
    test data
    :param grid_n: grid rows
    :param grid_m: grid columns
    :param num_gold: the number of cells with a door where gold hides behind
    :param num_tiger: the number of cells with a door where a tiger hides behind
    :param num_agent: the number of agents in the multi-agent system
    :param num_env: the number of environments in the data set (grids)
    :param traj_per_env: the number of trajectories per environment
    (different initial state, goal, initial belief)
    :param p_trans_intent: probability that the agent reaches intentional state
    given the previous state and action.
    :param p_obs_dir_corr: probability that the agent observes the direction of
    the gold-door correctly.
    :param p_obs_door_corr: probability that the agent observes that it has
    found the door cell correctly.
    """

    params = dotdict({
        # basic environmental configuration
        'grid_n': grid_n,
        'grid_m': grid_m,
        'num_agent': num_agent,
        'num_gold': num_gold,
        'num_tiger': num_tiger,
        'num_door': num_gold + num_tiger,

        # configs for PF/I-PF
        'num_particle_pf': 1000,

        # state-related
        'agent_loc_space': np.arange(grid_n * grid_m),
        'agent_birth_space': np.arange(grid_n * grid_m)[-grid_m:],
        'door_loc_space': np.arange(grid_n * grid_m)[:grid_m],

        # reward-related
        'reward_gold': 10.0,
        'reward_tiger': -100.0,
        'reward_listen': -1.0,
        'reward_wrong_open': -10.0,
        'reward_move': 0,

        # self_action-related: stay, move-N, move-E, move-S, move-W,
        # open the door
        'action_space': np.arange(6),
        'moves': [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
        'init_action': 0,

        # observation-related: glitter-N, glitter-E, glitter-S, glitter-W,
        # door-found
        'obs_space': np.arange(5),
        'observe_directions': [[0, 1], [1, 0], [0, -1], [-1, 0]],

        # pomdp_function-related
        'pr_trans_intent': p_trans_intent,
        'pr_obs_dir_corr': p_obs_dir_corr,
        'pr_obs_door_corr': p_obs_door_corr,

        'discount': 0.9,
    })
    params['phys_state_space'] = np.transpose(
        [np.tile(params.agent_loc_space, len(params.door_loc_space)),
         np.repeat(params.door_loc_space, len(params.agent_loc_space))])
    # params['phys_state_space'] = np.transpose(  # common state space
    #     [np.tile(params.phys_state_space, len(params.door_loc_space)),
    #      np.repeat(params.door_loc_space, len(params.phys_state_space))])
    params['num_phys_state'] = len(params['phys_state_space'])
    params['num_action'] = len(params['action_space'])
    params['num_joint_action'] = params['num_action'] ** 2
    params['joint_action_space'] = np.transpose(
        [np.tile(params['action_space'], params['num_action']),
         np.repeat(params['action_space'], params['num_action'])])
    params['num_obs'] = np.shape(params['obs_space'])[0]
    params['traj_limit'] = 4 * (params['grid_n'] + params['grid_m'])  # reason?

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
    db = TigerGridBase.create_db(filename=path + "data.hdf5",
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
        help='The directory for data sets.')
    parser.add_argument(
        '--train',
        type=int,
        default=10000,
        help='The number of training environments.')
    parser.add_argument(
        '--test',
        type=int,
        default=500,
        help='The number of test environments.')
    parser.add_argument(
        '--N',
        type=int,
        default=4,
        help='The height of the grid world.')
    parser.add_argument(
        '--M',
        type=int,
        default=4,
        help='The width of the grid world.')
    parser.add_argument(
        '--agent',
        type=int,
        default=2,
        help='The number of agents in the multi-agent system.')
    parser.add_argument(
        '--gold',
        type=int,
        default=1,
        help='The number of cells with a door where gold hides behind.')
    parser.add_argument(
        '--tiger',
        type=int,
        default=1,
        help='The number of cells with a door where a tiger hides behind.')
    parser.add_argument(
        '--p_trans_intent',
        type=float,
        default=0.9,
        help='The probability that the agent reaches the intentional state.')
    parser.add_argument(
        '--p_obs_dir_corr',
        type=float,
        default=0.85,
        help='The probability that the agent receives the correct observation'
             'of the direction of the door when it is not in a door-cell.')
    parser.add_argument(
        '--p_obs_door_corr',
        type=float,
        default=1.0,
        help='The probability that the agent receives the correct observation'
             'when it occupies a door-cell.')
    parser.add_argument(
        '--train_trajs',
        type=int,
        default=5,
        help='The number of trajectories per environment in the training set. '
             '1 by default.')
    parser.add_argument(
        '--test_trajs',
        type=int,
        default=1,
        help='The number of trajectories per environment in the test set. '
             '1 by default.')

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        os.mkdir(args.path)

    # training data
    generate_grid_data(args.path + '/train/',
                       grid_n=args.N, grid_m=args.M,
                       num_agent=args.agent,
                       num_tiger=args.tiger,
                       num_gold=args.gold,
                       p_trans_intent=args.p_trans_intent,
                       p_obs_dir_corr=args.p_obs_dir_corr,
                       p_obs_door_corr=args.p_obs_door_corr,
                       num_env=args.train,
                       traj_per_env=args.train_trajs)

    # test data
    generate_grid_data(args.path + '/test/',
                       grid_n=args.N, grid_m=args.M,
                       num_agent=args.agent,
                       num_tiger=args.tiger,
                       num_gold=args.gold,
                       p_trans_intent=args.p_trans_intent,
                       p_obs_dir_corr=args.p_obs_dir_corr,
                       p_obs_door_corr=args.p_obs_door_corr,
                       num_env=args.test,
                       traj_per_env=args.test_trajs)


# default
if __name__ == "__main__":
    main()
