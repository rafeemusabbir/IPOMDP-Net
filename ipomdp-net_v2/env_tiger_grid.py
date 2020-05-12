import argparse
import random
import os

import numpy as np
import scipy.sparse
import _pickle as pickle
import tables

from utils.dotdict import dotdict

try:
    import ipdb as pdb
except ImportError:
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

        self.l0_phys_state_space = params.l0_phys_state_space
        self.inter_phys_state_space = params.inter_phys_state_space
        self.action_space = params.action_space
        self.joint_action_space = params.joint_action_space
        self.obs_space = params.obs_space

        self.num_l0_phys_state = params.num_l0_phys_state
        self.num_inter_phys_state = params.num_inter_phys_state
        self.num_action = params.num_action
        self.num_joint_action = params.num_joint_action
        self.num_obs = params.num_obs

        self.l0_trans_func = None
        self.l0_reward_func = None
        self.l0_obs_func = None
        self.inter_trans_func = None
        self.inter_reward_func = None
        self.inter_obs_func = None

        self.pr_trans_intent = params.pr_trans_intent
        self.pr_obs_dir_corr = params.pr_obs_dir_corr
        self.pr_obs_door_corr = params.pr_obs_door_corr

        self.moves = params.moves

        self.grid = None
        self.gold_cell = None
        self.tiger_cell = None

    def simulate_policy(self, policy, grid, terminals,
                        l0_b0, ib0, init_phys_state,
                        first_action=None, level=1):
        params = self.params
        max_traj_len = params.traj_limit

        if not first_action:
            first_action = params.init_action

        self.grid = grid

        self.gen_ipomdp()

        phys_state_self_joint = lin_to_joint(
            init_phys_state,
            (self.agent_loc_space,
             self.agent_loc_space,
             self.door_loc_space)) \
            if not np.ndim(init_phys_state) else init_phys_state.copy()
        phys_state_self_lin = joint_to_lin(
            init_phys_state,
            (self.agent_loc_space,
             self.agent_loc_space,
             self.agent_loc_space)) \
            if np.ndim(init_phys_state) else init_phys_state
        reward_sum = 0.0  # accumulated reward

        failed = False
        step_i = 0

        # Initialize the policy.
        env_img = grid[None]
        terminal_img = terminals[None]
        assert ib0.shape[0] == self.num_inter_phys_state
        ib0 = np.array(
            [[np.append(s, l0_b0), ib0[s]] for s in np.arange(ib0.shape[0])])
        policy.reset(env_img, terminal_img, ib0)

        act_self = act_other = first_action
        act_joint_self = joint_to_lin(
            [act_self, act_other], (self.action_space, self.action_space))
        obs_self, obs_other = None, None

        while True:
            cell_self, cell_other = phys_state_self_joint[:2]

            # Terminate if the [s, a] pair trigger the env reset.
            if cell_self in [self.gold_cell, self.tiger_cell] and \
                    (act_self == self.action_space[-1]):
                step_i += 1
                break

            if cell_other in [self.gold_cell, self.tiger_cell] and \
                    (act_other == self.action_space[-1]):
                step_i += 1
                break

            # Terminate if trajectory limit is reached.
            if step_i >= max_traj_len:
                failed = True
                break

            # choose next action
            # if it is the initial step, stay will always be selected;
            # otherwise the action is selected based on the learned policy
            if not step_i:
                # dummy first action
                act_self = act_other = params.init_action
            else:
                act_self, act_other = policy.eval(
                    last_act=act_self, last_obs=obs_self)

            #  Simulate action
            phys_state_prime_self_joint, \
            phys_state_prime_self_lin, r = self.transit(
                state=phys_state_self_lin,
                action=act_joint_self,
                agent_level=level)
            obs_self = self.observe(
                state=phys_state_prime_self_lin,
                action=act_joint_self,
                agent_level=level)

            # Update expected/accumulated reward.
            reward_sum += r

            step_i += 1

        traj_len = step_i

        return (not failed), traj_len, reward_sum

    def generate_trajectories(self, db, num_traj, level):
        params = self.params
        max_traj_len = params.traj_limit

        i_pf, b0, ib0 = self.gen_component_env()
        i_pf.solve_qmdp()

        for traj_i in range(num_traj):
            # generate a I-PF instance, initial belief, initial state and
            # goal state, also generates a random grid for the first iteration
            init_state_self, init_state_other = self.gen_component_traj(
                b0_other=b0, print_init=False)

            terminal = np.zeros(self.N * self.M)
            terminal[self.door_loc_space] = 1
            terminal = terminal.reshape(self.grid_shape)

            state_self = init_state_self.copy()
            state_self_lin = state_self.copy()
            phys_state_self = state_self[:3]
            phys_state_self_lin = joint_to_lin(
                phys_state_self,
                (self.agent_loc_space,
                 self.agent_loc_space,
                 self.door_loc_space))
            state_self_lin = np.append(phys_state_self_lin, state_self_lin[3:])
            state_other = init_state_other
            state_other_lin = joint_to_lin(
                state_other, (self.agent_loc_space, self.door_loc_space))

            b_other = b0.copy()
            b_self = ib0.copy()
            reward_sum = 0.0  # accumulated reward

            # Trajectory of beliefs
            beliefs_self = [ib0]
            beliefs_other = [b0]
            # Trajectory of states
            states_self = [state_self]
            states_self_lin = [state_self_lin]
            phys_states_self_lin = [phys_state_self_lin]
            states_other = [state_other]
            states_other_lin = [state_other_lin]
            # Trajectory of actions: first action is always stay.
            actions_self = list()
            actions_other = list()
            # Trajectory of rewards
            rewards = list()
            # Trajectory of observations
            observs_self = list()
            observs_other = list()

            failed = False
            step_i = 0

            while True:
                # stop if trajectory limit reached
                if step_i >= max_traj_len:
                    failed = True
                    break

                # choose action
                if not step_i:
                    # dummy first action
                    act_self = act_other = params.init_action
                    is_first_step = True
                else:
                    act_self = i_pf.qmdp_policy(
                        belief=b_self,
                        agent_level=level,
                        is_self=True)
                    act_other = i_pf.qmdp_policy(
                        belief=b_other,
                        agent_level=level - 1,
                        is_self=False)
                    is_first_step = False
                print("AGENT i CHOOSES ACTION [%d]" % act_self)
                print("AGENT j CHOOSES ACTION [%d]" % act_other)

                actions_self.append(act_self)
                actions_other.append(act_other)
                act_joint_self = [act_self, act_other]
                act_joint_other = [act_other, act_self]

                #  Simulate action
                phys_state_prime_self, \
                phys_state_prime_self_lin, r = self.transit(
                    state=phys_state_self_lin,
                    action=act_joint_self,
                    agent_level=level)
                # state_prime_other, state_prime_other_lin, _ = self.transit(
                #     state=state_other_lin,
                #     action=act_joint,
                #     agent_level=level - 1)
                state_prime_other = phys_state_prime_self[1:]
                state_prime_other_lin = joint_to_lin(
                    state_prime_other,
                    (self.agent_loc_space,
                     self.door_loc_space))

                rewards.append(r)
                reward_sum += r

                obs_self = self.observe(
                    state=phys_state_prime_self_lin,
                    action=act_joint_self,
                    agent_level=level)
                obs_other = self.observe(
                    state=state_prime_other_lin,
                    action=act_joint_other,
                    agent_level=level - 1)
                print("AGENT i OBSERVES [%d]" % obs_self)
                print("AGENT j OBSERVES [%d]" % obs_other)

                observs_self.append(obs_self)
                observs_other.append(obs_other)

                b_other = i_pf.l0_particle_filtering(
                    l0belief=b_other,
                    act=act_other,
                    obs=obs_other)

                # Finish if either agent opens either door.
                cell_self = state_self[0]
                cell_other = state_other[0]

                if cell_self in [self.gold_cell, self.tiger_cell] and \
                        (act_self == self.action_space[-1]):
                    step_i += 1
                    break

                if cell_other in [self.gold_cell, self.tiger_cell] and \
                        (act_other == self.action_space[-1]):
                    step_i += 1
                    break

                phys_state_self = phys_state_prime_self.copy()
                phys_state_self_lin = phys_state_prime_self_lin
                phys_states_self_lin.append(phys_state_self_lin)
                state_other = state_prime_other.copy()
                state_other_lin = state_prime_other_lin
                states_other.append(state_other)
                states_other_lin.append(state_other_lin)
                print("AGENT i is at STATE:", phys_state_self)
                print("AGENT j is at STATE:", state_other)

                state_self = np.concatenate([phys_state_self, b_other])
                state_self_lin = np.append(phys_state_self_lin, b_other)
                states_self.append(state_self)
                states_self_lin.append(state_self_lin)

                b_self = i_pf.i_particle_filtering(
                    ibelief=b_self,
                    act=act_self,
                    obs=obs_self,
                    first_step=is_first_step)
                # print("THE BELIEF OF i:")
                # print(b_self)
                # print("THE BELIEF OF j:")
                # print(b_other)
                beliefs_self.append(b_self)
                beliefs_other.append(b_other)

                step_i += 1

            # add to database
            if not failed:
                db.root.valids.append([len(db.root.samples)])

            traj_len = step_i

            # step: state (linear), action, observation (linear)
            step = np.stack(
                [np.array(phys_states_self_lin[:traj_len]),
                 np.array(actions_self[:traj_len]),
                 np.array(observs_self[:traj_len])],
                axis=1)

            # print("--THE STATE TRAJECTORY OF i--")
            # for i in range(traj_len):
            #     print(states_self[i])
            # print("-" * 20)
            # print("--THE STATE TRAJECTORY OF j--")
            # for i in range(traj_len):
            #     print(states_other[i])
            # print("-" * 20)
            # print("--THE BELIEF TRAJECTORY OF i--")
            # for i in range(traj_len):
            #     print("Step:", i)
            #     print(beliefs_self[i])
            # print("-" * 20)
            # print("--THE BELIEF TRAJECTORY OF j--")
            # for i in range(traj_len):
            #     print("Step:", i)
            #     print(beliefs_other[i])
            # print("-" * 20)
            # print("--THE ACTION TRAJECTORY OF i--")
            # for i in range(traj_len):
            #     print(actions_self[i])
            # print("-" * 20)
            # print("--THE ACTION TRAJECTORY OF j--")
            # for i in range(traj_len):
            #     print(actions_other[i])
            # print("-" * 20)
            # print("--THE REWARD TRAJECTORY OF i--")
            # for i in range(traj_len):
            #     print(rewards[i])
            # print("-" * 20)
            # print("--THE OBSERVATION TRAJECTORY OF i--")
            # for i in range(traj_len):
            #     print(observs_self[i])
            # print("-" * 20)
            # print("--THE OBSERVATION TRAJECTORY OF j--")
            # for i in range(traj_len):
            #     print(observs_other[i])
            # print("-" * 20)

            # sample: env_id, goal_state, step_id, traj_len, failed
            # length includes both start and goal (so one step path is length 2)
            sample = np.array(
                [len(db.root.envs), len(db.root.terminals),
                 len(db.root.steps), traj_len, failed], 'i')

            db.root.samples.append(sample[None])
            db.root.terminals.append(np.array(terminal[None]))
            db.root.inter_bs.append(np.array([ib0[:, 1]]))
            db.root.l0_bs.append(np.array(beliefs_other[:1]))
            db.root.expRs.append([reward_sum])
            db.root.steps.append(step)

        # add environment only after adding all trajectories
        db.root.envs.append(self.grid[None])

    def gen_component_env(self, print_init=False):
        """

        :param print_init:
        :return:
        """
        self.grid = self.gen_grid(self.params.grid_n, self.params.grid_m)

        # The two door-cells are fixed for one environment over
        # different instances.
        door_cells = self.gen_door_cells()
        self.gold_cell, self.tiger_cell = door_cells

        # sample initial belief
        # b0: initial belief uniformly distributed over (4 * 4) states
        b0, ib0 = self.gen_init_belief()

        # Generate POMDP model: self.T, self.Z, self.R
        self.gen_ipomdp()

        # Initiate an instance of I-PF
        i_pf = self.get_i_pf()

        if print_init:
            print("THE INITIAL BELIEF FOR AGENT j:")
            print(b0)
            print("THE INITIAL BELIEF FOR AGENT i:")
            print(ib0)

        return i_pf, b0, ib0

    def gen_component_traj(self, b0_other, print_init=False):
        """
        Generate a random problem instance for a grid.
        Picks a random initial belief, initial state and goal states.
        :param b0_other:
        :param print_init:
        :return:
        """
        # Sample initial cells for the two agents.
        init_cells = self.gen_init_cells()
        init_self_cell, init_other_cell = init_cells
        init_phys_state_self = np.array(
            [init_self_cell, init_other_cell, self.gold_cell])
        init_inter_state_self = np.concatenate([init_phys_state_self, b0_other])
        init_phys_state_other = np.array([init_other_cell, self.gold_cell])

        if print_init:
            print("THE INITIAL PHYSICAL STATE FOR j:")
            print(init_phys_state_other)
            print("THE INITIAL PHYSICAL STATE FOR i:")
            print(init_phys_state_self)
            print("THE INITIAL INTERACTIVE STATE FOR i:")
            print(init_inter_state_self)

        return init_inter_state_self, init_phys_state_other

    def gen_ipomdp(self):
        # Construct all I-POMDP model (obs_func, trans_func, reward_func)
        self.l0_obs_func, self.inter_obs_func = self.build_obs_func("export")
        self.l0_trans_func, self.l0_reward_func, self.inter_trans_func, \
        self.inter_reward_func = self.build_trans_reward_func("export")

    def get_i_pf(self):
        i_pf = IParticleFilter(self.params)

        i_pf.process_trans_func(self.l0_trans_func, self.inter_trans_func)
        i_pf.process_reward_func(self.l0_reward_func, self.inter_reward_func)
        i_pf.process_obs_func(self.l0_obs_func, self.inter_obs_func)

        i_pf.transfer_all_sparse()

        return i_pf

    def build_obs_func(self, info_present=None):
        """
        Build the observation model (obs_func) for a grid.
        the gold door in it.
        :return: obs_func
        """
        num_door_birthplace = self.door_loc_space.shape[0]

        l0_obs_func = np.zeros(
            [self.num_joint_action, self.num_l0_phys_state, self.num_obs],
            dtype='f')
        inter_obs_func = np.zeros(
            [self.num_joint_action, self.num_inter_phys_state, self.num_obs],
            dtype='f')

        for act_other in self.action_space:
            # 1. action <- STAY $ LISTEN
            act_self = self.action_space[0]
            act_joint = np.array([act_self, act_other])
            act_joint_lin = joint_to_lin(
                act_joint, (self.action_space, self.action_space))
            for gold_cell in self.door_loc_space:
                gold_x, gold_y = self.cell_lin_to_bin(gold_cell)
                idx_gold_in_space = np.argwhere(
                    gold_cell == self.door_loc_space)[0][0]
                idx_tiger_in_space = \
                    num_door_birthplace - 1 - idx_gold_in_space
                tiger_cell = self.door_loc_space[idx_tiger_in_space]
                for cell_self in self.agent_loc_space:
                    l0_state = np.array([cell_self, gold_cell])
                    l0_state_lin = joint_to_lin(
                        l0_state, (self.agent_loc_space, self.door_loc_space))
                    # When the agent is located in either the gold cell or the
                    # tiger cell, its observation is 100% "door-found".
                    if cell_self in [gold_cell, tiger_cell]:
                        obs = self.obs_space[-1]
                        l0_obs_func[
                            act_joint_lin,
                            l0_state_lin,
                            obs] = self.pr_obs_door_corr

                    else:
                        # Mistakenly hear "door-found" at a non-door cell.
                        obs_wrong_door = self.obs_space[-1]
                        l0_obs_func[
                            act_joint_lin,
                            l0_state_lin,
                            obs_wrong_door] = 0.01
                        x, y = self.cell_lin_to_bin(cell_self)
                        if x > gold_x:  # gold cell is to the north
                            obs_ml = self.obs_space[0]
                            obs_side = self.obs_space[1:4]
                            l0_obs_func[
                                act_joint_lin,
                                l0_state_lin,
                                obs_ml] = self.pr_obs_dir_corr
                            l0_obs_func[
                                act_joint_lin,
                                l0_state_lin,
                                obs_side] = (1 - self.pr_obs_dir_corr - 0.01) / obs_side.shape[0]

                            if y < gold_y:  # gold cell is in the northeast
                                obs_ml = self.obs_space[:2]
                                obs_side = self.obs_space[2:4]
                                l0_obs_func[
                                    act_joint_lin,
                                    l0_state_lin,
                                    obs_ml] = (1 - 0.01 - l0_obs_func[
                                    act_joint_lin,
                                    l0_state_lin,
                                    obs_side] * obs_side.shape[0]) / obs_ml.shape[0]
                            elif y > gold_y:  # gold cell is in the northwest
                                obs_ml = np.array(
                                    [self.obs_space[0], self.obs_space[3]])
                                obs_side = self.obs_space[1:3]
                                l0_obs_func[
                                    act_joint_lin,
                                    l0_state_lin,
                                    obs_ml] = (1 - 0.01 - l0_obs_func[
                                    act_joint_lin,
                                    l0_state_lin,
                                    obs_side] * obs_side.shape[0]) / obs_ml.shape[0]

                        elif x < gold_x:  # gold cell is in the south
                            obs_ml = self.obs_space[2]
                            obs_side = np.concatenate(
                                (self.obs_space[:2], self.obs_space[3:4]))
                            l0_obs_func[
                                act_joint_lin,
                                l0_state_lin,
                                obs_ml] = self.pr_obs_dir_corr
                            l0_obs_func[
                                act_joint_lin,
                                l0_state_lin,
                                obs_side] = (1 - self.pr_obs_dir_corr - 0.01) / obs_side.shape[0]

                            if y < gold_y:  # gold cell is in the southeast
                                obs_ml = self.obs_space[1:3]
                                obs_side = np.array(
                                    [self.obs_space[0], self.obs_space[3]])
                                l0_obs_func[
                                    act_joint_lin,
                                    l0_state_lin,
                                    obs_ml] = (1 - 0.01 - l0_obs_func[
                                    act_joint_lin,
                                    l0_state_lin,
                                    obs_side] * obs_side.shape[0]) / obs_ml.shape[0]

                            elif y > gold_y:  # gold cell is in the southwest
                                obs_ml = self.obs_space[2:4]
                                obs_side = self.obs_space[:2]
                                l0_obs_func[
                                    act_joint_lin,
                                    l0_state_lin,
                                    obs_ml] = (1 - 0.01 - l0_obs_func[
                                    act_joint_lin,
                                    l0_state_lin,
                                    obs_side] * obs_side.shape[0]) / obs_ml.shape[0]

                        elif y < gold_y:  # gold cell is in the east
                            obs_ml = self.obs_space[1]
                            obs_side = np.concatenate(
                                (self.obs_space[:1], self.obs_space[2:4]))
                            l0_obs_func[
                                act_joint_lin,
                                l0_state_lin,
                                obs_ml] = self.pr_obs_dir_corr
                            l0_obs_func[
                                act_joint_lin,
                                l0_state_lin,
                                obs_side] = (1 - self.pr_obs_dir_corr - 0.01) / obs_side.shape[0]

                        elif y > gold_y:  # gold cell is in the west
                            obs_ml = self.obs_space[3]
                            obs_side = self.obs_space[:3]
                            l0_obs_func[
                                act_joint_lin,
                                l0_state_lin,
                                obs_ml] = self.pr_obs_dir_corr
                            l0_obs_func[
                                act_joint_lin,
                                l0_state_lin,
                                obs_side] = (1 - self.pr_obs_dir_corr - 0.01) / obs_side.shape[0]
                        else:
                            assert False, "(%d, %d)" % (x, y)

                    for cell_other in self.agent_loc_space:
                        inter_state = np.array(
                            [cell_self, cell_other, gold_cell])
                        inter_state_lin = joint_to_lin(
                            inter_state,
                            (self.agent_loc_space,
                             self.agent_loc_space,
                             self.door_loc_space))
                        # When the agent is located in either the gold cell
                        # or the tiger cell, its observation is 100% "door-
                        # found".
                        if cell_self in [gold_cell, tiger_cell]:
                            obs = self.obs_space[-1]
                            inter_obs_func[
                                act_joint_lin,
                                inter_state_lin,
                                obs] = self.pr_obs_door_corr

                        else:
                            # Mistakenly hear "door-found" at a non-door cell.
                            obs_wrong_door = self.obs_space[-1]
                            inter_obs_func[
                                act_joint_lin,
                                inter_state_lin,
                                obs_wrong_door] = 0.01
                            x, y = self.cell_lin_to_bin(cell_self)
                            if x > gold_x:  # gold cell is to the north
                                obs_ml = self.obs_space[0]
                                obs_side = self.obs_space[1:4]
                                inter_obs_func[
                                    act_joint_lin,
                                    inter_state_lin,
                                    obs_ml] = self.pr_obs_dir_corr
                                inter_obs_func[
                                    act_joint_lin,
                                    inter_state_lin,
                                    obs_side] = (1 - self.pr_obs_dir_corr - 0.01) / obs_side.shape[0]

                                if y < gold_y:  # gold cell is in the northeast
                                    obs_ml = self.obs_space[:2]
                                    obs_side = self.obs_space[2:4]
                                    inter_obs_func[
                                        act_joint_lin,
                                        inter_state_lin,
                                        obs_ml] = (1 - 0.01 - inter_obs_func[
                                        act_joint_lin,
                                        inter_state_lin,
                                        obs_side] * obs_side.shape[0]) / obs_ml.shape[0]
                                elif y > gold_y:  # gold cell is in the northwest
                                    obs_ml = np.array(
                                        [self.obs_space[0], self.obs_space[3]])
                                    obs_side = self.obs_space[1:3]
                                    inter_obs_func[
                                        act_joint_lin,
                                        inter_state_lin,
                                        obs_ml] = (1 - 0.01 - inter_obs_func[
                                        act_joint_lin,
                                        inter_state_lin,
                                        obs_side] * obs_side.shape[0]) / obs_ml.shape[0]

                            elif x < gold_x:  # gold cell is in the south
                                obs_ml = self.obs_space[2]
                                obs_side = np.concatenate(
                                    (self.obs_space[:2], self.obs_space[3:4]))
                                inter_obs_func[
                                    act_joint_lin,
                                    inter_state_lin,
                                    obs_ml] = self.pr_obs_dir_corr
                                inter_obs_func[
                                    act_joint_lin,
                                    inter_state_lin,
                                    obs_side] = (1 - self.pr_obs_dir_corr - 0.01) / obs_side.shape[0]

                                if y < gold_y:  # gold cell is in the southeast
                                    obs_ml = self.obs_space[1:3]
                                    obs_side = np.array(
                                        [self.obs_space[0], self.obs_space[3]])
                                    inter_obs_func[
                                        act_joint_lin,
                                        inter_state_lin,
                                        obs_ml] = (1 - 0.01 - inter_obs_func[
                                        act_joint_lin,
                                        inter_state_lin,
                                        obs_side] * obs_side.shape[0]) / obs_ml.shape[0]

                                elif y > gold_y:  # gold cell is in the southwest
                                    obs_ml = self.obs_space[2:4]
                                    obs_side = self.obs_space[:2]
                                    inter_obs_func[
                                        act_joint_lin,
                                        inter_state_lin,
                                        obs_ml] = (1 - 0.01 - inter_obs_func[
                                        act_joint_lin,
                                        inter_state_lin,
                                        obs_side] * obs_side.shape[0]) / obs_ml.shape[0]

                            elif y < gold_y:  # gold cell is in the east
                                obs_ml = self.obs_space[1]
                                obs_side = np.concatenate(
                                    (self.obs_space[:1], self.obs_space[2:4]))
                                inter_obs_func[act_joint_lin,
                                               inter_state_lin,
                                               obs_ml] = self.pr_obs_dir_corr
                                inter_obs_func[
                                    act_joint_lin,
                                    inter_state_lin,
                                    obs_side] = (1 - self.pr_obs_dir_corr - 0.01) / obs_side.shape[0]

                            elif y > gold_y:  # gold cell is in the west
                                obs_ml = self.obs_space[3]
                                obs_side = self.obs_space[:3]
                                inter_obs_func[
                                    act_joint_lin,
                                    inter_state_lin,
                                    obs_ml] = self.pr_obs_dir_corr
                                inter_obs_func[
                                    act_joint_lin,
                                    inter_state_lin,
                                    obs_side] = (1 - self.pr_obs_dir_corr - 0.01) / obs_side.shape[0]
                            else:
                                assert False, "(%d, %d)" % (x, y)

            # 2. action <- move-X or door-open
            arr_act_self = self.action_space[1:]
            arr_act_joint = [[a, act_other] for a in arr_act_self]
            arr_act_joint_lin = [joint_to_lin(
                aj, (self.action_space, self.action_space))
                for aj in arr_act_joint]
            for gold_cell in self.door_loc_space:
                for cell_self in self.agent_loc_space:
                    l0_state = np.array([cell_self, gold_cell])
                    l0_state_lin = joint_to_lin(
                        l0_state, (self.agent_loc_space, self.door_loc_space))
                    obs_door = self.obs_space[-1]
                    l0_obs_func[arr_act_joint_lin,
                                l0_state_lin,
                                obs_door] = 0.01
                    for obs in self.obs_space[:4]:
                        l0_obs_func[
                            arr_act_joint_lin,
                            l0_state_lin,
                            obs] = (1 - 0.01) / self.obs_space[:4].shape[0]

                    for cell_other in self.agent_loc_space:
                        inter_state = np.array(
                            [cell_self, cell_other, gold_cell])
                        inter_state_lin = joint_to_lin(
                            inter_state,
                            (self.agent_loc_space,
                             self.agent_loc_space,
                             self.door_loc_space))
                        inter_obs_func[arr_act_joint_lin,
                                       inter_state_lin,
                                       obs_door] = 0.01
                        for obs in self.obs_space[:4]:
                            inter_obs_func[
                                arr_act_joint_lin,
                                inter_state_lin,
                                obs] = (1 - 0.01) / self.obs_space[:4].shape[0]

        # for a in range(l0_obs_func.shape[0]):
        #     for s in range(l0_obs_func.shape[1]):
        #         if l0_obs_func[a, s, :].sum() != 1:
        #             print("LEVEL-0")
        #             print("A: %d, S:%d" % (a, s))
        #             print(l0_obs_func[a, s, :])
        #             print(l0_obs_func[a, s, :].sum())
        #
        # for a in range(inter_obs_func.shape[0]):
        #     for s in range(inter_obs_func.shape[1]):
        #         if inter_obs_func[a, s, :].sum() != 1:
        #             print("INTERACTIVE")
        #             print("A: %d, S:%d" % (a, s))
        #             print(inter_obs_func[a, s, :])
        #             print(inter_obs_func[a, s, :].sum())

        if info_present is "export":
            with open("./data/O.txt", mode='w') as f:
                f.write("LEVEL-0 OBSERVATION FUNCTION")
                for a in range(len(l0_obs_func)):
                    ai, aj = lin_to_joint(a, (
                        self.action_space, self.action_space))
                    f.write("\nACTION i: %d, ACTION j: %d\n" % (ai, aj))
                    f.write(str(l0_obs_func[a]))
                f.write("-" * 10)
                f.write("INTERACTIVE OBSERVATION FUNCTION")
                for a in range(len(inter_obs_func)):
                    ai, aj = lin_to_joint(
                        a, (self.action_space, self.action_space))
                    f.write("\nACTION i: %d, ACTION j: %d\n" % (ai, aj))
                    f.write(str(inter_obs_func[a]))
        elif info_present is "console":
            print("-" * 20)
            print("LEVEL-0 OBSERVATION FUNCTION")
            for a in range(len(l0_obs_func)):
                ai, aj = lin_to_joint(a, (self.action_space, self.action_space))
                print("ACTION i: %d, ACTION j: %d" % (ai, aj))
                print(l0_obs_func[a])
            print("-" * 10)
            print("INTERACTIVE OBSERVATION FUNCTION")
            for a in range(len(inter_obs_func)):
                ai, aj = lin_to_joint(a, (self.action_space, self.action_space))
                print("ACTION i: %d, ACTION j: %d" % (ai, aj))
                print(inter_obs_func[a])
        else:
            pass

        return l0_obs_func, inter_obs_func

    def build_trans_reward_func(self, info_present=None):
        """
        Build transition (trans_func) and reward (reward_func) model for a grid.
        :param info_present:
        :return:
        """
        num_agent_birthplace = self.agent_birth_space.shape[0]
        num_door_birthplace = self.door_loc_space.shape[0]
        params = self.params

        pr_trans_intent = params.pr_trans_intent
        pr_trans_side = (1 - pr_trans_intent) / 2

        reward_gold = params.reward_gold
        reward_tiger = params.reward_tiger
        reward_listen = params.reward_listen
        reward_wrong_open = params.reward_wrong_open
        reward_move = params.reward_move

        # Initialize matrices for transition and reward functions.
        # trans_func_l0 -- matrix for the level-0 transition function.
        # Shape: (A, S, S') where S = S' = N_a_i * N_g, A = A_i * A_j
        # DataType: float within [0., 1.]
        trans_func_l0 = [scipy.sparse.lil_matrix(
            (self.num_l0_phys_state, self.num_l0_phys_state), dtype='f')
            for _ in range(self.num_joint_action)]
        # reward_func_l0 -- matrix for the level-0 reward function.
        # shape: (A, S, S') where S = S' = N_a_i * N_g, A = A_i * A_j
        # DataType: float, any real value.
        reward_func_l0 = [scipy.sparse.lil_matrix(
            (self.num_l0_phys_state, self.num_l0_phys_state), dtype='f')
            for _ in range(self.num_joint_action)]
        # trans_func_inter -- matrix for the interactive transition function.
        # Shape: (A, S, S') where S = S' = N_a_i * N_a_j * N_g. A = A_i * A_j
        # DataType: float within [0., 1.]
        trans_func_inter = [scipy.sparse.lil_matrix(
            (self.num_inter_phys_state, self.num_inter_phys_state), dtype='f')
            for _ in range(self.num_joint_action)]
        # reward_func_inter -- matrix for the interactive reward function.
        # shape: (A, S, S') where S = S' = N_a_i * N_a_j * N_g, A = A_i * A_j
        # DataType: float, any real value.
        reward_func_inter = [scipy.sparse.lil_matrix(
            (self.num_inter_phys_state, self.num_inter_phys_state), dtype='f')
            for _ in range(self.num_joint_action)]

        # Assign values for the matrices in the order of A -> S -> S'
        # Order of the actions in the action space:
        # 1) STAY & LISTEN
        # 2) MOVE NORTH
        # 3) MOVE EAST
        # 4) MOVE SOUTH
        # 5) MOVE WEST
        # 6) OPEN THE DOOR
        # --------
        # 1. action <- STAY & LISTEN
        # When the subjective agent stays and listens for gaining observation,
        # it determinately stays at the previous state after applying the
        # action, which means that the probability for it to transit from one
        # state to the same state is 1, while for the transitions to all other
        # states are 0s.
        act_self = self.action_space[0]
        for act_other in self.action_space:
            act_joint = np.array([act_self, act_other])
            act_joint_lin = joint_to_lin(
                act_joint, (self.action_space, self.action_space))
            for gold_cell in self.door_loc_space:
                for cell_self in self.agent_loc_space:
                    l0_state = np.array([cell_self, gold_cell])
                    l0_state_lin = joint_to_lin(
                        l0_state, (self.agent_loc_space, self.door_loc_space))
                    l0_state_next_lin = l0_state_lin
                    trans_func_l0[
                        act_joint_lin][
                        l0_state_lin,
                        l0_state_next_lin] = 1
                    reward_func_l0[
                        act_joint_lin][l0_state_lin, :] = reward_listen

        # 2. action <- [MOVE-N, MOVE-E, MOVE-S, MOVE-W]
        #
        for act_self in self.action_space[1:-1]:
            for act_other in self.action_space:
                act_joint = np.array([act_self, act_other])
                act_joint_lin = joint_to_lin(
                    act_joint, (self.action_space, self.action_space))
                for gold_cell in self.door_loc_space:
                    for cell_self in self.agent_loc_space:
                        l0_state = np.array([cell_self, gold_cell])
                        l0_state_lin = joint_to_lin(
                            l0_state,
                            (self.agent_loc_space,
                             self.door_loc_space))
                        # Get the intentonal and side next cells respectively.
                        cell_self_intent, cell_self_side = \
                            self.get_intent_side_cells(
                                move_action=act_self,
                                cell=cell_self)
                        l0_state_intent = np.array(
                            [cell_self_intent, gold_cell])
                        l0_state_intent_lin = joint_to_lin(
                            l0_state_intent,
                            (self.agent_loc_space,
                              self.door_loc_space))
                        trans_func_l0[
                            act_joint_lin][
                            l0_state_lin,
                            l0_state_intent_lin] += pr_trans_intent

                        for cell in cell_self_side:
                            l0_state_side = np.array([cell, gold_cell])
                            l0_state_side_lin = joint_to_lin(
                                l0_state_side,
                                (self.agent_loc_space,
                                 self.door_loc_space))
                            trans_func_l0[
                                act_joint_lin][
                                l0_state_lin,
                                l0_state_side_lin] += pr_trans_side

                        reward_func_l0[
                            act_joint_lin][l0_state_lin, :] = reward_move

        # 3. action <- open the door
        #
        act_self = self.action_space[-1]
        for act_other in self.action_space:
            act_joint = [act_self, act_other]
            act_joint_lin = joint_to_lin(
                act_joint, (self.action_space, self.action_space))
            for gold_cell in self.door_loc_space:
                # Inferring the cell where the tiger occupies according to
                # the gold-cell.
                idx_gold_in_space = np.argwhere(
                    gold_cell == self.door_loc_space)[0][0]
                idx_tiger_in_space = \
                    num_door_birthplace - 1 - idx_gold_in_space
                tiger_cell = self.door_loc_space[
                    idx_tiger_in_space]

                for cell_self in self.agent_loc_space:
                    l0_state = np.array([cell_self, gold_cell])
                    l0_state_lin = joint_to_lin(
                        l0_state, (self.agent_loc_space, self.door_loc_space))

                    # Given that the subjective agent applies the door-opening
                    # action, if it occupies either gold or tiger cell for now,
                    # the environment will reset, where the two agents are
                    # re-located in random cell(s) of the spawning space. On the
                    # other hand, the location of the two doors remain fixed,
                    # while the gold and the tiger appear randomly behind them,
                    # which means they may either maintain or swap their
                    # locations after the reset. Hence, the probability of
                    # a transition from a state before the reset to one after
                    # should be equal to [1 / (agent_birthplace_size)] * (1/2).
                    #
                    # 1. When the agent is in the same cell as the gold or tiger
                    if cell_self in [gold_cell, tiger_cell]:
                        for cell_self_next in self.agent_birth_space:
                            # 1) The gold and the tiger remain in the same cell
                            # after the reset.
                            l0_state_next = [cell_self_next, gold_cell]
                            l0_state_next_lin = joint_to_lin(
                                l0_state_next,
                                (self.agent_loc_space,
                                 self.door_loc_space))
                            trans_func_l0[
                                act_joint_lin][
                                l0_state_lin,
                                l0_state_next_lin] = 1 / (num_agent_birthplace * 2)

                            # 2) The gold and the tiger swap their locations
                            # after the reset.
                            l0_state_next = [cell_self_next, tiger_cell]
                            l0_state_next_lin = joint_to_lin(
                                l0_state_next,
                                (self.agent_loc_space,
                                    self.door_loc_space))
                            trans_func_l0[
                                act_joint_lin][
                                l0_state_lin,
                                l0_state_next_lin] = 1 / (num_agent_birthplace * 2)
                        # Fill in the level-0 reward function matrix.
                        # If the agent is in the gold cell, then it will receive
                        # a reward of finding the gold after opening the door.
                        # Otherwise, the agent will get a punishment due to
                        # encountering the tiger after opening the door.
                        if cell_self == gold_cell:
                            reward_func_l0[
                                act_joint_lin][l0_state_lin, :] = reward_gold
                        else:
                            reward_func_l0[
                                act_joint_lin][l0_state_lin, :] = reward_tiger

                    # 2. When the agent is located in neither tiger cell nor
                    # gold cell, opening the door will not lead to any state
                    # change, but this will cost -10 in the reward.
                    else:
                        l0_state_next_lin = l0_state_lin
                        trans_func_l0[
                            act_joint_lin][
                            l0_state_lin,
                            l0_state_next_lin] = 1
                        reward_func_l0[
                            act_joint_lin][
                        l0_state_lin, :] = reward_wrong_open

        # 1.
        act_self = self.action_space[0]
        for act_other in self.action_space:
            act_joint = np.array([act_self, act_other])
            act_joint_lin = joint_to_lin(
                act_joint, (self.action_space, self.action_space))
            for gold_cell in self.door_loc_space:
                idx_gold_in_space = np.argwhere(
                    gold_cell == self.door_loc_space)[0][0]
                idx_tiger_in_space = \
                    num_door_birthplace - 1 - idx_gold_in_space
                tiger_cell = self.door_loc_space[idx_tiger_in_space]

                for cell_self in self.agent_loc_space:
                    cell_self_next = cell_self
                    for cell_other in self.agent_loc_space:
                        inter_state = np.array(
                            [cell_self, cell_other, gold_cell])
                        inter_state_lin = joint_to_lin(
                            inter_state,
                            (self.agent_loc_space,
                             self.agent_loc_space,
                             self.door_loc_space))
                        reward_func_inter[
                            act_joint_lin][inter_state_lin, :] = reward_listen
                        if act_other == self.action_space[0]:
                            inter_state_next_lin = inter_state_lin
                            trans_func_inter[
                                act_joint_lin][
                                inter_state_lin,
                                inter_state_next_lin] = 1
                        elif act_other in self.action_space[1:-1]:
                            cell_other_intent, cell_other_side = \
                                self.get_intent_side_cells(
                                    move_action=act_other,
                                    cell=cell_other)
                            inter_state_intent = np.array(
                                [cell_self_next,
                                 cell_other_intent,
                                 gold_cell])
                            inter_state_intent_lin = joint_to_lin(
                                inter_state_intent,
                                (self.agent_loc_space,
                                 self.agent_loc_space,
                                 self.door_loc_space))
                            trans_func_inter[
                                act_joint_lin][
                                inter_state_lin,
                                inter_state_intent_lin] += pr_trans_intent
                            for cell in cell_other_side:
                                inter_state_side = np.array(
                                    [cell_self_next, cell, gold_cell])
                                inter_state_side_lin = joint_to_lin(
                                    inter_state_side,
                                    (self.agent_loc_space,
                                     self.agent_loc_space,
                                     self.door_loc_space))
                                trans_func_inter[
                                    act_joint_lin][
                                    inter_state_lin,
                                    inter_state_side_lin] += pr_trans_side
                        else:
                            if cell_other in [gold_cell, tiger_cell]:
                                for cell_self_next in self.agent_birth_space:
                                    for cell_other_next in self.agent_birth_space:
                                        # There is a 0.5 chance that the gold
                                        # cell remains in the same location
                                        # after the reset.
                                        inter_state_next = np.array(
                                            [cell_self_next,
                                             cell_other_next,
                                             gold_cell])
                                        inter_state_next_lin = joint_to_lin(
                                            inter_state_next,
                                            (self.agent_loc_space,
                                             self.agent_loc_space,
                                             self.door_loc_space))
                                        trans_func_inter[
                                            act_joint_lin][
                                            inter_state_lin,
                                            inter_state_next_lin] += \
                                            1 / (num_agent_birthplace *
                                                 num_agent_birthplace * 2)
                                        # There is another 0.5 chance that the
                                        # gold and tiger cell will swap their
                                        # locations after the reset.
                                        inter_state_next = np.array(
                                            [cell_self_next,
                                             cell_other_next,
                                             tiger_cell])
                                        inter_state_next_lin = joint_to_lin(
                                            inter_state_next,
                                            (self.agent_loc_space,
                                             self.agent_loc_space,
                                             self.door_loc_space))
                                        trans_func_inter[
                                            act_joint_lin][
                                            inter_state_lin,
                                            inter_state_next_lin] += \
                                            1 / (num_agent_birthplace *
                                                 num_agent_birthplace * 2)
                            else:
                                inter_state_next_lin = inter_state_lin
                                trans_func_inter[
                                    act_joint_lin][
                                    inter_state_lin,
                                    inter_state_next_lin] = 1

        # 2.
        for act_self in self.action_space[1:-1]:
            for act_other in self.action_space:
                act_joint = np.array([act_self, act_other])
                act_joint_lin = joint_to_lin(
                    act_joint, (self.action_space, self.action_space))
                for gold_cell in self.door_loc_space:
                    idx_gold_in_space = np.argwhere(
                        gold_cell == self.door_loc_space)[0][0]
                    idx_tiger_in_space = \
                        num_door_birthplace - 1 - idx_gold_in_space
                    tiger_cell = self.door_loc_space[
                        idx_tiger_in_space]

                    for cell_self in self.agent_loc_space:
                        l0_state_self = np.array([cell_self, gold_cell])
                        l0_state_self_lin = joint_to_lin(
                            l0_state_self,
                            (self.agent_loc_space,
                             self.door_loc_space))
                        # Get the intentional and side next cells respectively.
                        cell_self_intent, cell_self_side = \
                            self.get_intent_side_cells(
                                move_action=act_self,
                                cell=cell_self)
                        # Get the level-0 state for the subjective agent when
                        # it moves along intentional directions.
                        l0_state_intent_self = np.array(
                            [cell_self_intent, gold_cell])
                        l0_state_intent_self_lin = joint_to_lin(
                            l0_state_intent_self,
                            (self.agent_loc_space,
                             self.door_loc_space))
                        pr_intent_self = trans_func_l0[
                            act_joint_lin][
                            l0_state_self_lin,
                            l0_state_intent_self_lin]
                        # Collect the level-0 state for the subjective agent
                        # when it moves along side directions.
                        l0_state_side_self_lin = list()
                        pr_side_self = list()
                        for cell in cell_self_side:
                            l0_state_side = np.array([cell, gold_cell])
                            l0_state_side_lin = joint_to_lin(
                                l0_state_side,
                                (self.agent_loc_space,
                                 self.door_loc_space))
                            l0_state_side_self_lin.append(l0_state_side_lin)
                            pr_side_self.append(
                                trans_func_l0[
                                    act_joint_lin][
                                    l0_state_self_lin,
                                    l0_state_side_lin])

                        for cell_other in self.agent_loc_space:
                            l0_state_other = np.array(
                                [cell_other, gold_cell])
                            l0_state_other_lin = joint_to_lin(
                                l0_state_other,
                                (self.agent_loc_space, self.door_loc_space))

                            inter_state = np.array(
                                [cell_self, cell_other, gold_cell])
                            inter_state_lin = joint_to_lin(
                                inter_state,
                                (self.agent_loc_space,
                                 self.agent_loc_space,
                                 self.door_loc_space))
                            reward_func_inter[
                                act_joint_lin][
                                inter_state_lin, :] = reward_move

                            # 1. The objective agent chooses to stay and to gain
                            # additional observation.
                            if act_other == self.action_space[0]:
                                cell_other_next = cell_other
                                # When the subjective agent moves towards the
                                # intentional direction and reaches its expected
                                # next cell.
                                inter_state_intent = np.array(
                                    [cell_self_intent,
                                     cell_other_next,
                                     gold_cell])
                                inter_state_intent_lin = joint_to_lin(
                                    inter_state_intent,
                                    (self.agent_loc_space,
                                     self.agent_loc_space,
                                     self.door_loc_space))
                                trans_func_inter[
                                    act_joint_lin][
                                    inter_state_lin,
                                    inter_state_intent_lin] += pr_trans_intent
                                # When the subjective agent moves towards either
                                # orthogonal direction of the intentional one
                                # and it ends up getting into a neighbor cell.
                                for cell in cell_self_side:
                                    inter_state_side = np.array(
                                        [cell, cell_other_next, gold_cell])
                                    inter_state_side_lin = joint_to_lin(
                                        inter_state_side,
                                        (self.agent_loc_space,
                                         self.agent_loc_space,
                                         self.door_loc_space))
                                    trans_func_inter[
                                        act_joint_lin][
                                        inter_state_lin,
                                        inter_state_side_lin] += pr_trans_side

                            # 2. The objective agent chooses to move along any
                            # cardinal direction.
                            elif act_other in self.action_space[1:-1]:
                                cell_other_intent, cell_other_side = \
                                    self.get_intent_side_cells(
                                        move_action=act_other,
                                        cell=cell_other)

                                pr_inter_state_next = dict()
                                cell_self_next_move = np.append(
                                    cell_self_intent, cell_self_side)
                                cell_other_next_move = np.append(
                                    cell_other_intent, cell_other_side)
                                pr_list = [pr_trans_intent, pr_trans_side, pr_trans_side]
                                for i in range(len(cell_self_next_move)):
                                    for j in range(len(cell_other_next_move)):
                                        inter_state_next = np.array(
                                            [cell_self_next_move[i],
                                             cell_other_next_move[j],
                                             gold_cell])
                                        inter_state_next_lin = joint_to_lin(
                                            inter_state_next,
                                            (self.agent_loc_space,
                                             self.agent_loc_space,
                                             self.door_loc_space))
                                        if inter_state_next_lin not in pr_inter_state_next:
                                            pr_inter_state_next[
                                                inter_state_next_lin] = pr_list[i] * pr_list[j]
                                        else:
                                            pr_inter_state_next[
                                                inter_state_next_lin] += pr_list[i] * pr_list[j]

                                for inter_state_next_lin in pr_inter_state_next:
                                    trans_func_inter[
                                        act_joint_lin][
                                        inter_state_lin,
                                        inter_state_next_lin] = pr_inter_state_next[inter_state_next_lin]

                            # 3. The objective agent chooses to open the door at
                            # its current cell.
                            else:
                                # 1) The objective agent is in a door-cell,
                                # which means the no matter the subjective agent
                                # moves along which direction, the environment
                                # is sure to reset.
                                if cell_other in [gold_cell, tiger_cell]:
                                    for cell_self_next in self.agent_birth_space:
                                        for cell_other_next in self.agent_birth_space:
                                            # Case 1:
                                            # The gold cell remains in the same
                                            # location after the reset.
                                            inter_state_next = np.array(
                                                [cell_self_next,
                                                 cell_other_next,
                                                 gold_cell])
                                            inter_state_next_lin = joint_to_lin(
                                                inter_state_next,
                                                (self.agent_loc_space,
                                                 self.agent_loc_space,
                                                 self.door_loc_space))
                                            trans_func_inter[
                                                act_joint_lin][
                                                inter_state_lin,
                                                inter_state_next_lin] += \
                                                1 / (num_agent_birthplace *
                                                     num_agent_birthplace * 2)
                                            # Case 2:
                                            # The gold and tiger cell swap
                                            # their locations after the reset.
                                            inter_state_next = np.array(
                                                [cell_self_next,
                                                 cell_other_next, tiger_cell])
                                            inter_state_next_lin = joint_to_lin(
                                                inter_state_next,
                                                (self.agent_loc_space,
                                                 self.agent_loc_space,
                                                 self.door_loc_space))
                                            trans_func_inter[
                                                act_joint_lin][
                                                inter_state_lin,
                                                inter_state_next_lin] += \
                                                1 / (num_agent_birthplace *
                                                     num_agent_birthplace * 2)

                                # 2) The objective agent is not in a door-cell,
                                # so its door-opening action has no impact to
                                # the current environment.
                                else:
                                    cell_other_next = cell_other

                                    # Case 1:
                                    # 1. The objective agent moves along
                                    # the intentional orientation.
                                    inter_state_intent = np.array(
                                        [cell_self_intent,
                                         cell_other_next,
                                         gold_cell])
                                    inter_state_intent_lin = joint_to_lin(
                                        inter_state_intent,
                                        (self.agent_loc_space,
                                         self.agent_loc_space,
                                         self.door_loc_space))
                                    trans_func_inter[
                                        act_joint_lin][
                                        inter_state_lin,
                                        inter_state_intent_lin] += pr_trans_intent
                                    # Case 2:
                                    # The objective agent moves towards
                                    # either side direction.
                                    for cell in cell_self_side:
                                        inter_state_side = np.array(
                                            [cell, cell_other_next, gold_cell])
                                        inter_state_side_lin = joint_to_lin(
                                            inter_state_side,
                                            (self.agent_loc_space,
                                             self.agent_loc_space,
                                             self.door_loc_space))
                                        trans_func_inter[
                                            act_joint_lin][
                                            inter_state_lin,
                                            inter_state_side_lin] += pr_trans_side

        # 3.
        act_self = self.action_space[-1]
        for act_other in self.action_space:
            act_joint = [act_self, act_other]
            act_joint_lin = joint_to_lin(
                act_joint, (self.action_space, self.action_space))
            for gold_cell in self.door_loc_space:
                # Inferring the cell where the tiger occupies according to
                # the gold-cell.
                idx_gold_in_space = np.argwhere(
                    gold_cell == self.door_loc_space)[0][0]
                idx_tiger_in_space = \
                    num_door_birthplace - 1 - idx_gold_in_space
                tiger_cell = self.door_loc_space[
                    idx_tiger_in_space]

                for cell_self in self.agent_loc_space:
                    for cell_other in self.agent_loc_space:
                        inter_state = np.array(
                            [cell_self, cell_other, gold_cell])
                        inter_state_lin = joint_to_lin(
                            inter_state,
                            (self.agent_loc_space,
                             self.agent_loc_space,
                             self.door_loc_space))

                        # Case 1:
                        # The subjective agent is in a door-cell. Once it opens
                        # the door, the environment deterministically resets.
                        if cell_self in [gold_cell, tiger_cell]:
                            if cell_self == gold_cell:
                                reward_func_inter[
                                    act_joint_lin][
                                    inter_state_lin, :] = reward_gold
                            else:
                                reward_func_inter[
                                    act_joint_lin][
                                    inter_state_lin, :] = reward_tiger

                            for cell_self_next in self.agent_birth_space:
                                for cell_other_next in self.agent_birth_space:
                                    # i) The gold and the tiger remain in their
                                    # original cells.
                                    inter_state_next = np.array(
                                        [cell_self_next,
                                         cell_other_next,
                                         gold_cell])
                                    inter_state_next_lin = joint_to_lin(
                                        inter_state_next,
                                        (self.agent_loc_space,
                                         self.agent_loc_space,
                                         self.door_loc_space))
                                    trans_func_inter[
                                        act_joint_lin][
                                        inter_state_lin,
                                        inter_state_next_lin] += \
                                        1 / (num_agent_birthplace *
                                             num_agent_birthplace * 2)
                                    # ii) The gold and tiger cell swap
                                    # their locations after the reset.
                                    inter_state_next = np.array(
                                        [cell_self_next,
                                         cell_other_next,
                                         tiger_cell])
                                    inter_state_next_lin = joint_to_lin(
                                        inter_state_next,
                                        (self.agent_loc_space,
                                         self.agent_loc_space,
                                         self.door_loc_space))
                                    trans_func_inter[
                                        act_joint_lin][
                                        inter_state_lin,
                                        inter_state_next_lin] += \
                                        1 / (num_agent_birthplace *
                                             num_agent_birthplace * 2)

                        # Case 2:
                        # The subjective agent is not in a door-cell, but it
                        # tries to open the door. Therefore, only the objective
                        # agent may impact the environment.
                        else:
                            reward_func_inter[
                                act_joint_lin][
                                inter_state_lin, :] = reward_wrong_open

                            if act_other == self.action_space[0]:
                                inter_state_next_lin = inter_state_lin
                                trans_func_inter[
                                    act_joint_lin][
                                    inter_state_lin, inter_state_next_lin] = 1

                            elif act_other in self.action_space[1:-1]:
                                cell_self_next = cell_self
                                cell_other_intent, cell_other_side = \
                                    self.get_intent_side_cells(
                                        move_action=act_other,
                                        cell=cell_other)
                                # i) The objective agent moves along
                                # the intentional orientation.
                                inter_state_intent = np.array(
                                    [cell_self_next,
                                     cell_other_intent,
                                     gold_cell])
                                inter_state_intent_lin = joint_to_lin(
                                    inter_state_intent,
                                    (self.agent_loc_space,
                                     self.agent_loc_space,
                                     self.door_loc_space))
                                trans_func_inter[
                                    act_joint_lin][
                                    inter_state_lin,
                                    inter_state_intent_lin] += pr_trans_intent
                                # ii) The objective agent moves towards
                                # either side direction.
                                for cell in cell_other_side:
                                    inter_state_side = np.array(
                                        [cell_self_next, cell, gold_cell])
                                    inter_state_side_lin = joint_to_lin(
                                        inter_state_side,
                                        (self.agent_loc_space,
                                         self.agent_loc_space,
                                         self.door_loc_space))
                                    trans_func_inter[
                                        act_joint_lin][
                                        inter_state_lin,
                                        inter_state_side_lin] += pr_trans_side

                            else:
                                if cell_other in [gold_cell, tiger_cell]:
                                    for cell_self_next in self.agent_birth_space:
                                        for cell_other_next in self.agent_birth_space:
                                            # i) The gold and the tiger remain
                                            # in their original cells.
                                            inter_state_next = np.array(
                                                [cell_self_next,
                                                 cell_other_next,
                                                 gold_cell])
                                            inter_state_next_lin = joint_to_lin(
                                                inter_state_next,
                                                (self.agent_loc_space,
                                                 self.agent_loc_space,
                                                 self.door_loc_space))
                                            trans_func_inter[
                                                act_joint_lin][
                                                inter_state_lin,
                                                inter_state_next_lin] += \
                                                1 / (num_agent_birthplace *
                                                     num_agent_birthplace * 2)
                                            # ii) The gold and tiger cell swap
                                            # their locations after the reset.
                                            inter_state_next = np.array(
                                                [cell_self_next,
                                                 cell_other_next,
                                                 tiger_cell])
                                            inter_state_next_lin = joint_to_lin(
                                                inter_state_next,
                                                (self.agent_loc_space,
                                                 self.agent_loc_space,
                                                 self.door_loc_space))
                                            trans_func_inter[
                                                act_joint_lin][
                                                inter_state_lin,
                                                inter_state_next_lin] += \
                                                1 / (num_agent_birthplace *
                                                     num_agent_birthplace * 2)
                                else:
                                    inter_state_next_lin = inter_state_lin
                                    trans_func_inter[
                                        act_joint_lin][
                                        inter_state_lin, inter_state_next_lin] = 1

        # for a in np.arange(self.num_joint_action):
        #     for s in np.arange(self.num_l0_phys_state):
        #         if trans_func_l0[a][s, :].sum() != 1:
        #             print("CASES IN LEVEL-0 TRANSITION FUNCTION:")
        #             print(trans_func_inter[a][s, :])
        #             print(trans_func_inter[a][s, :].sum())

        # for a in np.arange(self.num_joint_action):
        #     for s in np.arange(self.num_inter_phys_state):
        #         if trans_func_inter[a][s, :].sum() != 1:
        #             s_ = lin_to_joint(a, (self.agent_loc_space, self.agent_loc_space, self.door_loc_space))
        #             print("CURRENT STATE", s_)
        #             ai, aj = lin_to_joint(a, (self.action_space, self.action_space))
        #             print("ACTION: i: %d, j:%d" % (ai, aj))
        #             print("CASES IN INTERACTIVE TRANSITION FUNCTION:")
        #             print(trans_func_inter[a][s, :])
        #             print(trans_func_inter[a][s, :].sum())

        if info_present is "export":
            with open("./data/TR.txt", mode='w') as f:
                f.write("LEVEL-0 TRANSITION FUNCTION")
                for a in range(len(trans_func_l0)):
                    ai, aj = lin_to_joint(a, (
                    self.action_space, self.action_space))
                    f.write("\nACTION i: %d, ACTION j: %d\n" % (ai, aj))
                    f.write(str(trans_func_l0[a]))
                f.write("-" * 10)
                f.write("LEVEL-0 REWARD FUNCTION")
                for a in range(len(reward_func_l0)):
                    ai, aj = lin_to_joint(
                        a, (self.action_space, self.action_space))
                    f.write("\nACTION i: %d, ACTION j: %d\n" % (ai, aj))
                    f.write(str(reward_func_l0[a]))
                f.write("-" * 10)
                f.write("INTERACTIVE TRANSITION FUNCTION")
                for a in range(len(trans_func_inter)):
                    ai, aj = lin_to_joint(
                        a, (self.action_space, self.action_space))
                    f.write("\nACTION i: %d, ACTION j: %d\n" % (ai, aj))
                    f.write(str(trans_func_inter[a]))
                f.write("-" * 10)
                f.write("INTERACTIVE REWARD FUNCTION")
                for a in range(len(reward_func_inter)):
                    ai, aj = lin_to_joint(
                        a, (self.action_space, self.action_space))
                    f.write("\nACTION i: %d, ACTION j: %d\n" % (ai, aj))
                    f.write(str(reward_func_inter[a]))
        elif info_present is "console":
            print("-" * 20)
            print("LEVEL-0 TRANSITION FUNCTION")
            for a in range(len(trans_func_l0)):
                ai, aj = lin_to_joint(a, (self.action_space, self.action_space))
                print("ACTION i: %d, ACTION j: %d" % (ai, aj))
                print(trans_func_l0[a])
            print("-" * 10)
            print("LEVEL-0 REWARD FUNCTION")
            for a in range(len(reward_func_l0)):
                ai, aj = lin_to_joint(a, (self.action_space, self.action_space))
                print("ACTION i: %d, ACTION j: %d" % (ai, aj))
                print(reward_func_l0[a])
            print("-" * 10)
            print("INTERACTIVE TRANSITION FUNCTION")
            for a in range(len(trans_func_inter)):
                ai, aj = lin_to_joint(a, (self.action_space, self.action_space))
                print("ACTION i: %d, ACTION j: %d" % (ai, aj))
                print(trans_func_inter[a])
            print("-" * 10)
            print("INTERACTIVE REWARD FUNCTION")
            for a in range(len(reward_func_inter)):
                ai, aj = lin_to_joint(a, (self.action_space, self.action_space))
                print("ACTION i: %d, ACTION j: %d" % (ai, aj))
                print(reward_func_inter[a])
        else:
            pass

        return trans_func_l0, reward_func_l0, \
               trans_func_inter, reward_func_inter

    def transit(self, state, action, agent_level):
        """

        :param state:
        :param action:
        :param agent_level:
        :return:
        state_prime: next physical state (joint form)
        state_prime_lin: next physical state (linear form)
        reward: the reward an agent get when the action is applied at the state
        """
        if np.ndim(state):
            state = joint_to_lin(
                state, (self.agent_loc_space, self.door_loc_space))
        if np.ndim(action):
            action = joint_to_lin(
                action, (self.action_space, self.action_space))

        assert agent_level >= 0
        if not agent_level:
            dist = self.l0_trans_func[action][state, :].toarray()[0]
            state_prime_lin = np.random.choice(
                np.arange(self.num_l0_phys_state), p=dist)
            state_prime = lin_to_joint(
                state_prime_lin, (self.agent_loc_space, self.door_loc_space))
            reward = self.l0_reward_func[action][state, state_prime_lin]
        else:
            dist = self.inter_trans_func[action][state, :].toarray()[0]
            state_prime_lin = np.random.choice(
                np.arange(self.num_inter_phys_state), p=dist)
            state_prime = lin_to_joint(
                state_prime_lin,
                (self.agent_loc_space,
                 self.agent_loc_space,
                 self.door_loc_space))
            reward = self.inter_reward_func[action][state, state_prime_lin]

        return state_prime, state_prime_lin, reward

    def observe(self, state, action, agent_level):
        if not agent_level:
            if np.ndim(state):
                state = joint_to_lin(
                    state, (self.agent_loc_space, self.door_loc_space))
            if np.ndim(action):
                action = joint_to_lin(
                    action, (self.action_space, self.action_space))
            dist = self.l0_obs_func[action, state, :]
            obs = np.random.choice(self.obs_space, p=dist)

        else:
            if np.ndim(state):
                state = joint_to_lin(
                    state,
                    (self.agent_loc_space,
                     self.agent_loc_space,
                     self.door_loc_space))
            if np.ndim(action):
                action = joint_to_lin(
                    action, (self.action_space, self.action_space))
            dist = self.inter_obs_func[action, state, :]
            obs = np.random.choice(self.obs_space, p=dist)

        return obs

    def get_intent_side_cells(self, move_action, cell):
        # The agent moves towards the intentional direction and reaches its
        # intentional next cell. The likelihood of the agent moves intentionally
        # is pr_trans_intent.
        move_intent = self.moves[move_action]
        cell_intent = self.get_next_cell(move_intent, cell)

        # The agent falsely moves towards either orthogonal direction of the
        # intentional one and slides into one neighbor cell. The likelihood of
        # the agent slides accidentally at the right angle of the intentional
        # direction is evenly equal to (1 - pr_trans_intent) / 2.
        move_side = [[int(np.logical_not(x)) if x else x + 1
                      for x in move_intent],
                     [int(np.logical_not(y)) if y else y - 1
                      for y in move_intent]]
        cell_side = list()
        for move in move_side:
            cell_side.append(self.get_next_cell(move, cell))

        if len(cell_side) == 1:
            cell_side = cell_side[0]

        return cell_intent, cell_side

    def get_next_cell(self, move, cell):
        coord = np.array(self.cell_lin_to_bin(cell))
        coord_next = self.apply_move(coord_in=coord, move=move)
        if self.is_outofbound(coord_next):
            coord_next = coord.copy()
        cell_next = self.cell_bin_to_lin(coord_next)

        return cell_next

    @staticmethod
    def apply_move(coord_in, move):
        coord = coord_in.copy()
        coord += move
        return coord

    def gen_door_cells(self):
        # Randomly pick one cell from the door space.
        gold_cell = np.random.choice(self.params.door_loc_space)
        # Obtain the index of the gold cell in the door_loc_space.
        idx_gold_in_space = np.argwhere(
            gold_cell == self.door_loc_space)[0][0]
        # Infer the tiger cell according to the gold cell. In the Tiger-Grid
        # domain, the tiger always appears on the diagonal of the gold within
        # the top-right 4 cells.
        idx_tiger_in_space = \
            self.door_loc_space.shape[0] - 1 - idx_gold_in_space
        # Index the tiger cell.
        tiger_cell = self.door_loc_space[idx_tiger_in_space]

        return gold_cell, tiger_cell

    def gen_init_cells(self):
        return np.random.choice(
            self.agent_birth_space, size=self.num_agent)

    def gen_init_belief(self):
        num_agent_birthplace = self.agent_birth_space.shape[0]
        num_door_loc_space = self.door_loc_space.shape[0]

        # level-0 initial belief over physical state space
        b0 = np.zeros(self.num_l0_phys_state)
        birth_indices = list()

        for i in self.agent_birth_space:
            for j in self.door_loc_space:
                birth_indices.append(joint_to_lin(
                    [i, j], (self.agent_loc_space, self.door_loc_space)))

        b0[birth_indices] = np.divide(1, len(birth_indices))

        # Initial belief over interactive state space
        ib0 = np.zeros((self.num_inter_phys_state, 2), dtype='object')

        for i in self.agent_loc_space:
            for j in self.agent_loc_space:
                for k in self.door_loc_space:
                    idx = joint_to_lin(
                        [i, j, k],
                        (self.agent_loc_space,
                         self.agent_loc_space,
                         self.door_loc_space))
                    ib0[idx][0] = np.append(idx, b0)

        for i in self.agent_birth_space:
            for j in self.agent_birth_space:
                for k in self.door_loc_space:
                    idx = joint_to_lin(
                        [i, j, k],
                        (self.agent_loc_space,
                         self.agent_loc_space,
                         self.door_loc_space))
                    ib0[idx][1] = np.divide(
                        1, num_agent_birthplace * num_agent_birthplace * num_door_loc_space)

        return b0, ib0

    @staticmethod
    def gen_grid(grid_n, grid_m):
        grid = np.zeros([grid_n, grid_m])
        return grid

    def is_outofbound(self, coord_in):
        assert (np.array(coord_in).shape[0] == 2), "Only support 2D coordinates"
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

    @staticmethod
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
        num_inter_phys_state = params.num_inter_phys_state
        num_l0_phys_state = params.num_l0_phys_state
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
                         name='terminals',
                         atom=tables.Int32Atom(),
                         shape=(0, grid_n, grid_m),
                         expectedrows=total_traj_count)

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
                         name='inter_bs',
                         atom=tables.Float32Atom(),
                         shape=(0, num_inter_phys_state),
                         expectedrows=total_traj_count)

        db.create_earray(where=db.root,
                         name='l0_bs',
                         atom=tables.Float32Atom(),
                         shape=(0, num_l0_phys_state),
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
                         # env_id, terminals, step_id, traj_length, failed
                         expectedrows=total_traj_count)
        return db

    def process_beliefs(self, b_self, b_other):
        """
        :param b_self: belief of the subjective/modeling agent
        :param b_other: belief of the objective/modeled agent
        :return: belief reshaped to [batch_size, state_space_size]
        """
        batch = (b_other.shape[0] if b_other.ndim > 1 else 1)
        print("Shape of b_self:", b_self.shape)
        print("Shape of b_other:", b_other.shape)
        if b_other.shape[0] == self.num_l0_phys_state:
            b_other = b_other.reshape((batch, self.num_l0_phys_state))
        else:
            assert b_other.shape[0] == self.num_inter_phys_state
            b_other = b_other.reshape((batch, self.num_inter_phys_state))

        assert b_self.shape[0] == self.num_inter_phys_state
        b_self = b_self.reshape((batch, self.num_inter_phys_state))

        return b_self.astype('f'), b_other.astype('f')


def generate_grid_data(path,
                       grid_n,
                       grid_m,
                       num_gold,
                       num_tiger,
                       num_agent,
                       p_trans_intent,
                       p_obs_dir_corr,
                       p_obs_door_corr,
                       agent_level,
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
    :param agent_level:
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
        'num_particle_pf': 100,

        # state-related
        'agent_loc_space': np.arange(grid_n * grid_m),
        'agent_birth_space': np.arange(grid_n * grid_m)[-grid_m:],
        'door_loc_space': np.arange(grid_n * grid_m).reshape(grid_n, grid_m)[:2, -2:].reshape(4),

        # reward-related
        'reward_gold': 50.0,
        'reward_tiger': -100.0,
        'reward_listen': 0.0,
        'reward_wrong_open': -1.0,
        'reward_move': -1.0,

        # self_action-relate: stay, move-N, move-E, move-S, move-W,
        # open the door
        'action_space': np.arange(6),
        'moves': [[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]],
        'init_action': 0,

        # observation-related: glitter-N, glitter-E, glitter-S, glitter-W,
        # door-found
        'obs_space': np.arange(5),
        'observe_directions': [[-1, 0], [0, 1], [1, 0], [0, -1]],

        # pomdp_function-related
        'pr_trans_intent': p_trans_intent,
        'pr_obs_dir_corr': p_obs_dir_corr,
        'pr_obs_door_corr': p_obs_door_corr,

        'discount': 0.9,
        'agent_level': agent_level
    })
    # params.l0_phys_state_space is the physical state space for the level-0
    # agent, where each of the state consists of the agent's own location and
    # the location of the gold-door.
    params['l0_phys_state_space'] = np.array(
        [[x, y] for x in params.agent_loc_space for y in params.door_loc_space])
    # params.inter_phys_state_space is the physical state space for the higher-
    # level agent, including the other's locations as well
    params['inter_phys_state_space'] = np.array(
        [[x, y, z] for x in params.agent_loc_space
         for y in params.agent_loc_space
         for z in params.door_loc_space])
    params['num_l0_phys_state'] = params.l0_phys_state_space.shape[0]
    params['num_inter_phys_state'] = params.inter_phys_state_space.shape[0]

    params['num_action'] = params.action_space.shape[0]
    params['num_joint_action'] = params.num_action ** 2
    params['joint_action_space'] = np.array(
        [[x, y] for x in params.action_space for y in params.action_space])

    params['num_obs'] = params.obs_space.shape[0]
    params['traj_limit'] = 4 * (params.grid_n + params.grid_m)  # reason?

    # save params
    if not os.path.isdir(path):
        os.mkdir(path)
    pickle.dump(dict(params), open(path + "/params.pickle", 'wb'), -1)

    # randomize seeds, set to previous value to determine random numbers
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
        domain.generate_trajectories(
            db, num_traj=traj_per_env, level=agent_level)

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
        default=3,
        help='The height of the grid world.')
    parser.add_argument(
        '--M',
        type=int,
        default=3,
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
        default=0.9,
        help='The probability that the agent receives the correct observation'
             'of the direction of the door when it is not in a door-cell.')
    parser.add_argument(
        '--p_obs_door_corr',
        type=float,
        default=1.0,
        help='The probability that the agent receives the correct observation'
             'when it occupies a door-cell.')
    parser.add_argument(
        '--agent_level',
        type=int,
        default=1,
        help='The nested level of the subjective/modeling agent in the I-POMDP '
             'framework.')
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
                       agent_level=args.agent_level,
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
                       agent_level=args.agent_level,
                       num_env=args.test,
                       traj_per_env=args.test_trajs)


# default
if __name__ == "__main__":
    main()
