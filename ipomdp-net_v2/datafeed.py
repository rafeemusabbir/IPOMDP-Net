import numpy as np
from tensorpack import dataflow

from database import Database
from env_tiger_grid import TigerGridBase

try:
    import ipdb as pdb
except ImportError:
    import pdb


class DataFeed:
    def __init__(self, params, filename, mode="train", min_env=0, max_env=0):
        """
        Datafeed from filtered samples
        :param params: dotdict including both domain parameters and training
        parameters
        :param filename: database file path
        :param mode: "train" or "valid" or "eval"
        :param min_env: only include environments with id larger than this
        :param max_env: only include environments with id smaller than this.
        No limit if set to zero.
        """
        self.params = params
        self.mode = mode
        self.steps_in_epoch = None
        self.filename = filename
        self.min_env = min_env
        self.max_env = max_env

        # wrapper that returns a database object
        self.get_db = (lambda: Database(filename=filename))

        self.domain = TigerGridBase(params)

        # Preload filtered samples before being forked
        only_valid = (mode is not "eval" and not self.params.include_failed)
        self.filtered_samples = self.filter_samples(
            train_params=params,
            db=self.get_db(),
            only_valid=only_valid,
            min_env=min_env,
            max_env=max_env)

    @staticmethod
    def filter_samples(train_params, db, only_valid=True, min_env=0, max_env=0):
        """
        Preload samples and produces filtered_samples filtered according to
        training parameters
        Sample filter format: (sample_id, env_id, goal_state, step_id,
        effective_traj_len)
        """
        db.open()

        # Preload all samples, because random access is slow(?)
        # env_id, step_id, traj_length, failed
        samples = db.samples[:]

        # Filter valid samples.
        if only_valid:
            sample_indices = db.valids[:]
        else:  # Why we might need invalid samples for training?
            sample_indices = np.arange(len(db.samples))

        # Filter environment range.
        if max_env > 0 or min_env > 0:
            env_indices = samples[sample_indices, 0]  # Get env_id
            # Transform percentage to index
            if not max_env:
                maxenvi = 9999999
            elif max_env <= 1.0:
                maxenvi = int(len(db.envs) * max_env)
            else:
                maxenvi = int(max_env)

            if min_env <= 1.0:
                minenvi = int(len(db.envs) * min_env)
            else:
                minenvi = int(min_env)

            sample_indices = sample_indices[
                np.nonzero(np.logical_and(env_indices >= minenvi,
                                          env_indices < maxenvi))[0]]

            print("Environments are limited to the range %d-%d from %d" %
                  (minenvi, (maxenvi if max_env else 0), len(db.envs)))

        samples = samples[sample_indices]
        db.close()

        # arr traj lens: array of lengths of the trajectories.
        arr_traj_len = samples[:, 3]

        # limit effective_traj_lens to lim_traj_len
        if train_params.lim_traj_len > 0:
            print("Limit traj_len to %d" % train_params.lim_traj_len)
            arr_traj_len = np.clip(arr_traj_len, 0, train_params.lim_traj_len)

        # sanity check
        assert np.all(arr_traj_len > 0)

        filtered_samples = np.stack((
            sample_indices,  # sample_id
            samples[:, 0],  # env_i
            samples[:, 1],  # terminal_i
            samples[:, 2],  # step_i
            arr_traj_len,  # processed arr_traj_len
        ), axis=1)

        return filtered_samples

    def build_dataflow(self,
                       batch_size,
                       step_size,
                       restart_limit=None,
                       cache=None):
        """
        :param batch_size: batch size
        :param step_size: number of steps for BPTT
        :param restart_limit: restart after limit number of batches.
        Used for validation. If 0 (but not None) or larger than an epoch its
        set to one epoch.
        :param cache: preloaded cache
        :return: dataflow with input data
        """
        # update db wrapper function with shared cache
        if cache is not None:
            self.get_db = (
                lambda: Database(filename=self.filename, cache=cache))

        df = dataflow.DataFromList(
            lst=list(self.filtered_samples),
            shuffle=(self.mode == 'train'))

        if restart_limit is None:  # restart_limit can be 0, which is not None.
            # Reshuffles on every repeat
            df = dataflow.RepeatedData(df, 1000000)

        df = DynamicTrajBatch(
            df,
            batch_size=batch_size,
            step_size=step_size,
            arr_traj_len=self.filtered_samples[:, 4])

        self.steps_in_epoch = df.steps_in_epoch()
        # print("steps_in_epoch:", self.steps_in_epoch)
        if restart_limit is not None:
            if restart_limit == 0 or restart_limit >= self.steps_in_epoch:
                restart_limit = self.steps_in_epoch - 1
            self.steps_in_epoch = restart_limit
            df = OneShotData(df, size=restart_limit)

        df = TrajDataFeed(
            ds=df,
            get_db_func=self.get_db,
            domain=self.domain,
            batch_size=batch_size,
            step_size=step_size)

        # Switch of dataflow speed test. (Uncomment to enable)
        # dataflow.TestDataSpeed(df, size=1000).start()

        return df

    def build_eval_dataflow(self, policy=None, repeats=None):
        """
        :param policy: policy to evaluate when mode == eval
        :param repeats: repeat evaluation multiple times when mode == eval
        :return: dataflow with evaluation results
        """
        df = dataflow.DataFromList(lst=list(self.filtered_samples),
                                   shuffle=False)
        df = dataflow.RepeatedData(df, 1000000)

        df = EvalDataFeed(
            ds=df,
            get_db_func=self.get_db,
            domain=self.domain,
            policy=policy,
            repeats=repeats)

        return df

    def build_cache(self):
        """
        Preload cache of the database
        For multiprocessing, call this before fork. Input cache to all next
        instances of the database
        """
        db = self.get_db()
        cache = db.build_cache(cache_nodes=self.params.cache)
        db.close()

        return cache


class DynamicTrajBatch(dataflow.BatchDataByShape):
    def __init__(self, ds, batch_size, step_size, arr_traj_len):
        """
        Breaks trajectories into trainings steps and collets batches. Assume
        sequential input

        Makes batches for BPTT from trajectories of different length. Batch is
        divided into blocks where BPTT is performed. Trajectories are padded to
        block limits. New trajectory begins from the next block, even when other
        trajectories are not finished in the batch.

        Behaviour is similar to:
        https://blog.altoros.com/the-magic-behind-google-translate-sequence-to-
        sequence-models-and-tensorflow.html

        :param ds: sequential input dataflow. Expected sample form is
            (sample_id, env_id, step_id, traj_len)
        :param batch_size: batch size
        :param step_size: step size for BPTT
        :param arr_traj_len: list of numpy array of trajectory lengths for ALL
        samples in dataset. Used to compute size()
        :return batched data, each with shape [step_size, batch_size, ...].
            Adds is_start field, a binary indicator:
            1 if it is the first step of the trajectory, 0 otherwise.
            Output: (sample_id, env_id, step_id, traj_len, is_start)
        """
        super(DynamicTrajBatch, self).__init__(
            ds=ds,
            batch_size=batch_size,
            idx=0)

        self.batch_size = batch_size
        self.step_size = step_size

        self.batch_samples = None

        self.step_field = 3
        self.traj_len_field = 4
        self.sample_fields = 6  # sample_id, env_id, terminals, step_id, traj_len, is_start

        blocks = ((arr_traj_len - 1) // self.step_size) + 1
        self._steps_in_epoch = np.sum(blocks) // self.batch_size
        self._total_epochs = (
            None if ds.size() is None
            else ((ds.size() - 1) // len(arr_traj_len)) + 1)

    def size(self):
        return self._steps_in_epoch * self._total_epochs

    def steps_in_epoch(self):
        return self._steps_in_epoch

    def reset_state(self):
        super(DynamicTrajBatch, self).reset_state()

    def get_data(self):
        with self._guard:
            self.batch_samples = np.zeros(
                [self.batch_size, self.sample_fields], 'i')
            generator = self.ds.get_data()
            try:
                while True:
                    # Collect which samples should be replaced by a new one.
                    # For the non-zero indices, the sample is still valid in
                    # the batch.
                    self.batch_samples[:, self.step_field] += self.step_size
                    self.batch_samples[:, self.traj_len_field] -= self.step_size
                    self.batch_samples[:, -1] = 0  # is_start

                    new_indices = np.nonzero(
                        self.batch_samples[:, self.traj_len_field] <= 0)[0]
                    self.batch_samples[new_indices, -1] = 1

                    for idx in new_indices:
                        # replace these samples in batch
                        self.batch_samples[idx, :-1] = next(
                            generator)  # get new datapoint, list of fields

                    yield self.batch_samples
            except StopIteration:
                return


class EvalDataFeed(dataflow.ProxyDataFlow):
    def __init__(self, ds, get_db_func, domain, policy, repeats=1):
        super(EvalDataFeed, self).__init__(ds)
        self.get_db = get_db_func
        self.domain = domain
        self.policy = policy
        self.repeats = repeats

        self.db = None

    def __del__(self):
        self.close()

    def reset_state(self):
        super(EvalDataFeed, self).reset_state()
        if self.db is not None:
            print("WARNING: Reopening database. This is not recommended.")
            self.db.close()
        self.db = self.get_db()
        self.db.open()

    def close(self):
        if self.db is not None:
            self.db.close()
        self.db = None

    def get_data(self):
        for dp in self.ds.get_data():
            yield self.eval_sample(dp)

    def eval_sample(self, sample):
        """
        :param sample: sample vector in the form
        (sample_id, env_id, step_id, traj_len)
        :return result matrix, first row for expert policy, consecutive rows
        for evaluated policy.
        fields: success rate, trajectory length, accumulated reward
        """
        sample_i, env_i, terminal_i, step_i, _ = \
            [np.atleast_1d(x.squeeze())
             for x in np.split(sample, sample.shape[0], axis=0)]

        env = self.db.envs[env_i[0]]
        terminals = self.db.terminals[terminal_i[0]]
        b0 = self.db.l0_bs[sample_i[0]]
        ib0 = self.db.inter_bs[sample_i[0]]
        db_sample = self.db.samples[sample_i[0]]
        db_step = self.db.steps[step_i[0]]

        _, _, _, traj_len, failed = db_sample
        phys_state_self_lin, act_self_last, obs_self_last = db_step

        success = (1 if failed == 0 else 0)

        reward_sum = self.db.expRs[sample_i]

        # Statistics: success rate, trajectory length, accumulated reward.
        # First row for expert, second row for evaluated policy.
        results = np.zeros([self.repeats + 1, 3], 'f')
        results[0] = np.array([success, traj_len, reward_sum], 'f')

        for eval_i in range(self.repeats):
            success, traj_len, reward_sum = \
                self.domain.simulate_policy(
                    policy=self.policy,
                    grid=env,
                    terminals=terminals,
                    l0_b0=b0,
                    ib0=ib0,
                    init_phys_state=phys_state_self_lin,
                    first_action=act_self_last)
            success = (1 if success else 0)

            results[eval_i + 1] = np.array(
                [success, traj_len, reward_sum], 'f')

        return results  # success, traj_len, reward_sum


class TrajDataFeed(dataflow.ProxyDataFlow):
    """
    Load training data from database given batched samples.
    Inputs are batched samples of shape [step_size, batch_size, 5]
    Each sample corresponds to:
    (sample_id, env_id, goal_state, step_id, traj_len)
    """

    def __init__(self, ds, get_db_func, domain, batch_size, step_size):
        super(TrajDataFeed, self).__init__(ds)
        self.get_db = get_db_func
        self.domain = domain
        self.batch_size = batch_size
        self.step_size = step_size

        self.db = None

    def __del__(self):
        self.close()

    def reset_state(self):
        super(TrajDataFeed, self).reset_state()
        if self.db is not None:
            print("WARNING: reopening database. This is not recommended.")
            self.db.close()
        self.db = self.get_db()
        self.db.open()

    def close(self):
        if self.db is not None:
            self.db.close()
        self.db = None

    def get_data(self):
        for dp in self.ds.get_data():
            yield self.process_samples(dp)

    def process_samples(self, samples):
        """
        :param samples: numpy array, axis 0 for trajectory steps, axis 1 for
        batch, axis 2 for sample descriptor
        Sample descriptor is represented as follow:
        sample_id -- index in original database,
        env_i
        goal_states
        step_i,
        b_index,
        traj_len,
        is_start)
        """
        sample_i, env_i, terminal_i, step_i, traj_len, is_start = [np.atleast_1d(
            x.squeeze()) for x in np.split(samples, samples.shape[1], axis=1)]

        env_img = self.db.envs[:][env_i]
        terminal_map = self.db.terminals[:][terminal_i]

        # Initial belief for both level-0 and higher-level agents.
        b0_other = self.db.l0_bs[:][sample_i]
        b0_self = self.db.inter_bs[:][sample_i]

        # produce all steps consisting of last_action, last_obs, act_label
        step_indices = step_i[None, :] + np.arange(self.step_size + 2)[:, None]
        step_indices = step_indices.clip(max=len(self.db.steps) - 1)

        # mask for valid steps vs zero padding
        valid_mask = np.nonzero(
            np.arange(self.step_size)[:, None] < traj_len[None, :])

        # actions
        step_idx_helper = step_indices[:self.step_size][valid_mask]
        label_idx_helper = step_indices[1:self.step_size + 1][valid_mask]

        acts_last = np.zeros((self.step_size, self.batch_size), 'i')
        acts_last[valid_mask] = self.db.steps[step_idx_helper, 1]

        acts_label = np.zeros(acts_last.shape, 'i')
        # NOTE: inefficient if loading from disk.
        acts_label[valid_mask] = self.db.steps[label_idx_helper, 1]

        local_obs = self.db.steps[step_indices[:self.step_size][valid_mask], 2]
        obs = np.zeros((self.step_size, self.batch_size), 'i')

        obs[valid_mask] = local_obs

        # set weights
        weights = np.zeros(acts_last.shape, 'f')
        weights[valid_mask] = 1.0

        return [env_img, terminal_map, b0_other, b0_self, is_start, acts_last,
                obs, weights, acts_label]


class OneShotData(dataflow.FixedSizeData):
    """
    DataFlow repeated after fixed number of samples
    """

    def size(self):
        return 1000000

    def get_data(self):
        with self._guard:
            while True:
                itr = self.ds.get_data()
                try:
                    for cnt in range(self._size):
                        yield next(itr)
                except StopIteration:
                    print("End of dataset reached")
                    raise StopIteration
