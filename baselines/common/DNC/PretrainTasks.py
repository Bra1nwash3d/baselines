import numpy as np


class PretrainTask():
    def __init__(self, batch_size, obs_size, low=-1, high=1, task_type='int'):
        self._batch_size = batch_size
        self._obs_size = obs_size
        self._low = low
        self._high = high
        self._numbers = {
            'int': self._int_numbers,
            'float': self._float_numbers
        }.get(task_type, self._int_numbers)

    def _int_numbers(self, steps):
        return np.random.randint(self._low, self._high, size=(self._batch_size, steps, self._obs_size))

    def _float_numbers(self, steps):
        return np.random.random(size=(self._batch_size, steps, self._obs_size)) * (self._high - self._low) + self._low

    def sample(self, steps):
        input_ = self._numbers(steps)
        targ_f = input_
        targ_b = np.copy(input_)
        targ_b = targ_b[:, ::-1, :]
        # pad length so that backwards can be trained properly
        zeros = np.zeros_like(input_)
        dnc_mask = np.ones(shape=(self._batch_size, 2*steps, 1))
        train_f = np.ones(shape=(2*steps))
        train_b = np.concatenate((np.zeros(shape=(steps)), np.ones(shape=(steps))), axis=0)
        input_ = np.concatenate((input_, zeros), axis=1)
        targ_f = np.concatenate((zeros, targ_f), axis=1)
        targ_b = np.concatenate((zeros, targ_b), axis=1)
        return input_, dnc_mask, train_f, train_b, targ_f, targ_b


class PretrainTaskOneHot():
    def __init__(self, batch_size, obs_size, ob_depth=1):
        self._batch_size = batch_size
        self._obs_size = obs_size
        self._ob_depth = ob_depth
        self._numbers = self._int_numbers
        self._eye = np.eye(ob_depth)

    def _int_numbers(self, steps):
        return np.random.randint(0, self._ob_depth, size=(self._batch_size, steps, self._obs_size))

    def sample(self, steps):
        input_ = self._numbers(steps)
        targ_f = np.squeeze(self._eye[input_], axis=2)
        targ_b = np.copy(targ_f)
        targ_b = targ_b[:, ::-1, :]
        # pad length so that backwards can be trained properly
        dnc_mask = np.ones(shape=(self._batch_size, 2*steps, 1))
        train_mask_f = np.ones(shape=(2*steps))
        train_mask_b = np.concatenate((np.zeros(shape=(steps)), np.ones(shape=(steps))), axis=0)
        input_ = np.concatenate((input_, np.zeros_like(input_)), axis=1)
        targ_f = np.concatenate((targ_f, targ_f), axis=1)
        targ_b = np.concatenate((np.zeros_like(targ_b), targ_b), axis=1)
        return input_, dnc_mask, train_mask_f, train_mask_b, targ_f, targ_b