import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds
from baselines.common import tf_util
from gym.spaces.multi_discrete import MultiDiscrete

from baselines.a2c.utils import make_path, find_trainable_variables


class PretrainTask():
    def __init__(self, batch_size, obs_size, low=-1, high=1):
        self._batch_size = batch_size
        self._obs_size = obs_size
        self._low = low
        self._high = high

    def sample(self, steps):
        input_ = np.random.random(size=(self._batch_size, steps, self._obs_size)) * (self._high - self._low) + self._low
        targ_f = input_
        targ_b = np.copy(input_)
        targ_b = targ_b[:, ::-1, :]
        # pad length so that backwards can be trained properly
        zeros = np.zeros_like(input_)
        dnc_mask = np.ones(shape=(self._batch_size, 2*steps, 1))
        train_mask = np.concatenate((np.zeros(shape=(steps)), np.ones(shape=(steps))), axis=0)
        input_ = np.concatenate((input_, zeros), axis=1)
        targ_f = np.concatenate((zeros, targ_f), axis=1)
        targ_b = np.concatenate((zeros, targ_b), axis=1)
        return input_, dnc_mask, train_mask, targ_f, targ_b


class Model(object):

    def __init__(self, sess, policy, policy_args, ob_space, ac_space, nenvs, nsteps,
            max_grad_norm=0.5, momentum=0, alpha=0.99, epsilon=1e-5, reuse=True):

        with tf.variable_scope('model', reuse=reuse):

            model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, args=policy_args, reuse=reuse)
            model_in_xs = model.dnc_in_xs
            model_in_ms = model.dnc_in_ms
            model_out = model.dnc_out

            s = 1
            s_rem = model_out.get_shape().as_list()[2] - 2*s
            targs_f, targs_b, _ = tf.split(model_out, num_or_size_splits=[s, s, s_rem], axis=2)
            train_shape = (nenvs, nsteps, s)

            TARGET_FORWARD = tf.placeholder(tf.float32, shape=train_shape, name='TARGET_F')
            TARGET_BACKWARD = tf.placeholder(tf.float32, shape=train_shape, name='TARGET_B')
            MASK = tf.placeholder(tf.float32, shape=(nsteps))
            LR = tf.placeholder(tf.float32, [])

            def cost(logits, target, mask):
                # xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
                mse = tf.square(logits-target)/2.
                loss_time_batch = tf.reduce_sum(mse, axis=2)
                loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=1)
                batch_size = tf.cast(tf.shape(logits)[0], dtype=loss_time_batch.dtype)
                return tf.reduce_sum(loss_batch) / batch_size

            print('Creating model for length', nsteps)
            loss = cost(targs_f, TARGET_FORWARD, MASK) + cost(targs_b, TARGET_BACKWARD, MASK)

            params = find_trainable_variables("model")
            grads = tf.gradients(loss, params)
            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon, momentum=momentum)
            _train = trainer.apply_gradients(grads)

        def train(obs, mask_dnc, mask_train, target_f, targets_b, lr):
            loss_, _ = sess.run([loss, _train], feed_dict={
                LR: lr,
                model_in_xs: obs,
                model_in_ms: mask_dnc,
                TARGET_FORWARD: target_f,
                TARGET_BACKWARD: targets_b,
                MASK: mask_train,
            })
            return loss_

        def save(save_path, file_name, log=False):
            if log:
                logger.info("Saving to ", save_path, file_name)
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path+file_name)

        def load(load_path, file_name, log=True):
            try:
                loaded_params = joblib.load(load_path+file_name)
                restores = []
                for p, loaded_p in zip(params, loaded_params):
                    restores.append(p.assign(loaded_p))
                ps = sess.run(restores)
                if log:
                    logger.info("Loaded from ", load_path, file_name)
                return True
            except FileNotFoundError:
                logger.warn("Loading failed, path does not exist! ", load_path, file_name)
            return False

        self.train = train
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


def learn(policy, policy_args, env, env_args, seed=0, lr=1e-4, log_interval=100, max_len=9, start_len=4,
          max_iterations=10e6, max_loss_to_increase=0.05, save_path='', save_name='model'):
    tf.reset_default_graph()
    set_global_seeds(seed)
    sess = tf_util.make_session()

    nenvs = env.num_envs
    ob_space = env.observation_space
    nact = [ac.n for ac in env.action_space.spaces]
    ac_space = MultiDiscrete(nact)

    recent_avg_len, recent_loss = 0, 0
    cur_max_len = start_len
    task = PretrainTask(nenvs, 1)
    # generate models with different lengths but shared variables, easier than changing the policy
    models = [Model(sess, policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space,
                    max_grad_norm=50, nenvs=nenvs, nsteps=1, reuse=False)]
    for i in range(1, max_len):
        models.append(Model(sess, policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space,
                            max_grad_norm=50, nenvs=nenvs, nsteps=2*i))
    models[0].load(save_path, save_name)

    for i in range(1, int(max_iterations)):
        ran_len = np.random.randint(1, cur_max_len)
        input_, mask_dnc, mask_train, forward_, backward_ = task.sample(ran_len)
        l = models[ran_len].train(input_, mask_dnc, mask_train, forward_, backward_, lr)
        recent_avg_len += ran_len
        recent_loss += l

        if i % log_interval == 0:
            avg_loss = recent_loss / log_interval
            logger.record_tabular("niterations", i)
            logger.record_tabular("max_len", cur_max_len - 1)
            logger.record_tabular("avg_len", recent_avg_len / log_interval)
            logger.record_tabular("avg_loss", avg_loss)
            logger.record_tabular("lr", lr)
            logger.dump_tabular()
            models[0].save(save_path, save_name)
            if avg_loss < max_loss_to_increase:
                cur_max_len += 1
                if cur_max_len > max_len:
                    break
            recent_avg_len = 0
            recent_loss = 0

    env.close()
