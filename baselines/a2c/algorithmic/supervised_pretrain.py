import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds
from baselines.common import tf_util
from baselines.common.DNC import PretrainTasks
from gym.spaces.multi_discrete import MultiDiscrete

from baselines.a2c.utils import make_path, find_trainable_variables


class Model(object):

    def __init__(self, sess, policy, policy_args, ob_space, ac_space, nenvs, nsteps,
            max_grad_norm=0.5, momentum=0, alpha=0.99, epsilon=1e-5, one_hot=False, reuse=True):

        with tf.variable_scope('model', reuse=reuse):

            model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, args=policy_args, reuse=reuse)
            model_in_xs = model.dnc_in_xs
            model_in_ms = model.dnc_in_ms
            model_out = model.dnc_out

            s = 1
            if one_hot:
                s = ob_space.n
            s_rem = model_out.get_shape().as_list()[2] - 2*s
            targs_f, targs_b, _ = tf.split(model_out, num_or_size_splits=[s, s, s_rem], axis=2)
            train_shape = (nenvs, nsteps, s)

            TARGET_FORWARD = tf.placeholder(tf.float32, shape=train_shape, name='TARGET_F')
            TARGET_BACKWARD = tf.placeholder(tf.float32, shape=train_shape, name='TARGET_B')
            MASK_FORWARD = tf.placeholder(tf.float32, shape=(nsteps))
            MASK_BACKWARD = tf.placeholder(tf.float32, shape=(nsteps))
            LR = tf.placeholder(tf.float32, [])

            def cost(logits, target, mask):
                # xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
                mse = tf.square(logits-target)/2.
                loss_time_batch = tf.reduce_sum(mse, axis=2)
                loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=1)
                batch_size = tf.cast(tf.shape(logits)[0], dtype=loss_time_batch.dtype)
                return tf.reduce_sum(loss_batch) / batch_size

            print('Creating model for length', nsteps)
            loss = cost(targs_f, TARGET_FORWARD, MASK_FORWARD) + cost(targs_b, TARGET_BACKWARD, MASK_BACKWARD)

            params = find_trainable_variables("model")
            grads = tf.gradients(loss, params)
            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon, momentum=momentum)
            _train = trainer.apply_gradients(grads)

        def train(obs, mask_dnc, mask_f, mask_b, target_f, targets_b, lr):
            loss_, _ = sess.run([loss, _train], feed_dict={
                LR: lr,
                model_in_xs: obs,
                model_in_ms: mask_dnc,
                TARGET_FORWARD: target_f,
                TARGET_BACKWARD: targets_b,
                MASK_FORWARD: mask_f,
                MASK_BACKWARD: mask_b,
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


def learn(policy, policy_args, env, env_args, seed=0, lr=1e-4, log_interval=100, max_len=9, start_len=3,
          max_iterations=10e6, max_loss_to_increase=0.1, save_path='', save_name='model'):
    tf.reset_default_graph()
    set_global_seeds(seed)
    sess = tf_util.make_session()

    nenvs = env.num_envs
    ob_space = env.observation_space
    nact = [ac.n for ac in env.action_space.spaces]
    ac_space = MultiDiscrete(nact)

    recent_avg_len, recent_loss = 0, 0
    cur_max_len = start_len
    # one_hot = False
    # task = PretrainTasks.PretrainTask(nenvs, 1, high=ob_space.n, task_type=policy_args.get('pretrain_task_type', 'int'))
    one_hot = True
    task = PretrainTasks.PretrainTaskOneHot(nenvs, 1, ob_depth=ob_space.n)
    # generate models with different lengths but shared variables, easier than changing the policy
    models = [Model(sess, policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space,
                    max_grad_norm=50, nenvs=nenvs, nsteps=1, one_hot=one_hot, reuse=False)]
    for i in range(1, max_len):
        models.append(Model(sess, policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space,
                            max_grad_norm=50, nenvs=nenvs, nsteps=2*i, one_hot=one_hot))
    models[0].load(save_path, save_name)

    for i in range(1, int(max_iterations)):
        ran_len = np.random.randint(1, cur_max_len)
        input_, mask_dnc, mask_f, mask_b, forward_, backward_ = task.sample(ran_len)
        l = models[ran_len].train(input_, mask_dnc, mask_f, mask_b, forward_, backward_, lr)

        recent_avg_len += ran_len
        recent_loss += l

        if i % log_interval == 0:
            avg_loss = recent_loss / log_interval
            avg_len = recent_avg_len / log_interval
            logger.record_tabular("niterations", i)
            logger.record_tabular("max_len", cur_max_len - 1)
            logger.record_tabular("avg_len", avg_len)
            logger.record_tabular("avg_loss", avg_loss)
            logger.record_tabular("lr", lr)
            logger.dump_tabular()
            models[0].save(save_path, save_name)
            if avg_loss < (max_loss_to_increase*avg_len):
                cur_max_len += 1
                if cur_max_len > max_len:
                    break
            recent_avg_len = 0
            recent_loss = 0

    env.close()
