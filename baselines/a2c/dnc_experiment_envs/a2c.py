import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.DNC.DNCVisualizedPlayer import DNCVisualizedPlayer

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse


class Model(object):

    def __init__(self, policy, policy_args, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4, momentum=0,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.make_session()
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, args=policy_args, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, args=policy_args, reuse=True)

        neglogpac = train_model.pd.neglogp(A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon, momentum=momentum)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

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
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state()
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        self.nenv = env.num_envs
        self.obs = np.zeros((self.nenv,), dtype=np.uint8)
        self.obs = env.reset()
        self.batch_ob_shape = (self.nenv*nsteps, env.observation_space.n)
        self.nc = 1
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(self.nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions.tolist())
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


def learn(policy, policy_args, env, env_args, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, momentum=0,
          log_interval=100, save_path='', save_name='model'):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs,
                  nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr,
                  alpha=alpha, epsilon=epsilon, momentum=momentum, total_timesteps=total_timesteps,
                  lrschedule=lrschedule)
    model.load(save_path, save_name)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    total_updates = total_timesteps//nbatch+1
    initial_update = 1
    for update in range(initial_update, total_updates):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        if update % log_interval == 0 or update == 1:
            nseconds = time.time()-tstart
            fps = int((update*nbatch)/nseconds)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
            logger.info("Time since start: \t%.2fs" % nseconds)
            rem_time = (nseconds * (total_updates - initial_update) / (update - initial_update + 1)) - nseconds
            logger.info("ETA: \t\t\t\t%.2fs" % rem_time)
            model.save(save_path, save_name)

    model.save(save_path, save_name)
    env.close()


def play(policy, policy_args, env, env_args, seed, nep=5, save_path='', save_name='model'):
    tf.reset_default_graph()
    set_global_seeds(seed)

    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space, nenvs=1, nsteps=1)
    model.load(save_path, save_name)

    def update_obs(obs):
        return [obs]

    if nep <= 0:
        DNCVisualizedPlayer.player(env, model, nstack=1, env_args=env_args, player_args={
            'dnc_exp_env': True,
        })
    else:
        total_reward = 0
        for e in range(nep):
            done = False
            obs = env.reset()
            obs = update_obs(obs)
            states = model.initial_state
            episode_reward = 0
            while not done:
                actions, values, states, _ = model.step(obs, states, [done])
                obs, reward, done, info = env.step(actions.tolist()[0])
                obs = update_obs(obs)
                episode_reward += reward
                env.render()
            print('Episode reward:', episode_reward)
            total_reward += episode_reward
