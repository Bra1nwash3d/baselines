import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse
from baselines.common.DNCVisualizedPlayer import DNCVisualizedPlayer


class Model(object):

    def __init__(self, policy, policy_args, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, policy_args, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, policy_args, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states != []:
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

    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.model = model
        nenv = env.num_envs
        if len(env.observation_space.shape) == 3:
            nh, nw, self.nc = env.observation_space.shape
            self.batch_ob_shape = (nenv*nsteps, nh, nw, self.nc*nstack)
            self.obs = np.zeros((nenv, nh, nw, self.nc*nstack), dtype=np.uint8)
            self.update_obs = self.update_obs_3d
        else:
            self.nc = env.observation_space.shape[-1]
            self.batch_ob_shape = (nenv*nsteps, self.nc*nstack)
            self.obs = np.zeros((nenv, self.nc*nstack), dtype=np.float32)
            self.update_obs = self.update_obs_1d
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs_3d(self, obs):
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs

    def update_obs_1d(self, obs):
        self.obs = np.roll(self.obs, shift=-self.nc, axis=1)
        self.obs[:, -self.nc:] = obs

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
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
        #discount/bootstrap off value fn
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


def learn(policy, policy_args, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6),
          vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear',
          epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100,
          save_path='', save_name='model'):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    model = Model(policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs,
                  nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule)
    model.load(save_path, save_name)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    total_updates = total_timesteps//nbatch+1
    for update in range(1, total_updates):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        if update % log_interval == 0 or update == 1:
            nseconds = time.time()-tstart
            fps = int((update*nbatch)/nseconds)
            ev = explained_variance(values, rewards)
            rem_time = (nseconds * total_updates / update) - nseconds
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
            logger.info("Time since start: \t%.2fs" % nseconds)
            logger.info("ETA: \t\t\t\t%.2fs" % rem_time)
            model.save(save_path, save_name)

    model.save(save_path, save_name)
    env.close()


def play(policy, policy_args, env, seed, nep=5, save_path='', save_name='model'):
    tf.reset_default_graph()
    set_global_seeds(seed)

    ob_space = env.observation_space
    ac_space = env.action_space
    nstack = 4
    model = Model(policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space, nenvs=1,
                  nsteps=1, nstack=nstack, num_procs=1)
    model.load(save_path, save_name)

    if nep <= 0:
        DNCVisualizedPlayer.player(env, model, nstack=nstack)
    else:
        if len(env.observation_space.shape) == 3:
            nh, nw, nc = env.observation_space.shape
            observations = np.zeros((1, nh, nw, nc*nstack), dtype=np.uint8)

            def update_obs(stored_obs, new_obs):
                stored_obs = np.roll(stored_obs, shift=-nc, axis=3)
                stored_obs[:, :, :, -nc:] = new_obs
                return stored_obs
        else:
            nc = env.observation_space.shape[-1]
            observations = np.zeros((1, nc*nstack), dtype=np.float32)

            def update_obs(stored_obs, new_obs):
                stored_obs = np.roll(stored_obs, shift=-nc, axis=1)
                stored_obs[:, -nc:] = new_obs
                return stored_obs

        total_reward = 0
        for e in range(nep):
            done = False
            new_obs = env.reset()
            observations = update_obs(observations, new_obs)
            states = model.initial_state
            episode_reward = 0
            while not done:
                actions, values, states = model.step(observations, states, [done])
                new_obs, reward, done, info = env.step(actions[0])
                observations = update_obs(observations, new_obs)
                episode_reward += reward
                env.render()
            print('Episode reward:', episode_reward)
            total_reward += episode_reward
        print('Done! Total reward:', total_reward)
