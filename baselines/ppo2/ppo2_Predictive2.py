import time
from collections import deque

import joblib
import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.a2c.utils import make_path, mse
from baselines.common import explained_variance, set_global_seeds
from baselines.common.DNC import MaskedRNN
from baselines.common.DNC.DNCVisualizedPlayer import DNCVisualizedPlayer


class Model(object):
    def __init__(self, *, policy, policy_args, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, pred_coef, pred_reward_max, dec_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, args=policy_args, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, args=policy_args, reuse=True)

        A = train_model.pdtype.sample_placeholder([nbatch_train])
        V = tf.placeholder(tf.float32, [nbatch_train], name='V')
        R = tf.placeholder(tf.float32, [nbatch_train], name='R')
        M = tf.placeholder(tf.float32, [nbatch_train], name='M')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [nbatch_train])
        OLDVPRED = tf.placeholder(tf.float32, [nbatch_train])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        cm = tf.subtract(tf.ones_like(M), M)  # with this, 1 means continue, 0 means stop -> use to normalize rewards

        # this will cause the agent to experience intrinsic reward, if it is bad at predicting the next state
        # thus it will explore these states more
        OBS_DESC = tf.placeholder(tf.float32, [nbatch_train, policy_args.get('nobsdesc')], name='OBS_DESC')
        OBS_PRED = tf.placeholder(tf.float32, [nbatch_train, policy_args.get('nobsdesc')], name='OBS_PRED')
        pred_losses = tf.reduce_mean(mse(OBS_DESC, OBS_PRED), axis=1)
        pred_loss = tf.reduce_mean(pred_losses)

        cr = tf.add(R, tf.clip_by_value(pred_losses, 0, pred_reward_max))
        cr = cr * cm
        adv = cr - V
        std, var = tf.nn.moments(adv, [0])
        mean = tf.reduce_mean(adv)
        adv = (adv - mean) / (std + 1e-8)

        # difference of encoded observation (right after conv layers) and decoded (out of reduced information)
        # this is to produce a useful information compression
        self.encoding_shape_train = [nbatch_train, act_model.encoding_shape()[-1]]
        OBS_EN = tf.placeholder(tf.float32, self.encoding_shape_train, name='OBS_EN')
        OBS_DEC = tf.placeholder(tf.float32, self.encoding_shape_train, name='OBS_DEC')
        dec_loss = tf.reduce_mean(mse(OBS_EN, OBS_DEC))

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -adv * ratio
        pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + pred_loss*pred_coef + dec_loss*dec_coef

        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs,
                  descriptions, predictions, encoded, decoded, states=None):
            td_map = {train_model.X: obs, A: actions, V: values, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values,
                      OBS_DESC: descriptions, OBS_PRED: predictions, M: masks,
                      OBS_EN: encoded, OBS_DEC: decoded}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run([pg_loss, vf_loss, entropy, approxkl, clipfrac, dec_loss, pred_loss, _train], td_map)[:-1]

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

        self.loss_names = ['loss_policy', 'loss_value', 'policy_entropy', 'approxkl', 'clipfrac', 'loss_decoding', 'loss_prediction']
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value_and_obsdesc = act_model.value_and_obsdesc
        self.initial_state = act_model.initial_state()
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101


class Runner(object):

    def __init__(self, *, env, model, description_size, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.batch_desc_shape = (nenv*nsteps, description_size)

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_descs, mb_preds, mb_encs, mb_decs = [],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs,\
                obs_desc, obs_pred, obs_enc, obs_dec = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_descs.append(obs_desc)
            mb_preds.append(obs_pred)
            mb_encs.append(obs_enc)
            mb_decs.append(obs_dec)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values, last_obsdesc = self.model.value_and_obsdesc(self.obs, self.states, self.dones)
        mb_descs.append(last_obsdesc)  # removing first obs, adding one more, now obs[i] supposely matches pred[i]
        mb_descs.pop(0)
        mb_descs = np.asarray(mb_descs, dtype=np.float32)
        mb_preds = np.asarray(mb_preds, dtype=np.float32)
        mb_encs = np.asarray(mb_encs, dtype=np.float32)
        mb_decs = np.asarray(mb_decs, dtype=np.float32)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
                            mb_descs, mb_preds, mb_encs, mb_decs)), mb_states, epinfos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, policy_args, env, nsteps, total_timesteps, ent_coef, lr,
            pred_coef=0.25, pred_reward_max=0.5, dec_coef=0.25,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_path='', save_name='model'):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    model = Model(policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                  nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                  pred_coef=pred_coef, pred_reward_max=pred_reward_max, dec_coef=dec_coef,
                  max_grad_norm=max_grad_norm)
    model.load(save_path, save_name)
    runner = Runner(env=env, model=model, description_size=policy_args.get('nobsdesc'),
                    nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs,\
            descriptions, predictions, encoded, decoded,\
            states, epinfos = runner.run()  # pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        assert nenvs % nminibatches == 0
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        envsperbatch = nbatch_train // nsteps
        for _ in range(noptepochs):
            np.random.shuffle(envinds)
            for start in range(0, nenvs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs,
                                                      descriptions, predictions, encoded, decoded))
                mbstates = MaskedRNN.MaskedDNC.state_subset(states, mbenvinds)
                # mbstates = states[mbenvinds]  # TODO make dnc/other work both work
                mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            nseconds = time.time()-tfirststart
            rem_time = (nseconds * nupdates / update) - nseconds
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
            model.save(save_path, save_name)
            logger.info("Time since start: \t%.2fs" % nseconds)
            logger.info("ETA: \t\t\t\t%.2fs" % rem_time)
    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


# TODO will currently not work
def play(policy, policy_args, env, seed, nep=5, save_path='', save_name='model'):
    tf.reset_default_graph()
    tf.Session().__enter__()
    set_global_seeds(seed)

    ob_space = env.observation_space
    ac_space = env.action_space
    nstack = 4
    policy_args['nstack'] = nstack
    model = Model(policy=policy, policy_args=policy_args, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
                  nbatch_train=1, nsteps=1, ent_coef=0.05, vf_coef=0.5,
                  max_grad_norm=0.5)
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
            observations = np.zeros((1, nc*nstack), dtype=np.uint8)

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
                actions, values, states, _ = model.step(observations, states, [done])
                new_obs, reward, done, info = env.step(actions[0])
                observations = update_obs(observations, new_obs)
                episode_reward += reward
                env.render()
            print('Episode reward:', episode_reward)
            total_reward += episode_reward
        print('Done! Total reward:', total_reward)
