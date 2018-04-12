import numpy as np
import tensorflow as tf

from baselines.a2c.utils import conv, fc, conv_to_fc, seq_to_batch, sample
from baselines.common.DNC.MaskedRNN import MaskedDNC, MaskedRNNInput
from baselines.common.distributions import make_pdtype


class DncPolicyPredictive2(object):
    """
    This policy predicts the future observation that the hidden vector behind the conv layers will have.
        obs_description and obs_prediction belong to this part.
    It also attempts to improve the obs_description by adding some form of auto-encoder, that tries to recreate
    the values of the previous layer, so that by compressing, there is little to no information loss.
        obs_encoded and obs_decoded belong to this part.
    """
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, args={}, reuse=False):
        nenv = nbatch // nsteps
        nstack = args.get('nstack', 1)  # required only for watching it play (set to 4), not for training
        nlstm = args.get('nlstm', nlstm)
        if len(ob_space.shape) == 3:
            nh, nw, nc = ob_space.shape
            ob_shape = (nbatch, nh, nw, nc*nstack)
            uses_conv = True
        else:
            nc = ob_space.shape[-1]
            ob_shape = (nbatch, nc*nstack)
            uses_conv = False
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)

        access_config = {
            "memory_size": args.get('memory_size', 16),
            "word_size": args.get('word_size', 16),
            "num_reads": args.get('num_read_heads', 1),
            "num_writes": args.get('num_write_heads', 1),
        }
        controller_config = {
            "hidden_size": nlstm,
        }

        with tf.variable_scope("model", reuse=reuse):
            if uses_conv:
                h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
                h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
                h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
                obs_encoded = conv_to_fc(h3)
            else:
                h3 = conv_to_fc(tf.cast(X, tf.float32))
                obs_encoded = conv_to_fc(X)
            obs_description = fc(obs_encoded, 'fc1', nh=args.get('nobsdesc', 512), init_scale=np.sqrt(2))
            xs = tf.reshape(obs_description, [nenv, nsteps, -1])
            ms = tf.reshape(M, [nenv, nsteps, 1])
            dnc_model = MaskedDNC(access_config, controller_config, nlstm, args.get('clip_value', 200000))
            ms = tf.subtract(tf.ones_like(ms), ms, name='mask_sub')  # previously 1 means episode is over, now 1 means it continues
            S = dnc_model.initial_state(nenv)  # states
            dnc_input = MaskedRNNInput(
                input=xs,
                mask=ms
            )
            h5, snew = tf.nn.dynamic_rnn(
                cell=dnc_model,
                inputs=dnc_input,
                time_major=False,
                initial_state=S)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

            v0 = vf[:, 0]
            a0 = sample(pi)
            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)
            neglogp0 = self.pd.neglogp(a0)

            # decoding the obs description, auto-encoder style, hopefully matching the encoded description
            obs_decoded = fc(obs_description, 'obs_decoded', nh=obs_encoded.shape[-1], init_scale=np.sqrt(2), act=tf.nn.tanh)

            # predicting outcomes
            a0_assigned = tf.Variable(initial_value=np.zeros(a0.shape, dtype=np.int64),
                                      trainable=False, name="a0_assigned")
            obs_desc_assigned = tf.Variable(initial_value=np.zeros(obs_description.shape, dtype=np.float32),
                                            trainable=False, name="obs_desc_assigned")
            assign_a0 = a0_assigned.assign(a0)
            assign_obs_desc = obs_desc_assigned.assign(obs_description)
            with tf.control_dependencies([assign_a0, assign_obs_desc]):
                onehot_action = tf.cast(tf.one_hot(a0_assigned, nact), tf.float32)
                pred_delta_in = tf.concat([onehot_action, h5], 1)
                pred_delta_h = fc(pred_delta_in, 'pred_delta_h', args.get('nobsdesch', 512), act=tf.nn.relu)
                pred_delta = fc(pred_delta_h, 'pred_delta', args.get('nobsdesc', 512), act=tf.nn.relu)
                obs_prediction = tf.add(obs_desc_assigned, pred_delta)

        def step(ob, state, mask):
            a, v, s, nlp, obs_desc, obs_pred, obs_enc, obs_dec\
                = sess.run([a0, v0, snew, neglogp0, obs_description,
                            obs_prediction, obs_encoded, obs_decoded],
                           {X:ob, S:state, M:mask})
            return a, v, s, nlp, obs_desc, obs_pred, obs_enc, obs_dec

        def value_and_obsdesc(ob, state, mask):
            return sess.run([v0, obs_description], {X:ob, S:state, M:mask})

        def initial_state():
            return sess.run(S)

        def encoding_shape():
            return obs_encoded.shape

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value_and_obsdesc = value_and_obsdesc
        self.initial_state = initial_state
        self.encoding_shape = encoding_shape


class DncPolicyPredictive3(object):
    """
    This policy predicts the future observation that the hidden vector behind the conv layers will have.
        obs_description and obs_prediction belong to this part.
    This policy does not have the auto-encoder part and uses a single layer to predict the observation delta.
    """
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, args={}, reuse=False):
        nenv = nbatch // nsteps
        nstack = args.get('nstack', 1)  # required only for watching it play (set to 4), not for training
        nlstm = args.get('nlstm', nlstm)
        if len(ob_space.shape) == 3:
            nh, nw, nc = ob_space.shape
            ob_shape = (nbatch, nh, nw, nc*nstack)
            uses_conv = True
        else:
            nc = ob_space.shape[-1]
            ob_shape = (nbatch, nc*nstack)
            uses_conv = False
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)

        access_config = {
            "memory_size": args.get('memory_size', 16),
            "word_size": args.get('word_size', 16),
            "num_reads": args.get('num_read_heads', 1),
            "num_writes": args.get('num_write_heads', 1),
        }
        controller_config = {
            "hidden_size": nlstm,
        }

        with tf.variable_scope("model", reuse=reuse):
            if uses_conv:
                h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
                h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
                h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
                obs_encoded = conv_to_fc(h3)
            else:
                h3 = conv_to_fc(tf.cast(X, tf.float32))
                obs_encoded = conv_to_fc(X)
            obs_description = fc(obs_encoded, 'fc1', nh=args.get('nobsdesc', 512), init_scale=np.sqrt(2))
            xs = tf.reshape(obs_description, [nenv, nsteps, -1])
            ms = tf.reshape(M, [nenv, nsteps, 1])
            dnc_model = MaskedDNC(access_config, controller_config, nlstm, args.get('clip_value', 200000))
            ms = tf.subtract(tf.ones_like(ms), ms, name='mask_sub')  # previously 1 means episode is over, now 1 means it continues
            S = dnc_model.initial_state(nenv)  # states
            dnc_input = MaskedRNNInput(
                input=xs,
                mask=ms
            )
            h5, snew = tf.nn.dynamic_rnn(
                cell=dnc_model,
                inputs=dnc_input,
                time_major=False,
                initial_state=S)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

            v0 = vf[:, 0]
            a0 = sample(pi)
            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)
            neglogp0 = self.pd.neglogp(a0)

            # predicting outcomes
            a0_assigned = tf.Variable(initial_value=np.zeros(a0.shape, dtype=np.int64),
                                      trainable=False, name="a0_assigned")
            obs_desc_assigned = tf.Variable(initial_value=np.zeros(obs_description.shape, dtype=np.float32),
                                            trainable=False, name="obs_desc_assigned")
            assign_a0 = a0_assigned.assign(a0)
            assign_obs_desc = obs_desc_assigned.assign(obs_description)
            with tf.control_dependencies([assign_a0, assign_obs_desc]):
                onehot_action = tf.cast(tf.one_hot(a0_assigned, nact), tf.float32)
                pred_delta_in = tf.concat([onehot_action, h5], 1)
                pred_delta = fc(pred_delta_in, 'pred_delta', args.get('nobsdesc', 512), act=tf.nn.relu)
                obs_prediction = tf.add(obs_desc_assigned, pred_delta)

        def step(ob, state, mask):
            a, v, s, nlp, obs_desc, obs_pred\
                = sess.run([a0, v0, snew, neglogp0, obs_description, obs_prediction],
                           {X:ob, S:state, M:mask})
            return a, v, s, nlp, obs_desc, obs_pred

        def value_and_obsdesc(ob, state, mask):
            return sess.run([v0, obs_description], {X:ob, S:state, M:mask})

        def initial_state():
            return sess.run(S)

        def encoding_shape():
            return obs_encoded.shape

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value_and_obsdesc = value_and_obsdesc
        self.initial_state = initial_state
        self.encoding_shape = encoding_shape
