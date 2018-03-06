import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.DNC.MaskedDNC import MaskedDNC, MaskedDNCInput


class AlgorithmicDncPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, args, reuse=False):
        nenv = nbatch // nsteps
        X = tf.placeholder(tf.uint8, shape=(nbatch,), name='X')  # obs
        M = tf.placeholder(tf.float32, [nbatch], name='M')  # mask (done t-1)

        access_config = {
            "memory_size": args.get('memory_size', 16),
            "word_size": args.get('word_size', 16),
            "num_reads": args.get('num_read_heads', 1),
            "num_writes": args.get('num_write_heads', 1),
        }
        controller_config = {
            "hidden_size": args.get('num_controller_lstm', 256),
        }

        with tf.variable_scope("model", reuse=reuse):
            xs = tf.one_hot(X, ob_space.n)
            # xs = tf.convert_to_tensor(batch_to_seq(xs, nenv, nsteps))
            # ms = tf.convert_to_tensor(batch_to_seq(M, nenv, nsteps))
            xs = tf.reshape(xs, [nenv, nsteps, -1])
            ms = tf.reshape(M, [nenv, nsteps, -1])
            # use nlstm as output size again, so we can add the fc layers like before
            dnc_model = MaskedDNC(access_config,
                                  controller_config,
                                  args.get('num_dnc_out', 256),
                                  args.get('clip_value', 200000))
            ms = tf.subtract(tf.ones_like(ms), ms, name='mask_sub')  # previously 1 means episode is over, now 1 means it continues
            print('X', X)
            print('M', M)
            print('xs', xs)
            print('ms', ms)
            S = dnc_model.initial_state(nenv)  # states
            dnc_input = MaskedDNCInput(
                input=xs,
                mask=ms
            )
            h5, snew = tf.nn.dynamic_rnn(
                cell=dnc_model,
                inputs=dnc_input,
                time_major=False,  # TODO actually batch major?
                initial_state=S)
            # h5 = seq_to_batch(h5)
            h5 = tf.reshape(h5, [nenv*nsteps, -1])

            pi = fc(h5, 'pi', sum(ac_space.nvec))
            vf = fc(h5, 'v', 1)

            print('vf', vf)
            print('pi', pi)
            print('')

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        def initial_state():
            return sess.run(S)

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.initial_state = initial_state