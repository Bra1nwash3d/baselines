import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.DNC.MaskedRNN import MaskedDNC2 as MaskedDNC, MaskedRNNInput


class DncMbPolicy(object):
    # a dnc policy for the moving blocks env, which has a single vector as observation

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, args, reuse=False):
        nenv = nbatch // nsteps
        X = tf.placeholder(tf.uint8, shape=(nbatch, ob_space.n), name='X')  # obs
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
            xs = tf.one_hot(X, ob_space.depth)
            xs = tf.contrib.layers.flatten(xs)
            xs = tf.reshape(xs, [nenv, nsteps, -1])
            ms = tf.reshape(M, [nenv, nsteps, -1])
            dnc_model = MaskedDNC(access_config, controller_config,
                                  args.get('num_dnc_out', 256), args.get('clip_value', 200000))
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
            h5 = tf.reshape(h5, [nenv*nsteps, -1])
            pi = fc(h5, 'pi', ac_space.n)
            vf = fc(h5, 'v', 1)

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
