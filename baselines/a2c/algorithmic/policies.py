import tensorflow as tf

from baselines.a2c.utils import fc, seq_to_batch, sample
from baselines.common.DNC.MaskedDNC import MaskedDNC, MaskedDNCInput


class DncPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, args, reuse=False):
        nlstm = args.get('nlstm', 64)
        nbatch = nenv*nsteps
        X = tf.placeholder(tf.uint8, shape=(nbatch, 1))  # obs
        nact = [ac.n for ac in ac_space.spaces]
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
            xs = tf.one_hot(X, ob_space.n)
            xs = tf.reshape(xs, [nenv, nsteps, -1])
            ms = tf.reshape(M, [nenv, nsteps, 1])
            # use nlstm as output size again, so we can add the fc layers like before
            dnc_model = MaskedDNC(access_config, controller_config, nlstm, args.get('clip_value', 200000))
            ms = tf.subtract(tf.ones_like(ms), ms, name='mask_sub')  # previously 1 means episode is over, now 1 means it continues
            S = dnc_model.initial_state(nenv)  # states
            dnc_input = MaskedDNCInput(
                input=xs,
                mask=ms
            )
            h5, snew = tf.nn.dynamic_rnn(
                cell=dnc_model,
                inputs=dnc_input,
                time_major=False,
                initial_state=S)
            h5 = seq_to_batch(h5)
            pi = [fc(h5, 'pi-'+str(i), nact[i], act=lambda x:x) for i in range(len(nact))]
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = [sample(p) for p in pi]

        def step(ob, state, mask):
            query = [v0, snew]
            query.extend(a0)
            v, s, *a = sess.run(query, {X:ob, S:state, M:mask})
            return a, v, s

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
