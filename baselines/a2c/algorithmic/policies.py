import tensorflow as tf

from baselines.a2c.utils import fc, seq_to_batch, sample, batch_to_seq
from baselines.common.DNC.MaskedDNC import MaskedDNC, MaskedDNCInput
from collections import namedtuple


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


AC_State = namedtuple('AC_State', ('actor_state', 'critic_state'))


class DncPolicy2(object):
    # a DncPolicy with actor and critic in two separate DNCs, the critic does not have info about taken actions
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
            ms = tf.subtract(tf.ones_like(ms), ms, name='mask_sub')  # previously 1 means episode is over, now 1 means it continues

            dnc_actor_model = MaskedDNC(access_config, controller_config, nlstm, args.get('clip_value', 200000))
            dnc_critic_model = MaskedDNC(access_config, controller_config, nlstm, args.get('clip_value', 200000))
            S = AC_State(actor_state=dnc_actor_model.initial_state(nenv),
                         critic_state=dnc_critic_model.initial_state(nenv))
            actor_S = S.actor_state
            critic_S = S.critic_state

            # ACTOR
            dnc_actor_input = MaskedDNCInput(
                input=xs,
                mask=ms
            )
            actor_outputs, actor_new_state = tf.nn.dynamic_rnn(
                cell=dnc_actor_model,
                inputs=dnc_actor_input,
                time_major=False,
                initial_state=actor_S)
            h5a = seq_to_batch(actor_outputs)
            pi = [fc(h5a, 'pi-'+str(i), nact[i], act=lambda x:x) for i in range(len(nact))]
            a0 = [sample(p) for p in pi]

            # CRITIC
            dnc_actor_input = MaskedDNCInput(
                input=xs,
                mask=ms
            )
            critic_outputs, critic_new_state = tf.nn.dynamic_rnn(
                cell=dnc_critic_model,
                inputs=dnc_actor_input,
                time_major=False,
                initial_state=critic_S)
            h5c = seq_to_batch(critic_outputs)
            vf = fc(h5c, 'v', 1, act=lambda x:x)
            v0 = vf[:, 0]

        def step(ob, state, mask):
            query = [v0, actor_new_state, critic_new_state]
            query.extend(a0)
            a_s, c_s = state
            v, new_a_s, new_c_s, *a = sess.run(query, {X:ob, actor_S:a_s, critic_S:c_s, M:mask})
            return a, v, AC_State(actor_state=a_s, critic_state=c_s)

        def value(ob, state, mask):
            a_s, c_s = state
            return sess.run(v0, {X:ob, actor_S:a_s, critic_S:c_s, M:mask})

        def initial_state():
            a_s, c_s = sess.run([actor_S, critic_S])
            return AC_State(actor_state=a_s, critic_state=c_s)

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.initial_state = initial_state
