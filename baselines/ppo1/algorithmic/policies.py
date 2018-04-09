import baselines.common.tf_util as U
import tensorflow as tf
from baselines.common.distributions import make_pdtype


class AlgorithmicFfPolicy(object):
    recurrent = False

    def __init__(self, name, ob_space, ac_space, args={}, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, args, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, args, kind):
        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        # by default, algorithmic tasks have only one number as observation...
        self.ob = U.get_placeholder(name="ob", dtype=tf.uint8, shape=[sequence_length, 1])
        x = tf.one_hot(self.ob, ob_space.n)
        x = tf.squeeze(x, axis=1)

        h = tf.layers.dense(x, args.get('hidden', 20), name='h', kernel_initializer=U.normc_initializer(0.01))
        logits = tf.layers.dense(h, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=U.normc_initializer(1.0))[:,0]

        self.state_in = []
        self.state_out = []

        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self.ac = self.pd.sample()
        self._act = U.function([self.stochastic, self.ob], [self.ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, [ob])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

