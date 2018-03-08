import tensorflow as tf
from dnc import DNC, DNCState
from access import AccessState
from addressing import TemporalLinkageState
import collections


MaskedRNNInput = collections.namedtuple('MaskedRNNInput', ('input', 'mask'))


class MaskedDNC(DNC):
    def _build(self, inputs, prev_state):
        # first resets batch-specific state parts to zero (depending on mask), then performs original DNC action
        batch_size = prev_state.access_output.shape[0]

        split_mask = tf.split(inputs.mask, batch_size)
        split_access_output = tf.split(prev_state.access_output, batch_size)
        split_memory = tf.split(prev_state.access_state.memory, batch_size)
        split_read_weights = tf.split(prev_state.access_state.read_weights, batch_size)
        split_write_weights = tf.split(prev_state.access_state.write_weights, batch_size)
        split_link = tf.split(prev_state.access_state.linkage.link, batch_size)
        split_pw = tf.split(prev_state.access_state.linkage.precedence_weights, batch_size)
        split_usage = tf.split(prev_state.access_state.usage, batch_size)
        controller_states0 = tf.split(prev_state.controller_state[0], batch_size)
        controller_states1 = tf.split(prev_state.controller_state[1], batch_size)

        for i in range(batch_size):
            split_access_output[i] *= split_mask[i]
            split_memory[i] *= split_mask[i]
            split_read_weights[i] *= split_mask[i]
            split_write_weights[i] *= split_mask[i]
            split_link[i] *= split_mask[i]
            split_pw[i] *= split_mask[i]
            split_usage[i] *= split_mask[i]
            controller_states0[i] *= split_mask[i]
            controller_states1[i] *= split_mask[i]

        m_state = DNCState(tf.concat(split_access_output, 0),
                           AccessState(
                               tf.concat(split_memory, 0),
                               tf.concat(split_read_weights, 0),
                               tf.concat(split_write_weights, 0),
                               TemporalLinkageState(
                                   tf.concat(split_link, 0),
                                   tf.concat(split_pw,  0)
                               ),
                               tf.concat(split_usage,  0),
                           ),
                           (tf.concat(controller_states0,  0), tf.concat(controller_states1,  0)))
        return DNC._build(self, inputs.input, m_state)

    @staticmethod
    def state_subset(state, inds):
        return DNCState(state.access_output[inds],
                        AccessState(
                            state.access_state.memory[inds],
                            state.access_state.read_weights[inds],
                            state.access_state.write_weights[inds],
                            TemporalLinkageState(
                                state.access_state.linkage.link[inds],
                                state.access_state.linkage.precedence_weights[inds]
                            ),
                            state.access_state.usage[inds],
                        ),
                        (state.controller_state[0][inds], state.controller_state[1][inds]))


class MaskedDNC2(DNC):
    def _build(self, inputs, prev_state):
        # first resets batch-specific state parts to zero (depending on mask), then performs original DNC action
        batch_size = inputs.mask.get_shape().as_list()[0]
        mask2 = tf.reshape(inputs.mask, [batch_size, 1])
        mask3 = tf.reshape(inputs.mask, [batch_size, 1, 1])
        mask4 = tf.reshape(inputs.mask, [batch_size, 1, 1, 1])

        m_state = DNCState(access_output=tf.multiply(prev_state.access_output, mask3),
                           access_state=AccessState(
                               tf.multiply(prev_state.access_state.memory, mask3),
                               tf.multiply(prev_state.access_state.read_weights, mask3),
                               tf.multiply(prev_state.access_state.write_weights, mask3),
                               TemporalLinkageState(
                                   tf.multiply(prev_state.access_state.linkage.link, mask4),
                                   tf.multiply(prev_state.access_state.linkage.precedence_weights, mask3)
                               ),
                               tf.multiply(prev_state.access_state.usage, mask2),
                           ),
                           controller_state=(tf.multiply(prev_state.controller_state[0], mask2),
                                             tf.multiply(prev_state.controller_state[1], mask2),))
        return DNC._build(self, inputs.input, m_state)

    @staticmethod
    def state_subset(state, inds):
        return DNCState(state.access_output[inds],
                        AccessState(
                            state.access_state.memory[inds],
                            state.access_state.read_weights[inds],
                            state.access_state.write_weights[inds],
                            TemporalLinkageState(
                                state.access_state.linkage.link[inds],
                                state.access_state.linkage.precedence_weights[inds]
                            ),
                            state.access_state.usage[inds],
                        ),
                        (state.controller_state[0][inds], state.controller_state[1][inds]))


class MaskedLSTM(tf.nn.rnn_cell.LSTMCell):
    def __init__(self, num_units, **kwargs):
        super(MaskedLSTM, self).__init__(num_units=num_units, **kwargs)

    def call(self, inputs, state):
        c, h = state
        m_state = tf.nn.rnn_cell.LSTMStateTuple(c=tf.multiply(c, inputs.mask),
                                                h=tf.multiply(h, inputs.mask))
        out, out_state = tf.nn.rnn_cell.LSTMCell.call(self, inputs.input, m_state)
        return out, out_state

