from util_module import *
from tensorflow.contrib.rnn import GRUCell
def question_embeddings(inputs, inputs_length, word_embeddings, gru_hidden_units=256):

    input_embed = tf.nn.embedding_lookup(word_embeddings, inputs)

    cell_fw = GRUCell(gru_hidden_units)
    cell_bw = GRUCell(gru_hidden_units)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,inputs=input_embed, sequence_length=inputs_length,
                                        dtype=tf.float32, time_major=False, scope="Q_lstm")
    return outputs, states