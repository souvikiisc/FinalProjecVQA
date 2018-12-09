from question_embedding import *
from image_attention import *
def ov_model(embeddings, out_dims):

    q_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="question_input")
    q_inputs_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name="input_length")
    image_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 36, 2048], name="image_feat")
    word_embeddings = tf.Variable(embeddings, dtype=tf.float32, name="word_embeddings")
    # print word_embeddings.shape

    q_outputs, q_state = question_embeddings(q_inputs, q_inputs_length, word_embeddings)

    q_state = tf.concat([q_state[0], q_state[1]], 1)

    # print ("Q", q_state.shape)

    image_att = image_attention_2(image_inputs, q_state)
    hid_dim = q_state.shape[1]

    # W_q = Dense(hid_dim, use_bias=True, name="W_q")
    # W_q_prime = Dense(hid_dim, use_bias=True, name="W_q_prime")
    q_gated = relu_layernorm(q_state, hid_dim, "q_gated")

    # W_img = Dense(hid_dim, use_bias=True, name="W_img")
    # W_img_prime = Dense(hid_dim, use_bias=True, name="W_img_prime")
    img_att_gated = relu_layernorm(image_att, hid_dim, "img_gated")

    final_feat = tf.multiply(q_gated, img_att_gated)


    # W_final = Dense(hid_dim, use_bias=True, name="W_final")
    # W_final_prime = Dense(hid_dim, use_bias=True, name="W_final_prime")

    final_logit = relu_layernorm(final_feat, out_dims,"final_gated")


    W_clf = Dense(out_dims, use_bias=False,name="linear_clf")

    final_logit = W_clf(final_logit)


    return final_logit, q_inputs, q_inputs_length, image_inputs

