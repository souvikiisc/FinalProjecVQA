from util_module import *

def image_attention(image_feat, question_feat, scope_name="image_attention"):

    with tf.name_scope(scope_name) as scope:

        K = image_feat.get_shape().as_list()[1]

        dim = image_feat.shape[2]
        hid_dim = question_feat.get_shape().as_list()[1]
        # print (K, hid_dim)
        W = Dense(hid_dim, use_bias=True)
        W_prime = Dense(hid_dim, use_bias=True)
        weights = Dense(1, use_bias=True)
        question_feat = tf.reshape(tf.tile(question_feat, [1,K]), shape=[-1, K, hid_dim])
        # print ("Q",question_feat.shape)
        # print ("I", image_feat.shape)
        concatenated = tf.concat([image_feat, question_feat], axis = 2)

        concatenated = gated_tanh(concatenated, W, W_prime, "image_gate")

        a_i = weights(concatenated)

        a_i = tf.nn.softmax(tf.squeeze(a_i), name="attention_weights")

        a_i = tf.reshape(a_i, shape=(-1, K, 1), name="reshape")

        image_att = tf.reduce_sum(tf.multiply(a_i, concatenated), axis=1)


        return image_att

def image_attention_2(image_feat, question_feat, scope_name="image_attention"):

    with tf.name_scope(scope_name) as scope:

        K = image_feat.get_shape().as_list()[1]
        dim = image_feat.shape[2]
        hid_dim = question_feat.get_shape().as_list()[1]
        # print (K, hid_dim)
        # W = Dense(hid_dim, use_bias=True)
        # W_prime = Dense(hid_dim, use_bias=True)
        weights = Dense(1, use_bias=True)
        question_feat = tf.reshape(tf.tile(question_feat, [1,K]), shape=[-1, K, hid_dim])
        # print ("Q",question_feat.shape)
        # print ("I", image_feat.shape)
        concatenated = tf.concat([image_feat, question_feat], axis = 2)

        # concatenated = relu_layernorm(concatenated, hid_dim, "image_gate")

        a_i = weights(concatenated)

        a_i = tf.nn.softmax(tf.squeeze(a_i), name="attention_weights")

        a_i = tf.reshape(a_i, shape=(-1, K, 1), name="reshape")

        image_att = tf.reduce_sum(tf.multiply(a_i, concatenated), axis=1)


        return image_att




    







