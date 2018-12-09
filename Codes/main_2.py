from model_2 import *
from dataset import *
from tqdm import tqdm

EPOCHS = 25
print ("load data")
dictionary = Dictionary.load_from_file('data/dictionary.pkl')
train_dset = VQAFeatureDataset('train', dictionary)
eval_dset = VQAFeatureDataset('val', dictionary)
embeddings = np.load('data/glove6b_init_300d.npy')
print(embeddings.shape)
t = np.zeros((embeddings.shape[0]+1, embeddings.shape[1]))
t[0:embeddings.shape[0]] = embeddings[:]
t[embeddings.shape[0]] = np.random.uniform(-1,1, 300)
embeddings = t

print ("done")
out_dims = train_dset.get_ans_dims()
def loss_score():
    soft_labels = tf.placeholder(shape=(None, out_dims), dtype=tf.float32, name="soft_labels")
    final_logits, q_input, q_inputs_length, image_inputs = ov_model(embeddings, out_dims)
    pred = tf.nn.sigmoid(final_logits)
    global_step = tf.Variable(0, trainable=False)
    init_lr = 0.0001
    lr = tf.train.exponential_decay(init_lr, global_step, 2000,0.94)
    b_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = soft_labels, logits = final_logits), axis=1))

    # l_loss = tf.reduce_mean(tf.reduce_sum(-tf.multiply(soft_labels, tf.log(pred)) - tf.multiply(1-soft_labels, tf.log(1-pred)), axis=1),name="loss")

    classify_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    classify_gradients = classify_optimizer.compute_gradients(b_loss)
    _capped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in classify_gradients if
                         grad is not None]
    train_step = classify_optimizer.apply_gradients(_capped_gradients, global_step= global_step)
    # train_step = classify_optimizer.minimize(l_loss)

    max_i = tf.argmax(final_logits, axis = 1, name='max_indices')
    one_hot = tf.one_hot(max_i, depth= out_dims)

    # one_hot = tf.scatter_update(one_hot, max_i, 1.0)
    final_scores = tf.reduce_sum(tf.multiply(one_hot, soft_labels))
    # print(final_scores.shape)

    return b_loss, max_i, final_scores, q_input, q_inputs_length, image_inputs, soft_labels, train_step


print ("define graph")

b_loss, max_i, final_scores, q_input, q_inputs_length, image_inputs, soft_labels, opt = loss_score()

print ("done")
pth = "vqa_save/model.ckpt"
init = tf.global_variables_initializer()

config=tf.ConfigProto()
config.gpu_options.allow_growth=True

saver = tf.train.Saver()

with tf.Session(config= config) as sess:

    sess.run(init)


    best_score = 0.0
    best_epoch = 0
    for epoch in tqdm(range(EPOCHS)):

        train_loss = 0.0
        train_score = 0.0
        count = 0
        acc = 0.0
        for batch in train_dset.get_batch(512):
            np.random.shuffle(batch)
            q_batch = np.array(list(batch[:,0]))
            f_batch =  np.array(list(batch[:,1]))
            l_batch =  np.array(list(batch[:,2]))
            qlen_batch =  np.array(list(batch[:,3]))

            # print (q_batch)
            # print(q_batch.shape)
            #
            # break
            _ ,loss, scores, ind = sess.run([opt, b_loss, final_scores, max_i], {q_input:q_batch, q_inputs_length: qlen_batch, image_inputs:f_batch, soft_labels:l_batch})


            a = np.eye(out_dims)[ind]
            a = np.sum(np.multiply(a, l_batch), axis=1)
            a[a>0] = 1.0
            # print (a)
            acc += np.sum(a)

            train_score += scores
            train_loss += loss*(len(batch))
            count += len(batch)

        print ("loss: ", train_loss/count)
        print ("avg score", 100*train_score/count)
        print ("accuracy", 100*acc/count)

        val_loss = 0.0
        val_score = 0.0
        count = 0.0
        acc = 0.0
        for batch in eval_dset.get_batch(256):
            np.random.shuffle(batch)
            q_batch = np.array(list(batch[:, 0]))
            f_batch = np.array(list(batch[:, 1]))
            l_batch = np.array(list(batch[:, 2]))
            qlen_batch = np.array(list(batch[:, 3]))
            loss, scores, ind = sess.run([b_loss, final_scores, max_i],
                                       {q_input: q_batch, q_inputs_length: qlen_batch, image_inputs: f_batch,
                                        soft_labels: l_batch})

            a = np.eye(out_dims)[ind]
            a = np.sum(np.multiply(a, l_batch), axis=1)
            a[a > 0] = 1.0
            # print (a)
            acc += np.sum(a)

            val_score += scores
            val_loss += loss*len(batch)
            count += len(batch)

        print("loss: ", val_loss / count)
        print("avg score", 100 * val_score / count)
        print ("accuracy", 100 * acc / count)


        if(val_score/count > best_score):
            best_score = val_score/count
            best_epoch = epoch
            saver.save(sess, save_path=pth)


    print (best_epoch, best_score)





