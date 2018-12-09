from model_2 import *
from dataset import *
# from tqdm import tqdm
import argparse as ap
import os
import pickle

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)  
EPOCHS = 27

parser = ap.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, help='Batch Size')
parser.add_argument('--cont', action='store_true', help='Continue Training')
parser.add_argument('--decay', type=float, default=0.9, help='Decay Rate')
parser.add_argument('--epochs', type=int, default=3, help='Number of Epochs')
parser.add_argument('--gpu', type=int, default=0, help='Select GPU')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--model-tag', default='XVECTOR', help='Model Tag')
parser.add_argument('--num-features', type=int, default=23, help='Number of MFCC Co-efficients')
parser.add_argument('--ps', help='Parameter Server(s)')
parser.add_argument('--save', default='model_save', help='Save Location')
parser.add_argument('--steps', type=int, default=400000, help='Total global steps')
parser.add_argument('--type', default='ps', help='Instance Type')
parser.add_argument('--task-index', type=int, default=0, help='Task Index')
parser.add_argument('--workers', help='Worker Nodes')
args = parser.parse_args()

cwd = os.getcwd()
model_loc = os.path.join(cwd, args.save)
make_directory(model_loc)
ps = args.ps.split(',')
workers = args.workers.split(',')
num_workers = len(workers)
batch_size = 512*num_workers

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

cluster = tf.train.ClusterSpec({"ps": ps, "worker": workers})
server = tf.train.Server(cluster, job_name=args.type, task_index=args.task_index)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def loss_score():
        soft_labels = tf.placeholder(shape=(None, out_dims), dtype=tf.float32, name="soft_labels")
        final_logits, q_input, q_inputs_length, image_inputs = ov_model(embeddings, out_dims)
        # pred = tf.nn.sigmoid(final_logits)
        global_step = tf.contrib.framework.get_or_create_global_step()
        init_lr = 0.0001
        lr = tf.train.exponential_decay(init_lr, global_step, 2000,0.94)
        b_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = soft_labels, logits = final_logits), axis=1))

        # l_loss = tf.reduce_mean(tf.reduce_sum(-tf.multiply(soft_labels, tf.log(pred)) - tf.multiply(1-soft_labels, tf.log(1-pred)), axis=1),name="loss")

        classify_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        rep_op = tf.contrib.opt.ModelAverageOptimizer(classify_optimizer, num_worker=num_workers, is_chief = (args.task_index==0))
        classify_gradients = rep_op.compute_gradients(b_loss)
        _capped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in classify_gradients if
                            grad is not None]
        train_step = rep_op.apply_gradients(_capped_gradients, global_step= global_step)
        # train_step = classify_optimizer.minimize(l_loss)

        max_i = tf.argmax(final_logits, axis = 1, name='max_indices')
        one_hot = tf.one_hot(max_i, depth= out_dims)

        # one_hot = tf.scatter_update(one_hot, max_i, 1.0)
        final_scores = tf.reduce_sum(tf.multiply(one_hot, soft_labels))
        # print(final_scores.shape)

        return b_loss, max_i, final_scores, q_input, q_inputs_length, image_inputs, soft_labels, train_step, rep_op

if args.type == 'ps':
    print('Running on {} as parameter server.'.format(ps[args.task_index]))
    server.join()
elif args.type == 'worker':
    print('Running on {} as worker node.'.format(workers[args.task_index]))

    train = dict()
    val = dict()

    train_file = "trainfile"+str(args.task_index)+".pkl"
    val_file = "valfile"+str(args.task_index)+".pkl"
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
    
    print ("define graph")
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:{:d}".format(args.task_index),
                                                  cluster=cluster)):
        b_loss, max_i, final_scores, q_input, q_inputs_length, image_inputs, soft_labels, opt, rep_op = loss_score()

        sync_replicas_hook = rep_op.make_session_run_hook()

    print ("done")
    hooks = [sync_replicas_hook,tf.train.StopAtStepHook(last_step=args.steps)]
    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(args.task_index == 0),checkpoint_dir=model_loc, hooks=hooks, config=config) as mon_sess:
        
        best_score = 0.0
        best_epoch = 0
        epoch = 0
        while not mon_sess.should_stop() and epoch<EPOCHS:
            epoch += 1 
            print(epoch)
            train_loss = 0.0
            train_score = 0.0
            count = 0
            acc = 0.0
            for batch in train_dset.get_batch(batch_size):
                # np.random.shuffle(batch)
                
                q_batch = np.array(list(batch[:,0]))
                f_batch =  np.array(list(batch[:,1]))
                l_batch =  np.array(list(batch[:,2]))
                qlen_batch =  np.array(list(batch[:,3]))

                curr_index = np.arange(args.task_index, len(batch), num_workers)
                np.random.shuffle(curr_index)
                print(len(curr_index))
                q_batch = q_batch[curr_index]
                f_batch = f_batch[curr_index]
                l_batch = l_batch[curr_index]
                qlen_batch = qlen_batch[curr_index]

                # print (q_batch)
                # print(q_batch.shape)
                #
                # break
                _ ,loss, scores, ind = mon_sess.run([opt, b_loss, final_scores, max_i], {q_input:q_batch, q_inputs_length: qlen_batch, image_inputs:f_batch, soft_labels:l_batch})


                a = np.eye(out_dims)[ind]
                a = np.sum(np.multiply(a, l_batch), axis=1)
                a[a>0] = 1.0
                # print (a)
                acc += np.sum(a)

                train_score += scores
                train_loss += loss*(len(curr_index))
                count += len(curr_index)

            print ("loss: ", train_loss/count)
            print ("avg score", 100*train_score/count)
            print ("accuracy", 100*acc/count)

            p1 = train.get("loss", dict())
            p1[epoch] = train_loss/count
            train["loss"] = p1

            p2 = train.get("score", dict())
            p2[epoch] = train_score/count
            train["score"] = p2


            if args.task_index == 0:

                val_loss = 0.0
                val_score = 0.0
                count = 0.0
                acc = 0.0
                for batch in eval_dset.get_batch(512):
                    np.random.shuffle(batch)
                    q_batch = np.array(list(batch[:, 0]))
                    f_batch = np.array(list(batch[:, 1]))
                    l_batch = np.array(list(batch[:, 2]))
                    qlen_batch = np.array(list(batch[:, 3]))
                    loss, scores, ind = mon_sess.run([b_loss, final_scores, max_i],
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

                p1 = val.get("loss", dict())
                p1[epoch] = val_loss/count
                val["loss"] = p1

                p2 = val.get("score", dict())
                p2[epoch] = val_score/count
                val["score"] = p2

                if(val_score/count > best_score):
                    best_score = val_score/count
                    best_epoch = epoch



        f = open(train_file, "wb")
        pickle.dump(train, f)
        f.close()

        f = open(val_file, "wb")
        pickle.dump(val, f)
        f.close()
        print (best_epoch, best_score)





