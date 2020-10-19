import importlib
import logging
import os
import sys
import time
import tensorflow as tf
from tensorboardX import SummaryWriter
import cadgmm.gmm_utils as gmm
from utils.preprocessing import *

RANDOM_SEED = 30
FREQ_PRINT = 5000  # print frequency image tensorboard [20]
METHOD = "inception"


def display_parameters(batch_size, starting_lr,
                       l1, l2, l3, label):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('Weights loss - l1:', l1, '; l2:', l2, '; l3:', l3)
    print('Anomalous label: ', label)


def display_progression_epoch(j, id_max):
    '''See epoch progression
       sys.stdout.write(" ")的本质是print(" ", end="")
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def create_logdir(model_name, dataset, K, v, KNN, l1, l2, l3):
    """ Directory to save training logs, weights, biases, etc."""
    return "train_logs/{}/{}_KNN{}_K{}_v{}_l1{}_l2{}_l3{}".format(model_name, dataset, KNN, K, v, l1, l2, l3)


def reconstruction_error(x, x_rec):
    return tf.norm(x - x_rec, axis=1)


def train_and_test(model_name, dataset, n_epochs, it_e_val, K, v, KNN, l1, l2, l3, label,
                   random_seed):
    """ Runs the CADGMM on the specified dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        n_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    logger = logging.getLogger("{}.train.{}.{}".format(model_name, dataset, label))

    # Import model and data
    model = importlib.import_module('{}.{}_utilities'.format(model_name, dataset))
    data = importlib.import_module("data.{}".format(dataset))

    # Parameters
    starting_lr = model.params["learning_rate"]
    batch_size = model.params["batch_size"]
    if n_epochs == -1: n_epochs = model.params["n_epochs"]
    if l1 == -1: l1 = model.params["l1"]
    if l2 == -1: l2 = model.params["l2"]
    if l3 == -1: l3 = model.params["l3"]
    if K == -1: K = model.params["K"]
    if KNN == -1: KNN = model.params["KNN"]

    x_pl = tf.placeholder(tf.float32, data.get_shape_input(), name='x_pl')
    adj = tf.sparse_placeholder(tf.float32, name='adj')

    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    logger.info('Building training graph...')

    logger.warning("The CADGMM is training with the following parameters:")
    display_parameters(batch_size, starting_lr, l1, l2, l3,
                       label)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    if model.params["is_image"]:
        enc = model.encoder
    else:
        enc = model.encoder
    dec = model.decoder
    feat_ex = model.feature_extractor
    est = model.estimator


    x_features = x_pl
    n_features = x_features.shape[1]

    with tf.variable_scope('encoder_model'):
        if model.params["is_image"]:
            z_c = enc(x_features, adj, n_samples=batch_size, is_training=is_training_pl)
        else:
            z_c = enc(x_features, adj, n_samples=batch_size, is_training=is_training_pl)
    with tf.variable_scope('decoder_model'):
        x_rec = dec(z_c, n_features, is_training=is_training_pl)

    with tf.variable_scope('feature_extractor_model'):
        x_flat = tf.layers.flatten(x_features)
        x_rec_flat = tf.layers.flatten(x_rec)
        z_r = feat_ex(x_flat, x_rec_flat)

    z = tf.concat([z_c, z_r], axis=1)
    with tf.variable_scope('estimator_model'):
        gamma = est(z, K, is_training=is_training_pl)

    with tf.variable_scope('gmm'):
        energy, penalty = gmm.compute_energy_and_penalty(z, gamma, is_training_pl)

    with tf.variable_scope('reg'):
        regularization = tf.reduce_sum(tf.pow(z_c, 2), axis=1)


    with tf.name_scope('loss_functions'):
        # reconstruction error
        rec_error = reconstruction_error(x_flat, x_rec_flat)
        loss_rec = tf.reduce_mean(rec_error)
        loss_reg = tf.reduce_mean(regularization)
        # probabilities to observe
        loss_energy = tf.reduce_mean(energy)

        # full loss
        full_loss = loss_rec + l1 * loss_energy + l2 * penalty + l3 * loss_reg

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.5, name='dis_optimizer')

        train_op = optimizer.minimize(full_loss, global_step=global_step)

        with tf.name_scope('predictions'):
            # Highest 20% are anomalous
            if dataset == "kdd":
                per = tf.contrib.distributions.percentile(energy, 80)
            else:
                per = tf.contrib.distributions.percentile(energy, 80)
            y_pred = tf.greater_equal(energy, per)

    with tf.name_scope('summary'):
        with tf.name_scope('loss_summary'):
            tf.summary.scalar('loss_rec', loss_rec, ['loss'])
            tf.summary.scalar('loss_reg', loss_rec, ['loss'])
            tf.summary.scalar('mean_energy', loss_energy, ['loss'])
            tf.summary.scalar('penalty', penalty, ['loss'])
            tf.summary.scalar('full_loss', full_loss, ['loss'])

        sum_op_loss = tf.summary.merge_all('loss')

    # Data
    logger.info('Data loading...')

    trainx, trainy = data.get_train(label, v)
    trainx_copy = trainx.copy()

    rng = np.random.RandomState(random_seed)
    nr_batches_train = int(trainx.shape[0] / batch_size)

    save_dir = create_logdir(model_name, dataset, K, v, KNN, l1, l2, l3)
    logdir = os.path.sep.join([save_dir, str(random_seed)])

    writer = SummaryWriter(logdir)
    logger.info('Start training...')

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    logger.info('Initialization done')
    train_batch = 0
    epoch = 0
    while epoch < n_epochs:

        lr = starting_lr
        begin = time.time()

        # construct randomly permuted minibatches
        trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling dataset
        trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
        train_l_rec, train_l_reg, train_l_energy, train_penalty, train_f_loss = [0, 0, 0, 0, 0]

        # training
        for t in range(nr_batches_train):

            display_progression_epoch(t, nr_batches_train)
            ran_from = t * batch_size
            ran_to = ran_from + batch_size
            train_data = trainx[ran_from:ran_to]
            if model.params["is_image"]:
                x_inps = train_data
                x_inps = np.reshape(x_inps, [batch_size, -1])
                train_adj_norm = construct_data(x_inps, k_neig=KNN)
            else:
                train_adj_norm = construct_data(train_data, k_neig=KNN)
            feed_dict = {x_pl: train_data,
                         adj: train_adj_norm,
                         is_training_pl: True,
                         learning_rate: lr}

            _, sm, step = sess.run([train_op,
                                    sum_op_loss,
                                    global_step],
                                   feed_dict=feed_dict)
            l_rec, l_reg, l_energy, l_penalty, f_loss \
                = sess.run([loss_rec, loss_reg, loss_energy, penalty, full_loss], feed_dict=feed_dict)

            train_l_rec += l_rec
            train_l_reg += l_reg
            train_l_energy += l_energy
            train_penalty += l_penalty
            train_f_loss += f_loss

            if np.isnan(f_loss):
                logger.info("Loss is nan - Stopping")
                break

            train_batch += 1

        train_l_rec /= nr_batches_train
        train_l_reg /= nr_batches_train
        train_l_energy /= nr_batches_train
        train_penalty /= nr_batches_train
        train_f_loss /= nr_batches_train

        logger.info('Epoch terminated')
        print("Epoch %d | time = %ds | loss rec = %.4f | loss en = %.4f | loss pen = %.4f | loss reg = %.4f "
              "| loss = %.4f"
              % (epoch, time.time() - begin, train_l_rec,
                 l1 * train_l_energy,
                 l2 * train_penalty,
                 l3 * train_l_reg,
                 train_f_loss))

        writer.add_scalar('train_l_rec', train_l_rec, epoch)
        writer.add_scalar('train_l_reg', train_l_reg, epoch)
        writer.add_scalar('train_l_energy', train_l_energy, epoch)
        writer.add_scalar('train_penalty', train_penalty, epoch)
        writer.add_scalar('train_f_loss', train_f_loss, epoch)

        epoch += 1


def run(args):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(args.rd)

        print("#" * 30)
        print("Model: {}, dataset: {}, KNN={}, seed: {}".format(args.model,
                                                                args.dataset,
                                                                args.KNN, args.rd))
        print("#" * 30)

        train_and_test(args.model, args.dataset, args.nb_epochs, args.it_e_val,
                       args.K, args.v, args.KNN,
                       args.l1, args.l2, args.l3,
                       args.label, args.rd)
