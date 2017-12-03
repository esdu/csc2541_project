import numpy as np
import tensorflow as tf
import _pickle

# TODO figure out seed
# TODO hyperparam search

def save_graph_parameters(file):
    sess = tf.get_default_session()
    trained_vars = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        trained_vars.append(sess.run(var))    
    _pickle.dump(trained_vars, open(file, 'wb'))
    return file 

def load_graph_parameters(file):
    sess = tf.get_default_session()
    trained_vars = _pickle.load(open(file, "rb"))
    i=0
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        sess.run(var.assign(trained_vars[i]))
        i += 1
    return True

def get_nn_weights(nn_layer_dims, mean_W, stddev_W, mean_b, stddev_b):
    """
    :param nn_layer_dims <list int>: dimensions of each layer of nn.
        eg:
            [255, 50, 50, 1]
            means an input layer of dim 255,
                  followed by a hidden layer of dim 50,
                  followed by a hidden layer of dim 50,
                  followed by an output layer of dim 1.
    """
    params = []

    prv_dim = nn_layer_dims[0]
    for layer in range(1, len(nn_layer_dims)):
        cur_dim = nn_layer_dims[layer]

        W = tf.get_variable(initializer=tf.random_normal([prv_dim, cur_dim], mean=mean_W, stddev=stddev_W),
                            name="W{}".format(layer))
        b = tf.get_variable(initializer=tf.random_normal([1      , cur_dim], mean=mean_b, stddev=stddev_b),
                            name="b{}".format(layer))
        params.append(W)
        params.append(b)

        prv_dim = cur_dim

    return params

    ## For reference, if we were to code up a 5 layer NN, the params look like this:
    #
    #    W0 = tf.get_variable(initializer=tf.random_normal([D*2+Dp, nn_units_1]), name="W0")
    #    b0 = tf.get_variable(initializer=tf.random_normal([1     , nn_units_1]), name="b0")
    #    W1 = tf.get_variable(initializer=tf.random_normal([nn_units_1, nn_units_2]), name="W1")
    #    b1 = tf.get_variable(initializer=tf.random_normal([1         , nn_units_2]), name="b1")
    #    W2 = tf.get_variable(initializer=tf.random_normal([nn_units_2, nn_units_3]), name="W2")
    #    b2 = tf.get_variable(initializer=tf.random_normal([1         , nn_units_3]), name="b2")
    #    W3 = tf.get_variable(initializer=tf.random_normal([nn_units_3, 1]), name="W3")
    #    b3 = tf.get_variable(initializer=tf.random_normal([1         , 1]), name="b3")
    #    return W0, b0, W1, b1, W2, b2, W3, b3


def neural_net(x, nn_layer_dims, *args):
    """
    @param x <(None, 2*D+Dp) float>: input to the neural net
    """
    params = get_nn_weights(nn_layer_dims, *args)

    sofar = x
    for i in range(0, len(params), 2):
        W, b = params[i:i+2]
        sofar = tf.matmul(sofar, W) + b
        if i < len(params)-2: # if not last layer
            sofar = tf.nn.sigmoid(sofar)

    return sofar

    ## For reference, if we were to code up a 5 layer NN, the params look like this:
    #
    #    W0, b0, W1, b1, W2, b2, W3, b3 = get_nn_weights()
    #    z0 = tf.nn.sigmoid(tf.matmul(x , W0) + b0)
    #    z1 = tf.nn.sigmoid(tf.matmul(z0, W1) + b1)
    #    z2 = tf.nn.sigmoid(tf.matmul(z1, W2) + b2)
    #    z3 = tf.matmul(z2, W3) + b3
    #    return z3


def neural_net_tensor(x, N_LOOKUP, nn_layer_dims, *args):
    """
    @param x <(None, None, 2*D+Dp) float>: input to the neural net as a tensor.
    TODO Does this create a new graph every time it's ran? If so, change N_LOOKUP to a constant equal to number of items.
    TODO get N_LOOKUP by second dim of x
    TODO If I just use 1 function to handle tensor and non-tensor, what is the impact on speed?
    """
    # Tensor x Matrix multiplication, see:
    # Note: We can use tf.matmul if the pre- dimensions of the tensors are the same. In other words, matmul doesn't broadcast.
    # https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
    # https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/4tgsOSxwtkY
    # https://rdipietro.github.io/tensorflow-scan-examples/
    # TODO why doesn't scan work??
    # Error: TensorArray was passed element_shape [?,11] which does not match the Tensor at index 0: [3,7]
    #self.bar = tf.scan(lambda a, x: tf.matmul(x, foo), test_nn_input)

    params = get_nn_weights(nn_layer_dims, *args)

    prv_dim = nn_layer_dims[0]
    sofar = x
    for layer in range(1, len(nn_layer_dims)):
        cur_dim = nn_layer_dims[layer]
        W, b = params[2*(layer-1):2*(layer-1)+2]

        inp = tf.reshape(sofar, [-1, prv_dim])
        sofar = tf.reshape(tf.matmul(inp, W), [-1, N_LOOKUP, cur_dim]) + b

        if layer < len(nn_layer_dims)-1: # if not last layer
            sofar = tf.nn.sigmoid(sofar)

        prv_dim = cur_dim

    return sofar

    ## For reference, if we were to code up a 5 layer NN, the params look like this:

    #    W0, b0, W1, b1, W2, b2, W3, b3 = get_nn_weights()

    #    x_ = tf.reshape(x, [-1, D*2+Dp])
    #    z0 = tf.reshape(tf.matmul(x_, W0), [-1, N_LOOKUP, nn_units_1]) + b0
    #    z0 = tf.nn.sigmoid(z0)

    #    z0_ = tf.reshape(z0, [-1, nn_units_1])
    #    z1 = tf.reshape(tf.matmul(z0_, W1), [-1, N_LOOKUP, nn_units_2]) + b1
    #    z1 = tf.nn.sigmoid(z1)

    #    z1_ = tf.reshape(z1, [-1, nn_units_2])
    #    z2 = tf.reshape(tf.matmul(z1_, W2), [-1, N_LOOKUP, nn_units_3]) + b2
    #    z2 = tf.nn.sigmoid(z2)

    #    z2_ = tf.reshape(z2, [-1, nn_units_3])
    #    z3 = tf.reshape(tf.matmul(z2_, W3), [-1, N_LOOKUP, 1]) + b3

    #    return z3

def create_optimizer(optimizer, lr_init, lr_decay_steps, lr_decay_rate):
    """
    Ripped from: https://github.com/blei-lab/edward/blob/master/edward/inferences/variational_inference.py
    Also see: https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay

    :return
        optimizer: the optimizer
        global_step: variable used to count training step & decrease the learning rate
    """
    global_step = tf.Variable(0, trainable=False, name="global_step")

    learning_rate = tf.train.exponential_decay(lr_init, global_step, lr_decay_steps, lr_decay_rate, staircase=True)

    if optimizer == 'gradientdescent':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer class not found:', optimizer)

    return optimizer, global_step



class NNMF:
    def __init__(self, ratings_matrix,
                 D=30, Dp=50, var_ratio=1.,
                 nn_hidden_layer_dims=[50,50,50], nn_W_init_mean=0., nn_W_init_stddev=1., nn_b_init_mean=0., nn_b_init_stddev=1.,
                 batch_size=200,
                 optimizer='adam', lr_init=0.1, lr_decay_steps=100, lr_decay_rate=0.9):
        """
        Computes R = UV' with SVI.

        :param ratings_matrix: The full ratings matrix.
        :param D: Hidden dim size for for U and V.
        :param Dp: Hidden dim size for for Up and Vp.
        :param pZ_prior_stddev: The prior stddev for p(Z). This effectively acts as a regularizer.
        :param pR_stddev: The model's stddev, ie. for p(r_{ij}|U,V,U',V').
        :param nn_hidden_layer_dims <list int>: A list of hidden layer dimensions.
            eg. [50, 25] -> 2 hidden layers with 50 units and 25 units.
        :param nn_W_init_mean: for the neural network weights
        :param nn_W_init_stddev: -
        :param nn_b_init_mean: -
        :param nn_b_init_stddev: -
        :param batch_size: For each itration of SVI, how many samples from ratings matrix to use.
            Bigger batch => more stable gradients.
        :param n_samples: For each iteration of SVI, how many latent samples to draw to estimate the gradient.
            Higher n_samples => more stable gradients. But note it might result in biased gradients.
        :param optimizer <str>: One of:
            'gradientdescent'
            'adadelta'
            'adagrad'
            'momentum'
            'adam'
            'ftrl'
            'rmsprop'
        :param lr_init: Initial learning rate (before decay)
        :param lr_decay_steps: -
        :param lr_decay_rate: -

        Learning rate decay is done with tf.train.exponential_decay:
            decayed_learning_rate = lr_init * lr_decay_rate ^ (global_step / lr_decay_steps)

        Note: It's recommended to run this in a wrapped session. For example:

        ```
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.as_default():
            model = NNMF(...)
            model.train(...)
        sess.close()
        ```

        TODO the following are not parameterized yet, becaus they don't seem important:
        - q: mu, var
        - latent priors: mu
        """

        # Require an optimizer (don't use Edward's default)
        assert optimizer

        # Require at least 1 hidden layer.
        assert nn_hidden_layer_dims is not None
        # Prepend input size, append output size.
        nn_layer_dims = [D*2+Dp] + nn_hidden_layer_dims + [1]

        self.batch_size = batch_size
        self.R_ = ratings_matrix
        N, M = self.R_.shape; self.N = N; self.M = M

        ############
        # TRAINING #
        ############
        # We use r_ph to feed in only the elements in R_ that idx_i and idx_j correspond to.
        self.r_ph  = tf.placeholder(tf.float32, name="batch_r")
        self.idx_i = tf.placeholder(tf.int32, name="idx_i")
        self.idx_j = tf.placeholder(tf.int32, name="idx_j")

        # Latent Variables
        with tf.variable_scope('latent') as scope:
            self.U  = tf.Variable(tf.random_normal([N, D]), name='U')
            self.V  = tf.Variable(tf.random_normal([M, D]), name='V')
            self.Up = tf.Variable(tf.random_normal([N, Dp]), name='Up')
            self.Vp = tf.Variable(tf.random_normal([M, Dp]), name='Vp')

        U_selected  = tf.gather(self.U , self.idx_i)
        V_selected  = tf.gather(self.V , self.idx_j)
        Up_selected = tf.gather(self.Up, self.idx_i)
        Vp_selected = tf.gather(self.Vp, self.idx_j)

        nn_input = tf.concat([U_selected, V_selected, tf.multiply(Up_selected, Vp_selected)], axis=1)

        with tf.variable_scope('nn') as nn_scope:
            self.means = neural_net(nn_input, nn_layer_dims, nn_W_init_mean, nn_W_init_stddev, nn_b_init_mean, nn_b_init_stddev)

        self.predicted_R = tf.squeeze(self.means)
        
        # Cost Function
        self.regularizer = var_ratio*(tf.reduce_sum(tf.square(self.U))+tf.reduce_sum(tf.square(self.V))+tf.reduce_sum(tf.square(self.Up))+tf.reduce_sum(tf.square(self.Vp)))
        self.cost = tf.reduce_sum(tf.square(self.r_ph-self.predicted_R)) + self.regularizer
        
          
        # Parameter Updates
        optimizer, global_step = create_optimizer(optimizer, lr_init, lr_decay_steps, lr_decay_rate)
        self.train_step = optimizer.minimize(self.cost, global_step=global_step)
        
        tf.get_default_session().run(tf.global_variables_initializer())

    def sample_user_ratings(self, user_index, n_samples=100):
        # third parameter not used
        idx_i = [user_index] * self.M
        idx_j = list(range(self.M))
        feed_dict = {
            self.idx_i: idx_i,
            self.idx_j: idx_j,
        }
        return np.expand_dims(tf.get_default_session().run(self.predicted_R, feed_dict), axis=0)


    def train(self, mask, n_iter=1000, verbose=False):
        """
        :param mask: Same size as ratings_matrix.
            A 0 indicates to not use this rating, else use this rating.
            If None given, ratings_matrix will be used as the mask.
        :param n_iter: How many iterations of SVI to run.
        """
        sess = tf.get_default_session()
        losses = []
        seen_indices = np.array(np.where(mask))
        for _ in range(n_iter):
            # TODO is there a problem training with random batches? Look into lit more.
            # Train on a batch of self.batch_size random elements each iteration.
            rand_idx = np.random.choice(seen_indices.shape[1], self.batch_size, replace=False)
            idx_i = seen_indices[0, rand_idx]
            idx_j = seen_indices[1, rand_idx]
            feed_dict = {
                self.idx_i: idx_i,
                self.idx_j: idx_j,
                self.r_ph: self.R_[idx_i, idx_j]
            }
            
            sess.run(self.train_step, feed_dict=feed_dict)
            loss = sess.run(self.cost, feed_dict=feed_dict)
            losses.append(loss)

        return losses