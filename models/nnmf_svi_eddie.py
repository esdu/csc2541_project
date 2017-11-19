import numpy as np
import tensorflow as tf
import edward as ed

class NNMF:
    def __init__(self, ratings_matrix, D=30, Dp=50,
                 batch_size=200, n_samples=1,
                 n_test_samples=100, HIDDEN_UNITS=100):
        """
        Computes R = UV' with SVI.

        :param ratings_matrix: The full ratings matrix.
            Ratings should be positive, and a rating of 0 means unknown rating.
        :param D: Hidden dim size for for U and V.
        :param Dp: Hidden dim size for for Up and Vp.
        :param batch_size: For each itration of SVI, how many samples from
            ratings matrix to use.
            Bigger batch => more stable gradients.
        :param n_sample: For each iteration of SVI, how many latent samples to
            draw to estimate the gradient.
            Higher n_sample => more stable gradients.
        :param n_test_samples: How many test samples to compute together.
        :param HIDDEN_UNITS: How many hidden units to use in the 1 hidden layer.
        """
        # TODO: How to update this with more data? Right now it needs to retrain from beginning.
        #       Re-building a graph every time isn't ideal. We need to at least clean up the old graph.
        #       Workaround: user call tf.reset_default_graph before creating this.

        self.batch_size = batch_size
        self.R_ = ratings_matrix
        N, M = self.R_.shape; self.N = N; self.M = M

        # TODO a more flexible way to handle multiple layers of a NN.
        #HIDDEN_UNITS = 20

        # TODO did the paper use ReLU or Sigmoid?

        def get_nn_weights():
            W0 = tf.get_variable(initializer=tf.random_normal([D*2+Dp, HIDDEN_UNITS]), name="W0")
            b0 = tf.get_variable(initializer=tf.random_normal([1     , HIDDEN_UNITS]), name="b0")
            W1 = tf.get_variable(initializer=tf.random_normal([HIDDEN_UNITS, 1]), name="W1")
            b1 = tf.get_variable(initializer=tf.random_normal([1           , 1]), name="b1")
            return W0, b0, W1, b1

        def neural_net(x):
            """
            @param x <(None, 2*D+Dp) float>: input to the neural net
            """
            W0, b0, W1, b1 = get_nn_weights()

            z0 = tf.nn.sigmoid(tf.matmul(x , W0) + b0)
            z1 = tf.matmul(z0, W1) + b1
            return z1

        def neural_net_tensor(x, N_LOOKUP):
            """
            @param x <(None, None, 2*D+Dp) float>: input to the neural net as a tensor.
            TODO get N_LOOKUP by second dim of x
            TODO If I just use 1 function to handle tensor and non-tensor, what is the impact on speed?
            """
            W0, b0, W1, b1 = get_nn_weights()

            # Tensor x Matrix multiplication, see:
            # Note: We can use tf.matmul if the pre- dimensions of the tensors are the same. In other words, matmul doesn't broadcast.
            # https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
            # https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/4tgsOSxwtkY
            # https://rdipietro.github.io/tensorflow-scan-examples/
            # TODO why doesn't scan work??
            # Error: TensorArray was passed element_shape [?,11] which does not match the Tensor at index 0: [3,7]
            #self.bar = tf.scan(lambda a, x: tf.matmul(x, foo), test_nn_input)

            x_ = tf.reshape(x, [-1, D*2+Dp])
            z0 = tf.reshape(tf.matmul(x_, W0), [-1, N_LOOKUP, HIDDEN_UNITS]) + b0
            z0 = tf.nn.sigmoid(z0)

            z0_ = tf.reshape(z0, [-1, HIDDEN_UNITS])
            z1 = tf.reshape(tf.matmul(z0_, W1), [-1, N_LOOKUP, 1]) + b1

            return z1

        def create_mat_and_qmat(shape):
            mat = ed.models.Normal(loc=tf.zeros(shape), scale=tf.ones(shape))
            qmat = ed.models.Normal(loc=tf.Variable(tf.zeros(shape)),
                                    scale=tf.Variable(tf.ones(shape)))
            return mat, qmat

        ############
        # TRAINING #
        ############
        # We use r_ph to feed in only the elements in R_ that idx_i and idx_j correspond to.
        self.r_ph  = tf.placeholder(tf.float32, name="batch_r")
        self.idx_i = tf.placeholder(tf.int32, name="idx_i")
        self.idx_j = tf.placeholder(tf.int32, name="idx_j")

        # "Priors" p(Z) and the variational approximating dist q(Z)
        self.U, self.qU = create_mat_and_qmat([N, D])
        self.V, self.qV = create_mat_and_qmat([M, D])
        self.Up, self.qUp = create_mat_and_qmat([N, Dp])
        self.Vp, self.qVp = create_mat_and_qmat([M, Dp])

        # P(X|Z)
        U_selected  = tf.gather(self.U , self.idx_i)
        V_selected  = tf.gather(self.V , self.idx_j)
        Up_selected = tf.gather(self.Up, self.idx_i)
        Vp_selected = tf.gather(self.Vp, self.idx_j)

        nn_input = tf.concat([U_selected, V_selected, tf.multiply(Up_selected, Vp_selected)], axis=1)

        with tf.variable_scope('nn') as nn_scope:
            means = neural_net(nn_input)

        self.R = ed.models.Normal(loc=tf.squeeze(means), scale=tf.ones(self.batch_size))

        #############
        # Inference #
        #############
        self.inference = ed.KLqp({self.U: self.qU, self.V: self.qV,
                                  self.Up: self.qUp, self.Vp: self.qVp}, data={self.R: self.r_ph})
        self.inference.initialize(scale={self.R: N*M/self.batch_size}, n_samples=n_samples)
        # Note: global_variables_initializer has to be run after creating inference.
        ed.get_session().run(tf.global_variables_initializer())

        ###########
        # Testing #
        ###########
        self.test_idx_i = tf.placeholder(tf.int32, name="test_idx_i")
        self.test_idx_j = tf.placeholder(tf.int32, name="test_idx_j")
        N_LOOKUP = tf.size(self.test_idx_i)
        # TODO assert tf.size(self.test_idx_i) == tf.size(self.test_idx_j)

        # after gather => (n_test_samples, N_LOOKUP, D or Dp)
        qU_samples  = tf.gather(self.qU.sample(n_test_samples) , self.test_idx_i, axis=1)
        qV_samples  = tf.gather(self.qV.sample(n_test_samples) , self.test_idx_j, axis=1)
        qUp_samples = tf.gather(self.qUp.sample(n_test_samples), self.test_idx_i, axis=1)
        qVp_samples = tf.gather(self.qVp.sample(n_test_samples), self.test_idx_j, axis=1)

        # shape (n_test_samples, N_LOOKUP, 2*D+Dp)
        test_nn_input = tf.concat([qU_samples, qV_samples, tf.multiply(qUp_samples, qVp_samples)], axis=-1)

        with tf.variable_scope(nn_scope, reuse=True):
            self.sample_rhats = neural_net_tensor(test_nn_input, N_LOOKUP)


    def sample_user_ratings(self, user_index):
        idx_i = [user_index] * self.M
        idx_j = list(range(self.M))
        return np.squeeze(ed.get_session().run(self.sample_rhats, {self.test_idx_i:idx_i, self.test_idx_j:idx_j}))


    def train(self, mask, n_iter=1000, verbose=False):
        """
        :param mask: Same size as ratings_matrix.
            A 0 indicates to not use this rating, else use this rating.
            If None given, ratings_matrix will be used as the mask.
        :param n_iter: How many iterations of SVI to run.
        """
        seen_indices = np.array(np.where(mask))
        info_dicts = []
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
            info_dict = self.inference.update(feed_dict=feed_dict)
            info_dicts.append(info_dict)

            # TODO print out progress without using edward
            #if verbose: self.inference.print_progress(info_dict)

        losses = [x['loss'] for x in info_dicts]
        return losses
