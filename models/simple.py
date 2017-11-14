import numpy as np
import tensorflow as tf
import edward as ed

class SimpleMatrixFactorization:
    def __init__(self, ratings_matrix, mask=None, hidden_dim=30,
                 n_iter=1000, batch_size=200, n_samples=1):
        """
        Computes R = UV' with SVI.

        :param ratings_matrix: The full ratings matrix.
            Ratings should be positive, and a rating of 0 means unknown rating.
        :param mask: Same size as ratings_matrix.
            A 0 indicates to not use this rating, else use this rating.
            If None given, ratings_matrix will be used as the mask.
        :param hidden_dim: Hidden dim size for for U and V.
        :param n_iter: How many iterations of SVI to run.
        :param batch_size: For each itration of SVI, how many samples from
            ratings matrix to use.
            Bigger batch => more stable gradients.
        :param n_sample: For each iteration of SVI, how many latent samples to
            draw to estimate the gradient.
            Higher n_sample => more stable gradients.
        """

        self.R_ = ratings_matrix
        N, M = self.R_.shape
        D = hidden_dim
        self.BATCH = batch_size

        # We use r_ph to feed in only the elements in R_ that idx_i and idx_j correspond to.
        self.r_ph  = tf.placeholder(tf.float32, name="batch_r")
        self.idx_i = tf.placeholder(tf.int32, name="idx_i")
        self.idx_j = tf.placeholder(tf.int32, name="idx_j")

        # "Priors" p(Z)
        self.U = ed.models.Normal(loc=tf.zeros([N, D]), scale=tf.ones([N, D]))
        self.V = ed.models.Normal(loc=tf.zeros([M, D]), scale=tf.ones([M, D]))

        # P(X|Z)
        U_selected = tf.gather(self.U, self.idx_i)
        V_selected = tf.gather(self.V, self.idx_j)
        self.R = ed.models.Normal(loc=tf.reduce_sum(tf.multiply(U_selected, V_selected), axis=1),
                                  scale=tf.ones(self.BATCH))

        # VI
        self.qU = ed.models.Normal(loc=tf.Variable(tf.zeros([N, D])),
                                   scale=tf.Variable(tf.ones([N, D])))
        self.qV = ed.models.Normal(loc=tf.Variable(tf.zeros([M, D])),
                                   scale=tf.Variable(tf.ones([M, D])))

        # Testing
        self.U_samples = self.qU.sample(100)
        self.V_samples = self.qV.sample(100)

        # Inference
        self.inference = ed.KLqp({self.U: self.qU, self.V: self.qV}, data={self.R: self.r_ph})
        self.inference.initialize(scale={self.R: N*M/self.BATCH}, n_iter=n_iter, n_samples=n_samples)
        # Note: global_variables_initializer has to be run after creating inference.
        ed.get_session().run(tf.global_variables_initializer())

        self.trained = False


    def rhat_samples(self):
        U_samples_, V_samples_ = ed.get_session().run([self.U_samples, self.V_samples])
        R_samples_ = []
        for i in range(U_samples_.shape[0]):
            R_samples_.append(np.matmul(U_samples_[i], np.transpose(V_samples_[i])))
        R_samples_ = np.array(R_samples_)
        return R_samples_


    def mse(self):
        R_samples_ = self.rhat_samples()
        return np.mean(np.square(np.mean(R_samples_, axis=0) - self.R_)[np.where(self.R_)])


    def train(self, verbose=False):
        seen_indices = np.array(np.where(self.R_))
        info_dicts = []
        for _ in range(self.inference.n_iter):
            # Train on a batch of BATCH_SIZE random elements each iteration.
            rand_idx = np.random.choice(seen_indices.shape[1], self.BATCH, replace=False)
            idx_i_ = seen_indices[0, rand_idx]
            idx_j_ = seen_indices[1, rand_idx]
            feed_dict = {
                self.idx_i: idx_i_,
                self.idx_j: idx_j_,
                self.r_ph: self.R_[idx_i_, idx_j_]
            }
            info_dict = self.inference.update(feed_dict=feed_dict)
            info_dicts.append(info_dict)

            if verbose: self.inference.print_progress(info_dict)

        losses = [x['loss'] for x in info_dicts]
        return losses
