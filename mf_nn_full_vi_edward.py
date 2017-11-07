import numpy as np
import tensorflow as tf
import random
import edward as ed

class NNMF_FullVI_Edward(object):
    def __init__(self, n_users, n_items, latent_dim, latent_dim_p,
                 reg_rate=1., learning_rate=0.01, hidden_units=5):
        self.n_users = n_users
        self.n_items = n_items

        self.latent_dim = latent_dim
        self.latent_dim_p = latent_dim_p

        self.reg_rate = reg_rate
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units

        # TODO this is hacky!
        i_s = []
        j_s = []
        for i in range(self.n_users):
            for j in range(self.n_items):
                i_s.append(i)
                j_s.append(j)
        self.idx_i = tf.constant(i_s)
        self.idx_j = tf.constant(j_s)

        # Inputs
        self.mask = tf.placeholder(tf.float32, shape=[self.n_users, self.n_items])

        def create_mat_and_qmat(shape, name):
            mat = ed.models.Normal(loc=tf.zeros(shape), scale=tf.ones(shape), name=name)
            qmat = ed.models.Normal(loc=tf.Variable(tf.random_normal(shape)),
                                    scale=tf.nn.softplus(tf.Variable(tf.random_normal(shape))))
            return mat, qmat

        self.U,  self.qU = create_mat_and_qmat([self.n_users, latent_dim], name="user_matrix")
        self.V,  self.qV = create_mat_and_qmat([self.n_items, latent_dim], name="item_matrix")
        self.Up, self.qUp = create_mat_and_qmat([self.n_users, latent_dim_p], name="user_matrix_p")
        self.Vp, self.qVp = create_mat_and_qmat([self.n_items, latent_dim_p], name="item_matrix_p")

        # U,V,U',V'
        U_vector = tf.gather(self.U, self.idx_i)
        V_vector = tf.gather(self.V, self.idx_j)
        U_mat = tf.gather(self.Up, self.idx_i)
        V_mat = tf.gather(self.Vp, self.idx_j)
        UV_interaction = tf.multiply(U_mat, V_mat)
        all_latent_vars = tf.reshape(tf.concat([U_vector, V_vector, UV_interaction], axis=1),
                                     [self.n_users*self.n_items, latent_dim*2+latent_dim_p])

        self.W0, self.qW0 = create_mat_and_qmat([latent_dim*2+latent_dim_p, hidden_units], name="W0")
        self.b0, self.qb0 = create_mat_and_qmat([1, hidden_units], name="b0")
        self.W1, self.qW1 = create_mat_and_qmat([hidden_units, 1], name="W1")
        self.b1, self.qb1 = create_mat_and_qmat([1, 1], name="b1")

        layer_0_out = tf.nn.relu(tf.matmul(all_latent_vars, self.W0) + self.b0)
        layer_1_out = tf.nn.relu(tf.matmul(layer_0_out, self.W1) + self.b1)

        R_means = tf.reshape(layer_1_out, [self.n_users, self.n_items])
        self.R = ed.models.Normal(loc=R_means, scale=100 * (1-self.mask) + 0.1)

        # TODO is there a better way of getting predictive posterior from Edward?
        self.rhat_samples = tf.stack([self._draw_sample() for _ in range(100)])

        sess = ed.get_session()
        sess.run(tf.global_variables_initializer())

        self.prior = self._get_rhats()
        self.posterior = None
        self.posterior_map = None


    def train(self, R, mask, n_iter=2000, n_samples=5):
        '''
        Re-train model given the true R and a mask.
        '''
        # Note: Each inference run starts from scratch
        sess = ed.get_session()
        sess.as_default()
        inference = ed.KLqp({self.U: self.qU, self.V: self.qV,
                             self.Up: self.qUp, self.Vp: self.qVp,
                             self.W0: self.qW0, self.b0: self.qb0,
                             self.W1: self.qW1, self.b1: self.qb1
                            }, data={self.R: R, self.mask: mask})
        inference.run(n_iter=n_iter, n_samples=n_samples)

        self.posterior = self._get_rhats()
        # I think the marginals are gaussians, so we can use mean to find MAP.
        self.posterior_map = np.mean(self.posterior, axis=0)


    def get_prediction(self, i, j):
        '''
        Get the predicted mean and variance for user i and item j
        '''
        if self.posterior is None:
            raise Exception("Run `train` first to get posterior")
        else:
            return self.posterior_map[i,j], np.var(self.posterior[:,i,j])


    def _get_rhats(self):
        sess = ed.get_session()
        return self.rhat_samples.eval(session=sess)


    def _draw_sample(self):
        U_  = self.qU.sample()
        V_  = self.qV.sample()
        Up_ = self.qUp.sample()
        Vp_ = self.qVp.sample()
        W0_ = self.qW0.sample()
        b0_ = self.qb0.sample()
        W1_ = self.qW1.sample()
        b1_ = self.qb1.sample()
        U_vector = tf.gather(U_, self.idx_i)
        V_vector = tf.gather(V_, self.idx_j)
        U_mat = tf.gather(Up_, self.idx_i)
        V_mat = tf.gather(Vp_, self.idx_j)
        UV_interaction = tf.multiply(U_mat, V_mat)
        all_latent_vars = tf.reshape(tf.concat([U_vector, V_vector, UV_interaction], axis=1),
                                     [self.n_users*self.n_items, self.latent_dim*2+self.latent_dim_p])
        layer_0_out = tf.nn.relu(tf.matmul(all_latent_vars, W0_) + b0_)
        layer_1_out = tf.nn.relu(tf.matmul(layer_0_out, W1_) + b1_)
        R = tf.reshape(layer_1_out, [self.n_users, self.n_items])
        return R
