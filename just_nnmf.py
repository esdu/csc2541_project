import tensorflow as tf
import numpy as np
import _pickle as cPickle
from matplotlib import pyplot as plt

class NNMF:
    def __init__(self, num_users, num_items, user_indices, item_indices, ratings, D=2, D_prime=1, hidden_units_per_layer=10):
        self.num_users, self.num_items = num_users, num_items
        self.D = D
        self.D_prime = D_prime
        self.nhid = hidden_units_per_layer 
        self.user_indices = user_indices    
        self.item_indices = item_indices 
        self.ratings = ratings 
        self.nn_parameters = []
        
    def latent_factors(self):
        with tf.name_scope('latent_factors') as scope:
            self.U = tf.Variable(tf.truncated_normal([self.num_users, self.D]), name='U')
            self.V = tf.Variable(tf.truncated_normal([self.num_items, self.D]), name='V')
            self.U_prime = tf.Variable(tf.truncated_normal([self.num_users, self.D_prime]), name='U_prime')
            self.V_prime = tf.Variable(tf.truncated_normal([self.num_items, self.D_prime]), name='V_prime')
            
    def nn_inputs_preprocess(self):
        U_selected = tf.gather_nd(self.U, self.user_indices)
        V_selected = tf.gather_nd(self.V, self.item_indices)
        U_prime_selected = tf.gather_nd(self.U_prime, self.user_indices)
        V_prime_selected = tf.gather_nd(self.V_prime, self.item_indices)        
        self.nn_inputs = tf.concat([U_selected, V_selected, tf.multiply(U_prime_selected, V_prime_selected)], 1)
        
    def nn_fclayers(self):
        
        with tf.name_scope('fc_layers') as scope:
            #fc1
            self.fc1W = tf.Variable(tf.random_normal([2*self.D + self.D_prime, self.nhid], dtype=tf.float32, stddev=1e-1), name='fc1_weights')
            self.fc1b = tf.Variable(tf.random_normal([self.nhid], dtype=tf.float32, stddev=1e-1), name='fc1_bias')
            fc1_in = tf.nn.bias_add(tf.matmul(self.nn_inputs, self.fc1W), self.fc1b)
            fc1 = tf.nn.tanh(fc1_in)
            self.nn_parameters += [self.fc1W, self.fc1b]
            
            #fc2
            self.fc2W = tf.Variable(tf.random_normal([self.nhid, self.nhid], dtype=tf.float32, stddev=1e-1), name='fc2_weights')
            self.fc2b = tf.Variable(tf.random_normal([self.nhid], dtype=tf.float32, stddev=1e-1), name='fc2_bias')
            fc2_in = tf.nn.bias_add(tf.matmul(fc1, self.fc2W), self.fc2b)
            fc2 = tf.nn.tanh(fc2_in)
            self.nn_parameters += [self.fc2W, self.fc2b]
            
            #fc3
            self.fc3W = tf.Variable(tf.random_normal([self.nhid, 1], dtype=tf.float32, stddev=1e-1), name='fc3_weights')
            self.fc3b = tf.Variable(tf.random_normal([1], dtype=tf.float32, stddev=1e-1), name='fc3_bias')
            fc3_in = tf.nn.bias_add(tf.matmul(fc2, self.fc3W), self.fc3b)
            self.prediction = tf.reshape(5*tf.nn.sigmoid(fc3_in),shape=[-1])
            self.nn_parameters += [self.fc3W, self.fc3b]
        
    def build_graph(self):
        self.latent_factors()
        self.nn_inputs_preprocess()
        self.nn_fclayers()
        
    def load_latent_factors(self, file, sess):
        trained_latent_factors = cPickle.load(open(file, "rb"))
        sess.run(self.U.assign(trained_latent_factors['U']))
        sess.run(self.V.assign(trained_latent_factors['V']))    
        sess.run(self.U_prime.assign(trained_latent_factors['U_prime']))
        sess.run(self.V_prime.assign(trained_latent_factors['V_prime']))
    
    def load_nn_parameters(self, file, sess):
        trained_nn_parameters = cPickle.load(open(file, "rb"))
        nn_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc_layers")
        keys = sorted(trained_nn_parameters.keys())
        for i in range(len(self.nn_parameters)):
            sess.run(self.nn_parameters[i].assign(trained_nn_parameters[keys[i]]))
        



class learning_graph:
    def __init__(self, num_users, num_items,  model):
        self.user_indices = tf.placeholder(tf.int32, shape = [None, 1])
        self.item_indices = tf.placeholder(tf.int32, shape = [None, 1])
        self.ratings = tf.placeholder(tf.float32, shape = [None, None])
        
        # build model
        self.model = model(num_users, num_items, self.user_indices, self.item_indices, self.ratings)
        self.model.build_graph()
        
    def build_train_graph(self,rate):
        ratings_indices = tf.concat([self.user_indices, self.item_indices],1)
        self.ratings_selected = tf.gather_nd(self.ratings, ratings_indices)
        self.cost = tf.reduce_sum(tf.square(self.ratings_selected-self.model.prediction))  
        
        # alternating variable updates
        optimizer = tf.train.AdamOptimizer(rate)

        self.nn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc_layers")
        self.nn_vars_update = optimizer.minimize(self.cost, var_list=self.nn_vars)

        self.latent_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"latent_factors")     
        self.latent_vars_update = optimizer.minimize(self.cost, var_list=self.latent_vars)
        


        
def train(graph, user_indices, item_indices, ratings, rate, iter, sess):
    
    graph.build_train_graph(rate)
    sess.run(tf.global_variables_initializer())
    
    train_costs = []
    
    for i in range(iter):
        graph.nn_vars_update.run(feed_dict={graph.user_indices: user_indices, graph.item_indices: item_indices, graph.ratings:ratings})
        graph.latent_vars_update.run(feed_dict={graph.user_indices: user_indices, graph.item_indices: item_indices, graph.ratings:ratings})
        
        if (i+1)%5 == 0:
            print('iteration '+str(i+1))
            train_cost = sess.run(graph.cost, feed_dict={graph.user_indices:user_indices, graph.item_indices:item_indices, graph.ratings:ratings})
            print(train_cost)
            train_costs.append(train_cost)       
            
    plt.plot(train_costs)
    
    
    
    latent_factor_names = ['U', 'V', 'U_prime', 'V_prime']
    nn_parameter_names = ['fc1W', 'fc1b', 'fc2W', 'fc2b', 'fc3W', 'fc3b']
    
    # save trained parameters
    latent_factors = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"latent_factors")
    nn_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc_layers")
    
    trained_latent_factors = {}
    trained_nn_parameters = {}
    
    for j in range(len(latent_factor_names)):
        trained_latent_factors[latent_factor_names[j]] = sess.run(latent_factors[j])
    
    for j in range(len(nn_parameter_names)):
        trained_nn_parameters[nn_parameter_names[j]]= sess.run(nn_parameters[j])
    
    cPickle.dump(trained_latent_factors, open('latent_factors.pkl', 'wb'))
    cPickle.dump(trained_nn_parameters, open('nn_parameters.pkl', 'wb')) 


def predict(graph, user_indices, item_indices, latent_factor_file, nn_parameter_file, sess):
    sess.run(tf.global_variables_initializer())
    graph.model.load_latent_factors(latent_factor_file, sess)
    graph.model.load_nn_parameters(nn_parameter_file, sess)
    return sess.run(graph.model.prediction, feed_dict={graph.user_indices:user_indices, graph.item_indices: item_indices})

    
        
    
        