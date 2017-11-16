from util import make_bandits_mask

import numpy as np
import random

class BanditAlgorithms:
    def __init__(self, user, R, mask, gamma, model, retrain):

        self.user = user
        self.R = R
        self.mask = mask
        self.gamma = gamma

        self.model = model
        self.retrain = retrain


    def initialize_values(self, egreedy, ucb, thompson):
    	# initialize the user ratings
    	# egreedy: MAP, ucb: mean + std, thompson: sample

        if egreedy == True:
            user_ratings = np.mean(self.model.rhat_samples(),axis=0)[self.user]
            user_indices = np.array(range(len(user_ratings)))

        elif ucb == True:
            user_mean = np.mean(self.model.rhat_samples(),axis=0)[self.user]
            user_dev = np.std(self.model.rhat_samples(),axis=0)[self.user]

            user_ratings = user_mean + user_dev
            user_indices = np.array(range(len(user_ratings)))

        elif thompson == True:
            #TODO: take a sample from posterior directly instead of taking a sample of the 100 samples..
            random_sample = random.randint(0,99)
            user_ratings = self.model.rhat_samples()[random_sample][self.user]
            user_indices = np.array(range(len(user_ratings)))

        return user_ratings, user_indices



    def get_score(self, epsilon, egreedy, ucb, thompson):
        # TODO: retraining is prohibitively expensive here...
        
        my_mask = make_bandits_mask(self.mask, self.R) # Note: This returns a copy of mask.
        
        discount = 1
        score = 0
        
        n_iter = np.sum(my_mask[self.user] == 0)
        
        for _ in range(n_iter):
            user_ratings, user_indices = self.initialize_values(egreedy, ucb, thompson)

            user_ratings = user_ratings[my_mask[self.user] == 0]
            user_indices = user_indices[my_mask[self.user] == 0]
            
            if egreedy == True:
                if random.random() < epsilon:
                    # Choose greedy action
                    idx = np.argmax(user_ratings)
                else:
                    # Choose random action
                    idx = random.randint(0,len(user_ratings)-1)

            elif ucb == True or thompson == True:
                idx = np.argmax(user_ratings)
            

            my_mask[self.user,user_indices[idx]] = 1
            score += self.R[self.user,user_indices[idx]] * discount
            discount *= self.gamma

            if self.retrain:
                self.model.train(R=self.R, mask=my_mask, n_iter=5000)
        
        return score

    
    
    def get_best_score(self):
        '''
        user: user index
        R: true ratings matrix of size (n_users, n_items)
        mask: mask matrix of size (n_users, n_items), 1 if ratings is observed, 0 otherwise.
        '''
        mask_bandit = make_bandits_mask(self.mask, self.R)
        # Extract missing entries
        missing_ratings = self.R[self.user][np.where(mask_bandit[self.user] == 0)]
        # Sort biggest to smallest
        missing_ratings = missing_ratings[np.argsort(-missing_ratings)]
        best_score = 0
        discount = 1
        for r in missing_ratings:
            best_score += r * discount
            discount *= self.gamma
        return best_score