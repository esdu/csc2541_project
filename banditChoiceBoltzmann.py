import numpy as np
import random

class BanditChoiceBoltzmann(object):
    def __init__(self):
        pass

    def evaluate(self, posteriorMatrix, legalItemVector):

        user_indices = np.array(range(len(legalItemVector)))
        user_indices = user_indices[legalItemVector == 1]
        user_ratings = posteriorMatrix[:,user_indices]

        itemIndex = self.get_boltzmann_exploration(user_ratings, user_indices)

        return itemIndex


    def get_boltzmann_exploration(self, user_ratings, user_indices, tau=0.1):
        # tau - temperature parameter, if 0 -> greedy selection
        temp_scaled = [x/tau for x in np.mean(user_ratings,axis=0)]
        denom = np.sum(np.exp(temp_scaled))
        boltzmann_prob = [np.exp(x)/denom for x in temp_scaled]

        indices = list(range(len(boltzmann_prob)))
        idx = np.random.choice(indices,p=boltzmann_prob)
        # idx = np.argmax(boltzmann_prob)
        selected_item = user_indices[idx]

        return selected_item
