import numpy as np

class BanditChoice(object):
    def __init__(self):
        pass

    def evaluate(self, posteriorMatrix, legalItemVector, ratingMatrixForUser=None):
        # posteriorMatrix = (numSamples, numItems)
        # legalItemVector = (numItems)
        # itemIndex an integer within [0, numItems - 1]
        # user_ratings = posteriorMatrix[:,legalItemVector] 
        user_indices = np.array(range(len(legalItemVector)))
        user_indices = user_indices[legalItemVector == 1]
        user_ratings = posteriorMatrix[:,user_indices]
        itemIndex = self.get_ucb(user_ratings, user_indices)
        # itemIndex = self.get_egreedy(posteriorMatrix, legalItemVector)
        # itemIndex = self.get_thompson_sample(posteriorMatrix, legalItemVector)


        return itemIndex
    
    def get_ucb(self, user_ratings, user_indices):
        #get upper quantile of ratings
        sorted_ratings = np.sort(user_ratings,axis=0)

        uquantile = int(len(user_ratings)*(3/4))-1
        uquantile_ratings = user_ratings[uquantile]

        idx = np.argmax(uquantile_ratings)
        selected_item = user_indices[idx]

        return selected_item


    def get_egreedy(self, user_ratings, user_indices, epsilon=0.1):
        # default to greedy if no e specified
        mean_ratings = np.mean(user_ratings,axis=0)
        if random.random() < 1-epsilon:
            idx = np.argmax(mean_ratings)
        else:
            idx = random.randint(0, len(mean_ratings)-1)
        
        return user_indices[idx]
        

    def get_thompson_sample(self, user_ratings, user_indices):
        random_sample = random.randint(0,len(user_ratings)-1)
        idx = np.argmax(user_ratings[random_sample])
        selected_item = user_indices[idx]
        return selected_item


    def get_boltzmann_exploration(self, user_ratings, user_indices, tau=0.1):
        # tau - temperature parameter, if 0 -> greedy selection
        temp_scaled = [x/tau for x in np.mean(user_ratings,axis=0)]
        denom = np.sum(np.exp(temp_scaled))
        boltzmann_prob = [np.exp(x)/denom for x in temp_scale]

        idx = np.argmax(boltzmann_prob)
        selected_item = user_indices[idx]

        return selected_item

    def get_random(self, user_ratings, user_indices):
        idx = random.randint(0, len(user_ratings[0])-1)
        return user_indices[idx]

        
