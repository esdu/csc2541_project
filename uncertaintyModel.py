class UncertaintyModel(object):
    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix
        self.posteriorMatrix = None

    def reset(self):
        # Reset the weights as if no training was done
        # TODO: 
        pass

    def setRatingMatrix(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix

    def getPosteriorMatrix(self):
        return self.posteriorMatrix

    def train(self, legalTrainIndices):
        # TODO: train the weights based on current legalTrainIndices
        return self.posteriorMatrix
