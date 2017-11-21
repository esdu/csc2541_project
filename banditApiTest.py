'''
Example of API usage on how to work with this library
'''
from sclrecommender.mask import MaskGenerator
from sclrecommender.mask import RandomMaskGenerator
# TODO: from sclrecommender.mask import TimeMaskGenerator
# TODO: from sclrecommender.mask import ColdUserMaskGenerator
# TODO: from sclrecommender.mask import ColdItemMaskGenerator
from sclrecommender.mask import LegalMoveMaskGenerator
from sclrecommender.matrix import RatingMatrix
from sclrecommender.matrix import PositiveNegativeMatrix
from sclrecommender.parser import MovieLensParser

from sclrecommender.bandit.runner import BanditRunner
#from sclrecommender.bandit.model import UncertaintyModel
from uncertaintyModel import UncertaintyModel
from banditChoice import BanditChoice
#from sclrecommender.bandit.choice import BanditChoice

import copy 
import numpy as np

def pprint(obj):
    '''
    For debugging, print statements with numpy variable names and shape
    '''
    def namestr(obj):
        namespace = globals()
        return [name for name in namespace if namespace[name] is obj]
    # Assumes obj is a numpy array, matrix
    print(namestr(obj), obj.shape)
    print(obj)

if __name__ == '__main__':
    # Anything with pprint(numpyVariable) means it is a numpy matrix
    # Step 1: Get data based on dataset specific parser
    dataDirectory ="ml-100k"
    mlp = MovieLensParser(dataDirectory)
    ratingMatrix = mlp.getRatingMatrixCopy()
    pprint(ratingMatrix)

    # Step 2: Generate both Rating Matrix and Label Matrix for evaluation
    rmTruth = RatingMatrix(ratingMatrix)
    positiveThreshold = 3.0 # Threshold to set prediction to positive labels
    labelTruth = PositiveNegativeMatrix(ratingMatrix, positiveThreshold)

    # Step 3: Split rating matrix to train and test
    trainSplit = 0.8

    # Step 3.1: Choose splitting procedure

    # Option 3.1.1: Random Split
    randomMaskTrain, randomMaskTest = RandomMaskGenerator(rmTruth.getRatingMatrix(), trainSplit).getMasksCopy()

    #TODO: Option 3.1.2: Split based on time
    #TODO: Option 3.1.3: Split based on cold users
    #TODO: Option 3.1.4: Split based on cold items
   
    # Step 3.2: Apply mask
    rmTrain = copy.deepcopy(rmTruth)
    rmTest = copy.deepcopy(rmTruth)
    rmTrain.applyMask(randomMaskTrain)
    rmTest.applyMask(randomMaskTest)

    trainMatrix = rmTrain.getRatingMatrix()
    testMatrix = rmTest.getRatingMatrix()

    pprint(trainMatrix)
    pprint(testMatrix)

    # Step 4: RecommenderAlgorithm
    # Option 4.1: ReconstructionMatrix: Outputs a reconstruction of actual matrix, known as recommenderMatrix

    # Option 4.2: RankingMatrix: Outputs a matrix of ranking for each user or item
    # Bandit Specific, get Legal Move that can be trained on
    legalTrainMask = LegalMoveMaskGenerator(trainMatrix).getMaskCopy()
    legalTestMask = LegalMoveMaskGenerator(testMatrix).getMaskCopy()

    pprint(legalTrainMask)
    pprint(legalTestMask)

    banditRunner = BanditRunner(ratingMatrix, legalTrainMask, legalTestMask)

    nnmf = UncertaintyModel(ratingMatrix) # TODO: Use actual model
    ucb = BanditChoice() # TODO: Use actual choice
    banditRunner.setUncertaintyModel(nnmf)
    banditRunner.setBanditChoice(ucb)

    rankingMatrix = banditRunner.generateRanking()
    pprint(rankingMatrix)

    # Step 5: Evaluator
    # Evaluate the ranking matrix that was given

    print("DONE TESTING")
