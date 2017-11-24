'''
Example of API usage on how to work with this library
Tests almost everything.
'''

from sclrecommender.mask import MaskGenerator
from sclrecommender.mask import RandomMaskGenerator
# TODO: from sclrecommender.mask import TimeMaskGenerator
# TODO: from sclrecommender.mask import ColdUserMaskGenerator
# TODO: from sclrecommender.mask import ColdItemMaskGenerator
from sclrecommender.mask import LegalMoveMaskGenerator
from sclrecommender.matrix import RatingMatrix
from sclrecommender.matrix import PositiveNegativeMatrix
from sclrecommender.parser import ExampleParser
from sclrecommender.parser import MovieLensParser

from sclrecommender.bandit.runner import BanditRunner
#from sclrecommender.bandit.model import UncertaintyModel
from uncertaintyModel import UncertaintyModel
from nnmf import NNMF
from banditChoice import BanditChoice
#from sclrecommender.bandit.choice import BanditChoice

from sclrecommender.evaluator import Evaluator
# Reconstruction Evaluators
from sclrecommender.evaluator import ReconstructionEvaluator
from sclrecommender.evaluator import RootMeanSquareError
from sclrecommender.evaluator import PositiveNegativeEvaluator
from sclrecommender.evaluator import F1ScoreEvaluator

# Ranking Evaluators
from sclrecommender.evaluator import RankingEvaluator
from sclrecommender.evaluator import RecallAtK
from sclrecommender.evaluator import PrecisionAtK
from sclrecommender.evaluator import MeanAveragePrecisionAtK
# TODO: NDCG, used by 2017 paper by Pierre

# Bandit Evaluators 
from sclrecommender.evaluator import RegretOptimalEvaluator

import copy 
import numpy as np
import random

def pprint(obj):
    '''
    For debugging, print statements with numpy variable names and shape
    '''
    def namestr(obj):
        namespace = globals()
        return [name for name in namespace if namespace[name] is obj]
    # Assumes obj is a numpy array, matrix
    try:
        print(namestr(obj), obj.shape)
    except:
        print(namestr(obj))
    print(obj)

if __name__ == '__main__':
    # seedNum =  int(random.random() * 1000)
    # print("SEEDNUM IS", seedNum)
    seedNum = 196
    np.random.seed(seedNum)
    random.seed(seedNum)
    
    # Anything with pprint(numpyVariable) means it is a numpy matrix
    # Step 1: Get data based on dataset specific parser
    dataDirectory ="ml-100k"
    mlp = MovieLensParser(dataDirectory)
    numUser = 8
    numItem = 8 
    exParser = ExampleParser(dataDirectory)
    ratingMatrix = exParser.getRatingMatrix(numUser, numItem)
    ratingMatrix[0][0] = 1.0
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
    positiveNegativeMatrix = labelTruth.getPositiveNegativeMatrix()

    pprint(trainMatrix)
    pprint(testMatrix)
    pprint(positiveNegativeMatrix)


    # Step 4: RecommenderAlgorithm
    # Option 4.1: ReconstructionMatrix: Outputs a reconstruction of actual matrix, known as recommenderMatrix

    reconstructionMatrix = ratingMatrix.copy() # TODO: Calculate reconstruction matrix 
    reconstructionPrediction = PositiveNegativeMatrix(reconstructionMatrix, positiveThreshold)
    positiveNegativePredictionMatrix = reconstructionPrediction.getPositiveNegativeMatrix()

    pprint(reconstructionMatrix)
    pprint(positiveNegativePredictionMatrix)

    # Option 4.2: RankingMatrix: Outputs a matrix of ranking for each user or item
    # Bandit Specific, get Legal Move that can be trained on
    legalTrainMask = LegalMoveMaskGenerator(trainMatrix).getMaskCopy()
    legalTestMask = LegalMoveMaskGenerator(testMatrix).getMaskCopy()

    pprint(legalTrainMask)
    pprint(legalTestMask)

    banditRunner = BanditRunner(ratingMatrix, legalTrainMask, legalTestMask)

    #nnmf = UncertaintyModel(ratingMatrix) # TODO: Use actual model
    nnmf = NNMF(ratingMatrix) # TODO: Use actual model
    ucb = BanditChoice() # TODO: Use actual choice
    banditRunner.setUncertaintyModel(nnmf)
    banditRunner.setBanditChoice(ucb)

    rankingMatrix = banditRunner.generateRanking()
    pprint(rankingMatrix)

    # Step 5: Evaluator

    # Option 5.1 Reconstruction Matrix evaluators
    accuracy = ReconstructionEvaluator(ratingMatrix, reconstructionMatrix).evaluate()
    rmse = RootMeanSquareError(ratingMatrix, reconstructionMatrix).evaluate()
    f1ScoreEvaluator = F1ScoreEvaluator(ratingMatrix, reconstructionMatrix)
    f1Score = f1ScoreEvaluator.evaluate()
    recall = f1ScoreEvaluator.getRecall()
    precision= f1ScoreEvaluator.getPrecision()

    pprint(accuracy)
    pprint(rmse)
    pprint(f1Score)
    pprint(recall)
    pprint(precision)

    # Option 5.2  Ranking Matrix evaluators
    # Option 5.2.1 Confusion Matrix evaluators
    # Evaluate the ranking matrix that was given
    k = 2
    meanPrecisionAtK = PrecisionAtK(ratingMatrix, rankingMatrix, positiveThreshold, k).evaluate()
    meanRecallAtK = RecallAtK(ratingMatrix, rankingMatrix, positiveThreshold, k).evaluate()
    meanAveragePrecisionAtK = MeanAveragePrecisionAtK(ratingMatrix, rankingMatrix, positiveThreshold, k).evaluate()

    pprint(meanRecallAtK) # meanRecallAtK 
    pprint(meanPrecisionAtK) # meanPrecisionAtK 
    pprint(meanAveragePrecisionAtK)

    # Option 5.2.2  Bandit evaluators 
    # TODO: BanditEvaluator, subclasses into 2 different regrets!
    discountFactor = 0.99
    regretBasedOnOptimalRegret = RegretOptimalEvaluator(ratingMatrix, rankingMatrix, discountFactor).evaluate()

    pprint(regretBasedOnOptimalRegret)

    #pprint(regretBasedOnInstantaneousRegret)
    
    print("DONE TESTING")
