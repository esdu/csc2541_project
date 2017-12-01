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

from sclrecommender.analyzer import MatrixAnalyzer

from sclrecommender.parser import ExampleParser
from sclrecommender.parser import MovieLensParser

from sclrecommender.bandit.runner import BanditRunner

#from sclrecommender.bandit.model import UncertaintyModel
from uncertaintyModel import UncertaintyModel
from nnmf import NNMF
from banditChoice import BanditChoice # UCB
from banditChoice2 import BanditChoice2 # Epsilon Greedy
from sclrecommender.bandit.choice import RandomChoice # A random choice
from sclrecommender.bandit.choice import OptimalChoice # The optimal choice
# TODO: Worst choice

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
from sclrecommender.evaluator import RegretInstantaneousEvaluator

import copy 
import numpy as np
import random
import matplotlib.pyplot as plt

'''
def pprint(obj):
    # For debugging, print statements with numpy variable names and shape
    def namestr(obj):
        namespace = globals()
        return [name for name in namespace if namespace[name] is obj]
    # Assumes obj is a numpy array, matrix
    try:
        print(namestr(obj), obj.shape)
    except:
        try:
            print(namestr(obj), ",", len(obj))
        except:
            print(namestr(obj))
    print(obj)
'''
def pprint(obj):
    print(obj)

def runAll(nnmf, ucb, ratingMatrix, trainMatrix, testMatrix, modelName):
    positiveThreshold = 3.0 # Threshold to set prediction to positive labels
    labelTruth = PositiveNegativeMatrix(ratingMatrix, positiveThreshold)

    positiveNegativeMatrix = labelTruth.getPositiveNegativeMatrix()

    pprint(trainMatrix)
    pprint(testMatrix)
    pprint(positiveNegativeMatrix)


    # Step 5: RecommenderAlgorithm
    # Option 5.1: ReconstructionMatrix: Outputs a reconstruction of actual matrix, known as recommenderMatrix

    reconstructionMatrix = ratingMatrix.copy() # TODO: Calculate reconstruction matrix 
    reconstructionPrediction = PositiveNegativeMatrix(reconstructionMatrix, positiveThreshold)
    positiveNegativePredictionMatrix = reconstructionPrediction.getPositiveNegativeMatrix()

    pprint(reconstructionMatrix)
    pprint(positiveNegativePredictionMatrix)

    # Option 5.2: RankingMatrix: Outputs a matrix of ranking for each user or item
    # Bandit Specific, get Legal Move that can be trained on
    legalTrainMask = LegalMoveMaskGenerator(trainMatrix).getMaskCopy()
    legalTestMask = LegalMoveMaskGenerator(testMatrix).getMaskCopy()

    pprint(legalTrainMask)
    pprint(legalTestMask)

    banditRunner = BanditRunner(ratingMatrix.copy(), legalTrainMask.copy(), legalTestMask.copy())
    banditRunner.setUncertaintyModel(nnmf)
    banditRunner.setBanditChoice(ucb)
    
    rankingMatrix = banditRunner.generateRanking()
    orderChoices = banditRunner.getOrderChoices()

    pprint(rankingMatrix)
    pprint(orderChoices)

    # Step 6: Evaluator

    # Option 6.1 Reconstruction Matrix evaluators
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

    # Option 6.2  Ranking Matrix evaluators
    # Option 6.2.1 Confusion Matrix evaluators
    # Evaluate the ranking matrix that was given
    k = 10 # number of items for each user is 20, so should be less than 20 so recall not guaranteed to be 1

    #-----------------------------------------------------------------------
    tempMaxNumUser = 10 # TODO TEMPORARY, FOLLOWS NUMBER IN BANDIT RUNNER
    tempMaxNumItem = 20 # for printing ranking matrix
    print("TEMP SHRINK TO tempMaxNumUser!")
    print(ratingMatrix.shape)
    ratingMatrix = ratingMatrix[:tempMaxNumUser]
    # Choices were made from any position, so can't reduce rating matrix size by tempMaxNumItem
    # ratingMatrix = ratingMatrix[:, :tempMaxNumItem]
    print(ratingMatrix.shape)
    print(legalTestMask.shape)
    legalTestMask = legalTestMask[:tempMaxNumUser]
    # legalTestMask = legalTestMask[:, :tempMaxNumItem]
    print(legalTestMask.shape)
    print(rankingMatrix.shape)
    rankingMatrix = rankingMatrix[:tempMaxNumUser]
    # rankingMatrix = rankingMatrix[:, :tempMaxNumItem]
    print(rankingMatrix.shape)
    print("Ranking matrix for 10 users and 20 items")
    print(rankingMatrix)
    #-------------------------------------------------------------------------------------------------------
    meanPrecisionAtK = PrecisionAtK(ratingMatrix, rankingMatrix, positiveThreshold, k).evaluate()
    meanRecallAtK = RecallAtK(ratingMatrix, rankingMatrix, positiveThreshold, k).evaluate()
    meanAveragePrecisionAtK = MeanAveragePrecisionAtK(ratingMatrix, rankingMatrix, positiveThreshold, k).evaluate()
    print("Model: " + str(modelName))
    print("\nMeanRecallAtK")
    print(meanRecallAtK) # meanRecallAtK 
    print("MeanPrecisionAtK")
    print(meanPrecisionAtK) # meanPrecisionAtK 
    print("MeanAveragePrecisionAtK")
    print(meanAveragePrecisionAtK)

    # Option 6.2.2  Bandit evaluators 
    discountFactor = 0.99
    regretBasedOnOptimalRegret = RegretOptimalEvaluator(ratingMatrix, rankingMatrix, discountFactor).evaluate()
    instantaneousRegret = RegretInstantaneousEvaluator(ratingMatrix, rankingMatrix, discountFactor, legalTestMask, orderChoices)
    regretBasedOnInstantaneousRegret = instantaneousRegret.evaluate()
    cumulativeInstantaneousRegret =  instantaneousRegret.getCumulativeInstantaneousRegret()

    print("RegretBasedOnOptimalRegret")
    pprint(regretBasedOnOptimalRegret)
    print("RegretBasedOnInstantaneousRegret")
    pprint(regretBasedOnInstantaneousRegret)
    print("CumulativeInstantaneousRegret")
    pprint(cumulativeInstantaneousRegret)

    #-----------------------------------------------------------------
    # TEMP FOR DEBUGGING
    matrixAnalyzer.summarize() # TEMP FOR DEBUGGING
    x = list(range(len(cumulativeInstantaneousRegret)))
    y = cumulativeInstantaneousRegret.copy()
    #-----------------------------------------------------------------
    return x, y

        

    #-------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # seedNum =  int(random.random() * 1000)
    # print("SEEDNUM IS", seedNum)
    seedNum = 196
    np.random.seed(seedNum)
    random.seed(seedNum)
    
    # Anything with pprint(numpyVariable) means it is a numpy matrix
    # Step 1: Get data based on dataset specific parser
    # dataDirectory = "sclrecommender/data/movielens/ml-100k"
    dataDirectory ="ml-100k"
    mlp = MovieLensParser(dataDirectory)
    numUser = 10 
    numItem = 10
    exParser = ExampleParser(dataDirectory)
    ratingMatrix = exParser.getRatingMatrix(numUser, numItem)
    ratingMatrix[0][0] = 1.0
    ratingMatrix = mlp.getRatingMatrixCopy()

    # Step 2: Generate both Rating Matrix and Label Matrix for evaluation
    rmTruth = RatingMatrix(ratingMatrix)
    # Step 3 Analyze the rating matrix
    matrixAnalyzer = MatrixAnalyzer(ratingMatrix)
    matrixAnalyzer.summarize()

    # Step 4: Split rating matrix to train and test
    trainSplit = 0.8

    # Step 4.1: Choose splitting procedure

    # Option 4.1.1: Random Split
    randomMaskTrain, randomMaskTest = RandomMaskGenerator(rmTruth.getRatingMatrix(), trainSplit).getMasksCopy()

    #TODO: Option 4.1.2: Split based on time
    #TODO: Option 4.1.3: Split based on cold users
    #TODO: Option 4.1.4: Split based on cold items
   
    # Step 4.2: Apply mask
    rmTrain = copy.deepcopy(rmTruth)
    rmTest = copy.deepcopy(rmTruth)
    rmTrain.applyMask(randomMaskTrain)
    rmTest.applyMask(randomMaskTest)

    trainMatrix = rmTrain.getRatingMatrix()
    testMatrix = rmTest.getRatingMatrix()

    xLabel = 'Exploration Number'
    yLabel = 'Cumulative Instantaneous Regret'
    #----------------------------------------
    um = UncertaintyModel(ratingMatrix.copy())
    optimalChoice = OptimalChoice()
    modelString4 = "Optimal"
    x4, y4 = runAll(um, optimalChoice, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), modelString4)

    plt.plot(x4, y4, label=modelString4)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString4)
    plt.savefig("/home/soon/Desktop/optimalChoices.png")
    plt.clf()
    #----------------------------------------
    nnmf = NNMF(ratingMatrix.copy())
    ucb = BanditChoice()
    #nnmf = UncertaintyModel(ratingMatrix.copy())
    #ucb = RandomChoice()
    modelString1 = "NNMF, UCB"
    x1, y1 = runAll(nnmf, ucb, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), modelString1)

    print("SAVING FIG!")
    plt.plot(x1, y1, label=modelString1)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString1)
    plt.savefig("/home/soon/Desktop/ucbChoices.png")
    plt.clf()
    #----------------------------------------
    nnmf = NNMF(ratingMatrix.copy())
    egreedy = BanditChoice2()
    #nnmf = UncertaintyModel(ratingMatrix.copy())
    #egreedy = RandomChoice()
    modelString2 = "NNMF, Epsilon Greedy"
    x2, y2 = runAll(nnmf, egreedy, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), modelString2)

    plt.plot(x2, y2, label=modelString2)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString2)
    plt.savefig("/home/soon/Desktop/epsilonGreedyChoices.png")
    plt.clf()

    #----------------------------------------
    um = UncertaintyModel(ratingMatrix.copy())
    rc = RandomChoice()
    modelString3 = "Random"
    x3, y3 = runAll(um, rc, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), modelString3)
    plt.plot(x3, y3, label=modelString3)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString3)
    plt.savefig("/home/soon/Desktop/randomChoices.png")
    plt.clf()
    #----------------------------------------
    modelString = "All Models"
    plt.plot(x1, y1, label=modelString1)
    plt.plot(x2, y2, label=modelString2)
    plt.plot(x3, y3, label=modelString3)
    plt.plot(x4, y4, label=modelString4)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString)
    plt.savefig("/home/soon/Desktop/AllInOne.png")
    plt.clf()
    print("DONE SAVING FIG!")

    print("DONE TESTING")
