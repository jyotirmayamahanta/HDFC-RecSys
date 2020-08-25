from SalesData import SalesData
from surprise import KNNBasic
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadProductLensData():
    p1 = SalesData()
    print("Loading product data...")
    data = p1.loadSalesData()
    print("\nComputing product popularity ranks so we can measure novelty later...")
    rankings = p1.getPopularityRanks()
    return (p1, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(p1, evaluationData, rankings) = LoadProductLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# User-based KNN
UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN, "User KNN")

# Item-based KNN
ItemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN, "Item KNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(True)

testSubject = 8
evaluator.SampleTopNRecs(p1, testSubject)
