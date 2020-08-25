from SalesData import SalesData
from surprise import SVD
import numpy as np
import random

#get reproducible results
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

def BuildAntiTestSetForUser(testSubject, trainset):
    fill = trainset.global_mean

    anti_testset = []
    
    u = trainset.to_inner_uid(str(testSubject)) # raw_uid is supposed to be string - potential bug?
    
    user_items = set([j for (j, _) in trainset.ur[u]])
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
    return anti_testset

# Pick an arbitrary test subject
testSubject = 93

pl = SalesData()

print("Loading sales data...")
data = pl.loadSalesData()

userRatings = pl.getUserRatings(testSubject)
bought = []

for ratings in userRatings:
    if (float(ratings[1]) > 4.0):
        bought.append(ratings)


print("\nUser ", testSubject, " bought these products:")
for ratings in bought:
    print(pl.getProductName(ratings[0]))

print("\nBuilding recommendation model...")
trainSet = data.build_full_trainset()

algo = SVD()
algo.fit(trainSet)

print("Computing recommendations...")
testSet = BuildAntiTestSetForUser(testSubject, trainSet)
predictions = algo.test(testSet)

recommendations = []

print ("\nWe recommend:")
for userID, productID, actualRating, estimatedRating, _ in predictions:
    intProductID = int(productID)
    recommendations.append((intProductID, estimatedRating))

recommendations.sort(key=lambda x: x[1], reverse=True)

for ratings in recommendations[:4]:
    print(pl.getProductName(ratings[0]), end = ' ')
    #print('{0:8.5f}'.format(ratings[1]))
    print()


