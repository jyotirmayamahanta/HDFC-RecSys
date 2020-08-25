import os
import sys

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import pandas as pd

class SalesData:

    productID_to_name = {}
    name_to_productID = {}
    ratingsPath = '../sales-data/ratings.csv'
    productPath = '../sales-data/product-id.csv'
    
    def loadSalesData(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}

        reader = Reader()
        
        df = pd.read_csv(self.ratingsPath, nrows=100000)
        df = df[['UserID', 'ProductID', 'Rating']]
        #df = df.apply(pd.to_numeric)
        ratingsDataset = Dataset.load_from_df(df, reader=reader)
        
        df2 = pd.read_csv(self.productPath)
        for index, row in df2.iterrows():
            productID = int(row['ProductID'])
            productName = row['Product']
            self.productID_to_name[productID] = productName
            self.name_to_productID[productName] = productID

        return ratingsDataset

    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        
        df = pd.read_csv(self.ratingsPath)
        for index, row in df.iterrows():
            userID = int(row['UserID'])
            if (user == userID):
                productID = int(row['ProductID'])
                rating = float(row['Rating'])
                userRatings.append((productID, rating))
                hitUser = True
            if (hitUser and (user != userID)):
                break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        
        df = pd.read_csv(self.ratingsPath)
        for index, row in df.iterrows():
            productID = int(row['ProductID'])
            ratings[productID] += 1
            
        rank = 1
        for productID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[productID] = rank
            rank += 1
        return rankings
    
    def getProductName(self, productID):
        if productID in self.productID_to_name:
            return self.productID_to_name[productID]
        else:
            return ""
        
    def getproductID(self, productName):
        if productName in self.name_to_productID:
            return self.name_to_productID[productName]
        else:
            return 0