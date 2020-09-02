#import libraries
import os 
import sys
import random
import numpy as np

import pandas as pd
import scipy.sparse as sparse
import implicit 

my_seed = 10
random.seed(my_seed)
np.random.seed(my_seed)

# All User Defined Parameters are set here ====================================

# Path to data
datapath = '../data-folder/frequency.csv'
productpath = '../data-folder/product-id.csv'

# No. of rows of data to read
limit_rows = True
nrows = 1000000

# User=Defined Model Parameters
alpha_val = 4.5
factors=10
regularization=0.2
iterations=32
# 4.5, 10, 0.2, 32

# Testing Model Parameters
N = 5

#==============================================================================

# Data Preprocessing ==========================================================

def read_data(datapath, productpath, nrows, limit_rows=False):
    # Look for files relative to the directory we are running from
    os.chdir(os.path.dirname(sys.argv[0]))
    
    # Get product-id map ======================================================
    pdf = pd.read_csv(productpath)
    pdf = pdf[['Product','ProductID']]
    # Compensate for zero indexing in sparse matrix
    pdf['ProductID'] = pdf['ProductID'] - 1
    productID_to_name = pdf.set_index('ProductID').to_dict()['Product']
    name_to_productID = pdf.set_index('Product').to_dict()['ProductID']
    
    # Get purchase freq data ==================================================
    if(limit_rows):
        df=pd.read_csv(datapath, nrows=nrows)
    else :
        df=pd.read_csv(datapath)
    #df=pd.read_csv(datapath)
    df = df[['UserID', 'ProductID', 'Count']]
    
    # Compensate for zero indexing in sparse matrix
    df['UserID'] = df['UserID'] - 1
    df['ProductID'] = df['ProductID'] - 1
    
    return df, productID_to_name, name_to_productID

# Load frequency data and product-id maps
data, id_to_name, name_to_id = read_data(datapath, productpath, nrows, limit_rows)

# Creating the Sparse Confidence Matrix =======================================

# Confidence Expression - Quantifying implicit data into confidence values
def conf_exp(x):
    confidence = (x - 1.0)*alpha_val + 1.0
    return confidence

# Convert count data into confidence data
data['Confidence'] = data['Count'].astype('double').apply(lambda x : conf_exp(x))

# Create Confidence Matrix
sparse_item_user = sparse.csr_matrix((data['Confidence'], (data['ProductID'], data['UserID'])))
sparse_user_item = sparse.csr_matrix((data['Confidence'], (data['UserID'], data['ProductID'])))

#debug
#sparse_item_user.data = (sparse_item_user.data -1)*alpha_val +1
#sample = sparse_item_user.toarray()

# Training the model using the Sparse Confidence Matrix =======================

model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
model.fit(sparse_item_user, True)

# Running the model (on replay) ===============================================

while(True):
    temp = input("Input UserID (or enter 'q' to quit): ")
    if temp == 'q':
        break    
    
    try:
        user_id = int(temp)
        user_id -= 1
        
        # Calulate Recommendations (might throw Index Error)
        recommended = model.recommend(user_id, sparse_user_item, N=N, filter_already_liked_items=True)
        
        # Print Purchases
        print('\nUser', user_id+1, 'has made the following purchases:')
        print("{:<20} {:<10}".format("Product", "Count"))
        for index, row in data[data['UserID']==user_id].iterrows():
            pid = int(row['ProductID'])
            count = int(row['Count'])
            print("{:<20} {:<10}".format(id_to_name[pid], count))
        
        print('\nHis top', N, 'recommendations are:')
        print("{:<20} {:<10}".format("Product", "Score"))
        
        # Print Recomendations 
        for item in recommended:
            print("{:<20} {:<10.4f}".format(id_to_name[int(item[0])], item[1]*(1e2)))
            
    except (IndexError, ValueError) :
        print("\nUserID does not exist\n")