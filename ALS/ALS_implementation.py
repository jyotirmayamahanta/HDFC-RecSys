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

#Data Preprocessing
def create_data(datapath, productpath):
    # Look for files relative to the directory we are running from
    os.chdir(os.path.dirname(sys.argv[0]))
    
    df=pd.read_csv(datapath, nrows=100000)
    df = df[['UserID', 'ProductID', 'Count']]
    #compensate for zero indexing in sparse matrix
    df['UserID'] = df['UserID'] - 1
    df['ProductID'] = df['ProductID'] - 1
    productID_to_name = {}
    name_to_productID = {}
    df2 = pd.read_csv(productpath)
    for index, row in df2.iterrows():
        productID = int(row['ProductID']) - 1
        productName = row['Product']
        productID_to_name[productID] = productName
        name_to_productID[productName] = productID
    
    return df, productID_to_name, name_to_productID
    
#path to data
datapath = '../sales-data/frequency.csv'
productpath = '../sales-data/product-id.csv'
data, id_to_name, name_to_id = create_data(datapath, productpath)

sparse_item_user = sparse.csr_matrix((data['Count'].astype(float), (data['ProductID'], data['UserID'])))
sparse_user_item = sparse.csr_matrix((data['Count'].astype(float), (data['UserID'], data['ProductID'])))

#debug 
sample = sparse_item_user.toarray()

#Building the model
model = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.2, iterations=32)

alpha_val = 2

sparse_item_user.data = (sparse_item_user.data -1)*alpha_val +1
sample = sparse_item_user.toarray()

data_conf = (sparse_item_user* alpha_val).astype('double')

model.fit(data_conf, True)

user_id =   73#57#27#99#30#45
print('User', user_id, 'has made the following purchases:')
user_id -= 1
print("{:<20} {:<10}".format("Product", "Count"))
#print all the items that user has bought
for index, row in data.iterrows():
    if row['UserID'] == user_id:
        pid = int(row['ProductID'])
        count = int(row['Count'])
        print("{:<20} {:<10}".format(id_to_name[pid], count))

N = 5

print('\nHis top', N, 'recommendations are:')
print("{:<20} {:<10}".format("Product", "Score"))

###USING THE MODEL
#Get Recommendations
recommended = model.recommend(user_id, sparse_user_item, N=N, filter_already_liked_items=True)
for item in recommended:
    print("{:<20} {:<10.4f}".format(id_to_name[int(item[0])], item[1]*(1e2)))

'''
print('\n')

#Get similar items
item_id = 19
item_id -=1
n_similar = 5
print('Products most similar to', id_to_name[item_id], 'are:')
print("{:<20} {:<10}".format("Product", "Score"))
similar = model.similar_items(item_id, n_similar)
for item in similar:
    if item[0] != item_id:
        print("{:<20} {:<10.4f}".format(id_to_name[int(item[0])], item[1]*(1e10)))
'''