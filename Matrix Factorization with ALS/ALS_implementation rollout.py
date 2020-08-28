#import libraries
import os 
import sys
import random
import numpy as np

import pandas as pd
import scipy.sparse as sparse
import implicit 

from RolloutCombinations import RolloutCombinations

my_seed = 10
random.seed(my_seed)
np.random.seed(my_seed)

# All User Defined Parameters are set here ====================================

# Path to data
datapath = '../data-folder/frequency.csv'
productpath = '../data-folder/product-id.csv'

# No. of rows of data to read
limit_rows = True
nrows = 100000

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

# Create Dummy Data & output part 1
dummy = RolloutCombinations()
dummy_data, output_prt1 = dummy.createDummyData(list(range(1, len(id_to_name)+1)))

last = int(data.iloc[-1]['UserID'])

#debug
#print(last)
#print(dummy_data.iloc[0]['UserID'])

# Compensate for zero indexing in sparse matrix for dummy data
dummy_data['ProductID'] = dummy_data['ProductID'] - 1

# Correct user ids of dummy data
dummy_data['UserID'] = dummy_data['UserID'] + last
    
# add both datasets
data = data.append(dummy_data, ignore_index = True)

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

# Exporting the output ========================================================

output_prt2 = pd.DataFrame(columns=['UserID',
                             'Rec1',
                             'Rec2',
                             'Rec3',
                             'Rec4',
                             'Rec5'])

uid_ = 1
start = last + 1

for uid in range(start, int(data.iloc[-1]['UserID'])+1):
    rec = model.recommend(uid, sparse_user_item, N=N, filter_already_liked_items=True)
    load = {'UserID': uid_,
           'Rec1': rec[0][0]+1,
           'Rec2': rec[1][0]+1,
           'Rec3': rec[2][0]+1,
           'Rec4': rec[3][0]+1,
           'Rec5': rec[4][0]+1}
    uid_ += 1
    output_prt2 = output_prt2.append(load, ignore_index=True)


output_in_id = pd.merge(output_prt1, output_prt2, on='UserID')

#output.to_csv('../data-folder/output-in-id.csv', index = False)
output_in_name = output_in_id.copy()

output_in_name['Pur1'] = output_in_name['Pur1'].dropna().apply(lambda x: id_to_name[x-1])
output_in_name['Pur2'] = output_in_name['Pur2'].dropna().apply(lambda x: id_to_name[x-1])
output_in_name['Pur3'] = output_in_name['Pur3'].dropna().apply(lambda x: id_to_name[x-1])
output_in_name['Rec1'] = output_in_name['Rec1'].dropna().apply(lambda x: id_to_name[x-1])
output_in_name['Rec2'] = output_in_name['Rec2'].dropna().apply(lambda x: id_to_name[x-1])
output_in_name['Rec3'] = output_in_name['Rec3'].dropna().apply(lambda x: id_to_name[x-1])
output_in_name['Rec4'] = output_in_name['Rec4'].dropna().apply(lambda x: id_to_name[x-1])
output_in_name['Rec5'] = output_in_name['Rec5'].dropna().apply(lambda x: id_to_name[x-1])


os.mkdir('../data-folder/roll-combinations')
output_in_id.to_csv('../data-folder/roll-combinations/recommendations-output-in-id.csv', index=False)
output_in_name.to_csv('../data-folder/roll-combinations/recommendations-output-in-name.csv', index=False)