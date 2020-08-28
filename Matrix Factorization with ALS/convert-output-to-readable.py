# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 02:57:54 2020

@author: Jyotirmaya Mahanta
"""

import pandas as pd
import os
import sys

df = pd.read_csv('output.csv')

def create_data(productpath):
    # Look for files relative to the directory we are running from
    os.chdir(os.path.dirname(sys.argv[0]))
    
    
    productID_to_name = {}
    name_to_productID = {}
    df2 = pd.read_csv(productpath)
    for index, row in df2.iterrows():
        productID = int(row['ProductID'])
        productName = row['Product']
        productID_to_name[productID] = productName
        name_to_productID[productName] = productID
    
    return productID_to_name, name_to_productID
    
#path to data
productpath = '../data-folder/product-id.csv'
id_to_name, name_to_id = create_data(productpath)

df['Pur1'] = df['Pur1'].dropna().apply(lambda x: id_to_name[x])
df['Pur2'] = df['Pur2'].dropna().apply(lambda x: id_to_name[x])
df['Pur3'] = df['Pur3'].dropna().apply(lambda x: id_to_name[x])
df['Rec1'] = df['Rec1'].dropna().apply(lambda x: id_to_name[x])
df['Rec2'] = df['Rec2'].dropna().apply(lambda x: id_to_name[x])
df['Rec3'] = df['Rec3'].dropna().apply(lambda x: id_to_name[x])
df['Rec4'] = df['Rec4'].dropna().apply(lambda x: id_to_name[x])
df['Rec5'] = df['Rec5'].dropna().apply(lambda x: id_to_name[x])

df.to_csv('output2.csv', index=False)