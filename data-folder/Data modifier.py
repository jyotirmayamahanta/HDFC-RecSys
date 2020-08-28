import pandas as pd

'''
Compatible raw data file name format : {Name}-part-{i}.csv
Columns - 'PRODUCT' , 'KEYS'
'''
name = 'sales-data'    # Enter Name in to above format
num_parts = 1   # Enter the no. of raw data partitions

df = pd.read_csv(name + '-part-1.csv')    # Read first partitiion

# Append left partitions
if num_parts > 1:
    for i in range(2, num_parts+1):
        temp = pd.read_csv(name+'-part-'+str(i)+'.csv')
        df = df.append(temp, ignore_index = True)

# Rename columns 
cols = {'KEY':'UserID', 'PRODUCT':'Product'}
df.rename(columns = cols, inplace = True) 

# Assign IDs to Product - Numbered alphabetically starting from 1
df2 = df.groupby(['Product']).count().reset_index().sort_values('Product')
df2['ProductID'] = [x for x in range(1,df2.shape[0]+1)]

# Rename 'UserID' aggregate to 'Total Sales
df2.rename(columns = {'UserID':'Total Sales'}, inplace=True)

# Dump product-id info into csv file
df3 = df2[['ProductID', 'Product']]
df3.to_csv('product-id.csv', index=False)

# Sort by total sales
df4 = df2.sort_values('Total Sales', ascending=False).reset_index()

# Dump total product sales stats into csv file
df4 = df4[['ProductID', 'Product', 'Total Sales']]
df4.to_csv('total-product-sales-sorted.csv', index=False)

# Assign Product IDs to the main dataset
df_merge = pd.merge(df, df3, on='Product').sort_values('UserID')

# Delete the 'Product' column
del df_merge['Product']

# Create the purchase frequency dataset for implicit model implementation (Matrix Factorization with ALS Model)
df_count = df_merge.groupby(["UserID", "ProductID"]).size().reset_index(name="Count")
#df_count = df_count.sort_values(['ProductID', 'UserID'])
df_count.to_csv('frequency.csv', index=False)    # Dump

'''
# Create the ratings dataset for explicit model implementation (Collaborative Filtering Models)
df_merge.drop_duplicates(subset =['ProductID', 'UserID'], keep = 'first', inplace = True)    # Remove duplicates transactions of same user & same product 
df_merge['Rating'] = 5    # Assign dummy ratings
df_merge.to_csv('ratings.csv', index=False)    # Dump
'''