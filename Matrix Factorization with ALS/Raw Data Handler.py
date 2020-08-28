import os
import sys
import pandas as pd

'''
Compatible raw data file name format : {name}-part-{i}.csv
Columns - 'PRODUCT' , 'KEYS'
'''
#==============================================================================
name = 'sales-data'    # Enter name in to above format
num_parts = 4   # Enter the no. of raw data partitions

#==============================================================================

# Look for files relative to the directory we are running from
os.chdir(os.path.dirname(sys.argv[0]))

'''
Compatible raw data file name format : {Name}-part-{i}.csv
Columns - 'PRODUCT' , 'KEYS'
'''
print('Loading data ...\n')

df = pd.read_csv('../data-folder/'+name+'/'+name+'-part-1.csv')    # Read first partition
print(len(df), 'Transactions loaded from partition 1')

# Append left partitions
if num_parts > 1:
    for i in range(2, num_parts+1):
        temp = pd.read_csv('../data-folder/'+name+'/'+name+'-part-'+str(i)+'.csv')
        print(len(temp), 'Transactions loaded from partition ', i)
        df = df.append(temp, ignore_index = True)

# Rename columns 
cols = {'KEY':'UserID', 'PRODUCT':'Product'}
df.rename(columns = cols, inplace = True) 

print('\nAssigning IDs to products ... ', end='')

os.chdir('../data-folder')

# Assign IDs to Product - Numbered alphabetically starting from 1
df2 = df.groupby(['Product']).count().reset_index().sort_values('Product')
df2['ProductID'] = [x for x in range(1,df2.shape[0]+1)]

# Rename 'UserID' aggregate to 'Total Sales
df2.rename(columns = {'UserID':'Total Sales'}, inplace=True)

# Dump product-id info into csv file
df3 = df2[['ProductID', 'Product']]
df3.to_csv('product-id.csv', index=False)

print('Done')

# Sort by total sales
df4 = df2.sort_values('Total Sales', ascending=False).reset_index()

# Dump total product sales stats into csv file
df4 = df4[['ProductID', 'Product', 'Total Sales']]
df4.to_csv('total-product-sales-sorted.csv', index=False)

print('\nFormatting data ... ', end='')
# Assign Product IDs to the main dataset
df_merge = pd.merge(df, df3, on='Product').sort_values('UserID')

# Delete the 'Product' column
del df_merge['Product']

# Create the purchase frequency dataset for implicit model implementation (Matrix Factorization with ALS Model)
df_count = df_merge.groupby(["UserID", "ProductID"]).size().reset_index(name="Count")
#df_count = df_count.sort_values(['ProductID', 'UserID'])
df_count.to_csv('frequency.csv', index=False)    # Dump

print('Done')

files = ['frequency.csv','product-id.csv','total-product-sales-sorted.csv']
details = ['Formatted data', 'Product-ID Catalog', 'Total Product Sales' ]

print('\nList of files created in /data-folder directory:\n')
print("{:<20}  {:<10}".format("Details", "Files"))

for i in range(len(files)):
    print("{:<20}  {:<10}".format(details[i], files[i]))
    

'''
# Create the ratings dataset for explicit model implementation (Collaborative Filtering Models)
df_merge.drop_duplicates(subset =['ProductID', 'UserID'], keep = 'first', inplace = True)    # Remove duplicates transactions of same user & same product 
df_merge['Rating'] = 5    # Assign dummy ratings
df_merge.to_csv('ratings.csv', index=False)    # Dump
'''