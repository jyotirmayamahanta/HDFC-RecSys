# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:50:47 2020

@author: Jyotirmaya Mahanta
"""
#import libraries
import itertools
import pandas as pd

class RolloutCombinations:
    
    def createDummyData(self, pids):
        
        # for appending to purchase frequency dataframe
        df = pd.DataFrame(columns = ['UserID', 'ProductID'])
        # for appending to output dataframe
        df2 = pd.DataFrame(columns = ['UserID', 'Pur1', 'Pur2', 'Pur3'])
        
        userid = 1
        
        #single product
        for items in pids:
            # append to df
            data = {'UserID': userid, 'ProductID': items}
            df = df.append(data, ignore_index = True)
            # append to df1
            data2 = {'UserID': userid, 'Pur1': items}
            df2 = df2.append(data2, ignore_index = True)
            
            userid +=1
        
        #two product combinations with repititions
        for items in itertools.combinations_with_replacement(pids, 2):
            for j in range(2):
                # append to df
                data = {'UserID': userid, 'ProductID': items[j]}
                df = df.append(data, ignore_index = True)
            # append to df1
            data2 = {'UserID': userid, 'Pur1': items[0], 'Pur2': items[1]}
            df2 = df2.append(data2, ignore_index = True)
            
            userid +=1
        
        #three product combinations with repitition
        for items in itertools.combinations_with_replacement(pids, 3):
            for j in range(3):
                # append to df
                data = {'UserID': userid, 'ProductID': items[j]}
                df = df.append(data, ignore_index = True)
            # append to df1
            data2 = {'UserID': userid, 'Pur1': items[0], 'Pur2': items[1], 'Pur3': items[2]}
            df2 = df2.append(data2, ignore_index = True)
            
            userid +=1

        # Create the purchase frequency dataset for implicit model implementation (Matrix Factorization with ALS Model)
        df_count = df.groupby(["UserID", "ProductID"]).size().reset_index(name="Count")
        #df_count = df_count.sort_values(['ProductID', 'UserID'])
        
        #df_count.to_csv('frequency_sim.csv', index=False)    # Dump
        #df2.to_csv('output-part-1.csv', index=False)
        
        return df_count, df2