import sys, os, pickle
from random import shuffle
from utils import *
from FPMC import FPMC

class values:
    
    def __init__(self):
        self.input_dir = '../data-folder'   # The directory of input
        self.n_epoch = 15                   # number of epoch
        self.n_neg = 5                      # number of neg samples
        self.n_factor = 12                  # dimensions of factorization
        self.learn_rate = 0.01              # learning rate
        self.regular = 0.001                # regularization
        self.N = 10                         # No. of recommendations to output
        self.std = 0.01                     # Standard Deviation of Normal distribution
        
    def attach(self, input_dir, n_epoch, n_neg, n_factor, learn_rate, regular, N):
        self.input_dir = input_dir
        self.n_epoch = n_epoch
        self.n_neg = n_neg
        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular
        self.N = N


args = values()

#args.attach(input_dir= , n_epoch= , n_neg= , n_factor= , learn_rate= , regular= , N= )

f_dir = args.input_dir
data_list, last_bucket_list, user_set, item_set, id_to_item = load_data_from_dir(f_dir)

bucket_pair_list = []
prev_bucket = None
for user, b_num, bucket in data_list:
    if b_num == 1:
        prev_bucket = bucket
        continue
    bucket_pair_list.append((user, prev_bucket, bucket))
    prev_bucket = bucket

shuffle(bucket_pair_list)

#train_ratio = 0.8
#split_idx = int(len(bucket_pair_list) * train_ratio)
#tr_data = bucket_pair_list[:split_idx]
#te_data = bucket_pair_list[split_idx:]
#tr_data = bucket_pair_list[:10000]
tr_data = bucket_pair_list

fpmc = FPMC(n_user=len(user_set), n_item=len(item_set), 
            n_factor=args.n_factor, learn_rate=args.learn_rate, regular=args.regular)
fpmc.user_set = user_set
fpmc.item_set = item_set
fpmc.init_model(args.std)

fpmc.learnSBPR_FPMC(tr_data, n_epoch=args.n_epoch, 
                               neg_batch_size=args.n_neg)

def print_recommendations(test_user):
    print("\nItems bought by user ", test_user, ":")
    items_bought = [i[2][0] for i in data_list if i[0] == test_user]
    for i in items_bought:
        print(id_to_item[i])
        
    print("\nRecommended items:")
    test_user -=1 #compensate for internal id
    topN = fpmc.recommend_topN(test_user, last_bucket_list[test_user][2], args.N)
    
    for i in topN:
        print(id_to_item[i])
        
while(True):
    test_user = input("Enter user id : ")
    if test_user == 'quit':
        break
    
    print_recommendations(int(test_user))