import sys, os, pickle
from random import shuffle
from utils import *
try:
    from FPMC_numba import FPMC
except ImportError:
    from FPMC import FPMC

class values:
    
    def __init__(self):
        self.input_dir = 'data/'          # The directory of input
        self.n_epoch = 15                   # number of epoch
        self.n_neg = 10                     # number of neg samples
        self.n_factor = 12                  # dimensions of factorization
        self.learn_rate = 0.01              # learning rate
        self.regular = 0.001                # regularization
        
    def attach(self, input_dir, n_epoch, n_neg, n_factor, learn_rate, regular):
        self.input_dir = input_dir
        self.n_epoch = n_epoch
        self.n_neg = n_neg
        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular


args = values()

f_dir = args.input_dir

data_list, last_bucket_list, user_set, item_set = load_data_from_dir(f_dir)

bucket_pair_list = []
prev_bucket = None
for user, b_num, bucket in data_list:
    if b_num == 1:
        prev_bucket = bucket
        continue
    bucket_pair_list.append((user, prev_bucket, bucket))
    prev_bucket = bucket

shuffle(bucket_pair_list)

train_ratio = 0.8
split_idx = int(len(bucket_pair_list) * train_ratio)
tr_data = bucket_pair_list[:split_idx]
te_data = bucket_pair_list[split_idx:]

fpmc = FPMC(n_user=max(user_set)+1, n_item=max(item_set)+1, 
            n_factor=args.n_factor, learn_rate=args.learn_rate, regular=args.regular)
fpmc.user_set = user_set
fpmc.item_set = item_set
fpmc.init_model()

acc, mrr = fpmc.learnSBPR_FPMC(tr_data, te_data, n_epoch=args.n_epoch, 
                               neg_batch_size=args.n_neg, eval_per_epoch=False)

print ("Accuracy:%.2f MRR:%.2f" % (acc, mrr))
