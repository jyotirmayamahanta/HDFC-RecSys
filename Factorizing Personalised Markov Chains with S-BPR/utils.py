import csv, math
import numpy as np

def sigmoid(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))

def load_data_from_dir(dirname):
    fname_user_idxseq = dirname + '/' + 'idxseq.txt'
    fname_user_list = dirname + '/' + 'user_idx_list.txt'
    fname_item_list = dirname + '/' + 'item_idx_list.txt'
    user_set = load_idx_list_file(fname_user_list)
    item_set = load_idx_list_file(fname_item_list)

    data_list = []
    last_bucket_list = []
    label = 1
    current_user = None
    with open(fname_user_idxseq ,'r') as f:
        for l in f:
            l = [int(s) for s in l.strip().split()]
            user = l[0]
            b_tm1 = list(set(l[1:]))
            
            if user==current_user:
                label +=1
            else :
                if data_list:
                    last_bucket_list.append(data_list[-1])
                label = 1
                current_user = user

            data_list.append((user, label, b_tm1))
            
    
    return data_list, last_bucket_list, user_set, item_set    


def load_idx_list_file(fname, delimiter=','):
    idx_set = set()
    with open(fname, 'r') as f:
        # discard header
        f.readline()

        for l in csv.reader(f, delimiter=delimiter, quotechar='"'):
            idx = int(l[0])
            idx_set.add(idx)
    return idx_set


def data_to_3_list(data_list):
    u_list = []
    i_list = []
    b_tm1_list = []
    max_l = 0
    for d in data_list:
        u_list.append(d[0])
        i_list.append(d[1])
        b_tm1_list.append(d[2])
        if len(d[2]) > max_l:
            max_l = len(d[2])
    for b_tm1 in b_tm1_list:
        b_tm1.extend([-1 for i in range(max_l - len(b_tm1))])
    b_tm1_list = np.array(b_tm1_list)
    
    return (u_list, i_list, b_tm1_list)
