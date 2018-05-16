import sys
import numpy as np
import time
from scipy.io import mmread
import pandas as pd
import argparse
import pickle as pkl
import os
import importlib.util
import rec9 as rec
import data_path
import ntpath
from sklearn.preprocessing import StandardScaler
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--cv', type=int)
parser.add_argument('--topn', type=int)
parser.add_argument('--thread-num',type=int)
parser.add_argument('--log-dir',type=str, default="./")
parser.add_argument('--model-path',type=str, required=True)
parser.add_argument('--model',type=str, required=True)
parser.add_argument('--pca-dim',type=int)
parser.add_argument('--norm', action='store_true', default=False,
                    help='if normalize the ')

args = parser.parse_args()

def read_bprmf(model_path):
    with open(model_path, 'r') as f:
        s = f.read().strip().split('\n')
        # remove header and writer version
        s = s[2:]
        flag = 'default'
        for line in s:
            line_list = line.split()


            if len(line_list) == 2:
                if flag == "default":
                    flag = 'uf'
                    uf_shape=[int(i) for i in line_list]
                    uf_mtx = np.empty(uf_shape)
                    # print(flag)
                    continue
                if flag == 'ib':
                    flag = 'if'
                    if_shape = [int(i) for i in line_list]
                    if_mtx = np.empty(if_shape)
                    # print(flag)
                    continue
            if len(line_list) == 3 :
                if flag == "uf":
                    i,j,v = line_list
                    i,j = int(i),int(j)
                    v = float(v)
                    uf_mtx[i,j] = v
                    # raise Exception('aaa')
                    continue
                if flag == 'if':
                    i,j,v = line_list
                    i,j = int(i),int(j)
                    v = float(v)
                    if_mtx[i,j] = v
                    continue

            if len(line_list) == 0 and flag == 'uf':
                flag = 'ibs'
                print(flag)

                continue
            if len(line_list) == 1 :
                if flag == 'ibs':
                    ib_shape = (int(line_list[0]),1)
                    ib_list = []
                    flag = 'ib'
                    # print(flag)
                    continue
                if flag == 'ib':
                    ib_list.append(float(line_list[0]))
                    continue
        ib = np.array(ib_list).reshape(-1,1)
    return uf_mtx,if_mtx,ib



rec.THREAD_NUM = args.thread_num

print("reading data...", end=' ')
start_time = time.time()
total_start_time = time.time()



profile_path, input_path,train_path, test_path = data_path.get_path(args.data, args.info_train, args.split, args.cv)

train_data = mmread(train_path).A
input_data = mmread(input_path).A
test_data = mmread(test_path).A
targets = np.unique(test_data.nonzero()[0])

profile_data = mmread(args.model_path)
rep_fname = ntpath.basename(args.model_path)
rep_dim = args.pca_dim
rep_param_str = 'repfname_%s-repdim_%s'%(rep_fname,rep_dim)

if args.norm:
    sts = StandardScaler()
    profile_data = sts.fit_transform(profile_data)
    # mean = profile_data.mean(axis=1)
    # var = profile_data.var(axis=1)
    # # profile_data2 = np.empty_like(profile_data)
    # for i in range(profile_data.shape[0]):
    #     var[i] = max(var[i],1e-9)
    #     profile_data[i,:] = (profile_data[i,:] - mean[i])/var[i]
# raise Exception('aa')
end_time = time.time()
print("load datatime: ", end_time-start_time, "secs")
res = []
for sim in ['cosine','asymcos' ,'dot']:

    start_time = time.time()
    print("rec on [sim,topn,data,split,cv] %s ..."%([sim,args.topn,args.data,args.split,args.cv]), end=' ')
    ibcf = rec.IBCF(sim)
    ibcf.fit(train_data,profile=profile_data)

    ibcf.compute_score2(input_data, topN=args.topn, targets=targets)
    ibcf.evaluate(input_data,test_data,targets, rec_len_list=[1,5])
    end_time = time.time()
    print("time: ", end_time-start_time, "secs")
    perf = ibcf.perf_
    # param = {'similarity':sim,'topN':args.topn,'data':args.data,'split':args.split,'cv':args.cv, 'rec_time':end_time-start_time}
    param = {'similarity':sim,'topN':args.topn,'data':args.data,'split':args.split,'cv':args.cv, 'rep_param':rep_param_str, 'dim' : rep_dim,'norm':args.norm}
    perf.update(param)
    perf['model'] = "%s_ibcf"%args.model

    print(ibcf.perf_)
    print("writing results...", end=' ')

    start_time = time.time()
    param_str = ("%s"%param).strip("{}").replace("'","").replace(":","_").replace(",","-").replace(" ","")

    df = pd.DataFrame.from_dict([perf],orient='columns')
    df.to_csv(os.path.join(args.log_dir, "rep_%s_ibcf_result_%s.csv"%(args.model,param_str)),index=False)
    model_fh = open(os.path.join(args.log_dir, "rep_%s_ibcf_model_%s.pkl"%(args.model,param_str)),'wb')
    pkl.dump({"param":param,"predicted_score":ibcf.predicted_score_, "recommendation":ibcf.rec.recommendation_},model_fh)
    model_fh.close()
    end_time = time.time()
    print("time: ", end_time-start_time, "secs")
total_end_time = time.time()

print("done. ", total_end_time - total_start_time , "secs")




