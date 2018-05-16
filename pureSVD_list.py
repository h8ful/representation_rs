from scipy import  linalg
import numpy as np
import time
from scipy.io import mmread
import pandas as pd
import argparse
import pickle
import os
import importlib.util
import rec9 as rec
import data_path


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--cv', type=int)
parser.add_argument('--dim-list', type=int, nargs='+')
parser.add_argument('--thread-num', type=int)
parser.add_argument('--log-dir',type=str, default="./")
args = parser.parse_args()

rec.THREAD_NUM = args.thread_num


def svd_on_dim(U,s,Vh, dim,train_data,test_data,targets):
    u2 = U[:, :dim]
    s2 = s[:dim]
    vh2 = Vh[:dim, :]
    pred = np.dot(u2, np.dot(np.diag(s2), vh2))
    r = rec.Rec()
    perf = r.evaluate2(train_data,test_data,targets,pred, [1,5])
    # def evaluate2(self,train,test,targets,pred,rec_len_list):

    return dim,perf


print("reading data...", end=' ')
start_time = time.time()
total_start_time = time.time()
if args.split.startswith('b') or args.split.startswith('d') or args.split.startswith('f'):
    train_path, test_path = data_path.get_path(args.data, args.split, args.cv)
    train_data = mmread(train_path).A
    input_data = train_data
    test_data = mmread(test_path).A
    targets = np.unique(test_data.nonzero()[0])
if args.split.startswith('c') :
    profile_path, input_path,train_path, test_path = data_path.get_path(args.data, args.info_train, args.split, args.cv)
    # target_path = train_path.replace("train","targets")

    train_data = mmread(train_path).A
    input_data = mmread(input_path).A
    profile_data = mmread(profile_path).A
    test_data = mmread(test_path).A
    targets = np.unique(test_data.nonzero()[0])
# raise Exception("aa")
end_time = time.time()
print("load datatime: ", end_time-start_time, "secs")

print("decomposing matrix", end=' ')
U, s, Vh = linalg.svd(train_data, full_matrices=False)
end_time = time.time()
print("decomposing time: ", end_time - start_time, "secs")
for dim in args.dim_list:
    start_time = time.time()
    print("rec on dim %s ..."%(dim), end=' ')
    dim,perf = svd_on_dim(U,s,Vh,dim, train_data,test_data,targets)
    end_time = time.time()
    print("rec time: ", end_time - start_time, "secs")
    param = {'dim':dim,'data':args.data,'split':args.split,'cv':args.cv, 'rec_time':end_time-start_time}
    perf.update(param)
    perf['model'] = "svd"
    print(perf)

    df = pd.DataFrame.from_dict([perf],orient='columns')

    # wirte decoposed matrix, no dim differences
    param_str = ("%s" % param).strip("{}").replace("'", "").replace(":", "_").replace(",", "-").replace(" ", "")
    df.to_csv(os.path.join(args.log_dir, "pureSVD_result_%s.csv"%param_str),index=False)
    model_fh = open(os.path.join(args.log_dir, "pureSVD_model_%s.pkl"%param_str),'wb')
    pickle.dump({"param":param,"U":U, "s":s, "Vh":Vh},model_fh)
    model_fh.close()
    end_time = time.time()
    print("time: ", end_time-start_time, "secs")

