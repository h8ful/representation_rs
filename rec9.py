import multiprocessing
import os
import warnings

import numpy as np

THREAD_NUM = 16
# all out of class method for pairwise auc, because multiprocessing bug does not support classmethod
def heaviside(x):
    ret = 0
    if x > 0:
        ret = 1
    return ret

def precision_recall_ap_hr_f1(x):
    uid, rec_list, true_list, rec_len = x
    hit = {}
    precision ={}
    recall = {}
    ap = {}
    f1_score = {}
    true_list_len = len(true_list)
    for i in range(1,rec_len+1):
        tp = set(rec_list[:i]).intersection(true_list)
        precision[i] = len(tp)*1.0/i
        recall[i] = len(tp)*1.0 / true_list_len
        hit[i] = 1 if len(tp) > 0 else 0
        ap[i] = np.mean([precision[j] for j in range(1,i+1)])
        f1_score[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i]) if (precision[i]+recall[i]) != 0 else 0
    return uid, precision,recall,ap, hit, f1_score
def precision_recall_ap_hr_mp(targets,rec_lists, test_mat,rec_len):
    params = []
    for u in targets:
        true_list = test_mat[u].nonzero()[0]
        params.append((u,rec_lists[u],true_list,rec_len))
    pool = multiprocessing.Pool(THREAD_NUM)
    res = pool.map(precision_recall_ap_hr_f1,params)
    pool.close()
    # res = []
    # for p in params:
    #     res.append(precision_recall_ap_hr(p))

    user_perf = {}
    for x in res:
        uid,precision,recall,ap, hit,f1_score = x
        user_perf[uid] = (precision,recall,ap, hit,f1_score)
    def compute_mean(idx,user_perf):
        mean_perf = {}
        tmp_perf = [t[idx] for t in user_perf.values() ]
        for i in range(1,rec_len+1):
            mean_perf[i] = np.mean([p[i] for p in tmp_perf])
        return mean_perf
    mean_precision = compute_mean(0,user_perf)
    mean_recall = compute_mean(1,user_perf)
    mean_ap = compute_mean(2,user_perf)
    mean_hr = compute_mean(3,user_perf)
    mean_f1_score = compute_mean(4,user_perf)
    return user_perf, mean_precision,mean_recall,mean_ap,mean_hr,mean_f1_score

class Rec():
    def __init__(self,filter_known = True):
        self.default_rec_len = 10
        self.filter_known = filter_known

        
    def set_prediction_matrix(self,known_ratings, prediction_matrix):
        '''
        :param prediction_matrix: user*item
        :return:
        '''
        # self.prediction_matrix = prediction_matrix

        self.known_ratings = known_ratings
        self.raw_prediction_matrix = prediction_matrix
        self.item_num = self.raw_prediction_matrix.shape[1]
        self.user_num = self.raw_prediction_matrix.shape[0]
        return self
    
    def produce_rec_list(self, targets):
        self.targets = targets
        prediction_matrix = self.raw_prediction_matrix.copy()
        if self.filter_known == True:
            prediction_matrix = np.multiply(prediction_matrix, 1-self.known_ratings)

        self.recommendation_ = dict()
        for user in targets:
            # find the disordered recommendation list
            disorder_idx = np.argpartition(prediction_matrix[user, :], self.item_num - self.default_rec_len)[
                         -1 * self.default_rec_len:]
            # order the recomendataion list, descending order
            # import pdb; pdb.set_trace();
            tmp = np.argsort(-1 * prediction_matrix[user,disorder_idx])
            self.recommendation_[user] = list(( disorder_idx[tmp] ))
        return self

    def precision_recall_hr(self, test, targets, rec_len):
        '''
                :param test: user*item
                :return: self
                '''
        if rec_len > self.default_rec_len:
            raise AttributeError("rec_len > default_rec_len." )
        self.rec_len = rec_len

        self.user_perf = dict()
        self.rec_list_ = dict()
        self.test = test
        hit = 0
        for u in targets:
            y_true = test[u].nonzero()[0]
            # if u == 8: print ('\n',u,y_true )
            # if len(y_true) == 0: continue

            y_pred = self.recommendation_[u][:self.rec_len]
            self.rec_list_[u] = y_pred
            right_rec = len(set(y_true).intersection(set(y_pred))) * 1.0
            hit = 1 if right_rec > 0 else 0
            precision = right_rec / self.rec_len
            # if user has no rating in test set, ignor him
            recall = right_rec / len(y_true)
            self.user_perf[u] = (precision, recall, right_rec)
        self.precision_ = np.average([i[0] for i in list(self.user_perf.values())])
        self.recall_ = np.average([i[1] for i in list(self.user_perf.values())])
        self.hr_ = hit / len(targets)
        return self.precision_, self.recall_, self.hr_

    def evaluate2(self,train,test,targets,pred,rec_len_list):
        self.set_prediction_matrix(train, pred)
        self.produce_rec_list( targets)
        self.scores = {}
        recommendations = self.recommendation_
        user_perf, mean_precision, mean_recall, mean_ap, mean_hr,mean_f1_score = precision_recall_ap_hr_mp(targets,self.recommendation_, test,max(rec_len_list))
        self.user_perf_ = user_perf
        self.mean_precision_ = mean_precision
        self.mean_recall_ = mean_recall
        self.mean_ap_ = mean_ap
        self.mean_hr_ = mean_hr
        self.mean_f1_score = mean_f1_score
        def extract_score(name,perf):
            for i in perf.keys():
                # i is the length of rec list
                key = name+"@"+str(i)
                self.scores[key] = perf[i]
        for name,perf in zip(['prec','reca','MAP','HR',"F1"],[mean_precision, mean_recall, mean_ap, mean_hr,mean_f1_score]):
            extract_score(name,perf)
        self.user_perf_name_ = ["precision","recall","ap", "hit",'F1']
        return self.scores

    
    
    
                    ########################
                    ## START OF IBCF PART ##
                    ########################


from sklearn.metrics.pairwise import pairwise_distances
def asymcos(x,y,alpha=0.2):
    xy = np.dot(x,y)
    if xy == 0:
        return 0
    else:
        xx = x.dot(x)
        yy = y.dot(y)
        sim = xy*1.0 / np.power(xx,alpha) / np.power(yy,(1-alpha))

        return sim
from sklearn import metrics

def asymcos_sk16(mat):
    sim = metrics.pairwise_distances(mat,metric=asymcos ,n_jobs=THREAD_NUM)
    return sim
def _asymcos(x,alpha=0.2):
    return asymcos_sk16(x)

def find_neighbors_mp(similarities, items, topN):
    # print('find_neighbors_mp',os.getpid())

    tmp_item_neighbors = dict()

    # items_similarity = [(i, similarities[i],topN) for i in items]
    res = []
    pool = multiprocessing.Pool(THREAD_NUM)
    # p = multiprocessing.Pool(THREAD_NUM)
    # auc_of_targets_ = p.map(user_auc2,user_profile)
    # p.close()
    item_params = []
    for i in items:
        item_params.append((i, similarities[i],topN))
    res = pool.map(find_neighbor_single, item_params)

    pool.close()
    pool.join()
    tmp_item_neighbors = {}
    for r in res:
        item, neighbor = r
        tmp_item_neighbors[item] = neighbor

    # for item in range(self.item_num_):
        # self.item_neighbors_[item] = np.argpartition(-1 * self.similarities_[item], self.topN)[:self.topN]
    return  tmp_item_neighbors


def find_neighbor_single(x):
    # print('find_neighbor_single',os.getpid())
    item, similaritiy, topN = x
    neighbor = np.argpartition(-1 * similaritiy, topN)[:topN]
    return (item,neighbor)


def score_mp(user_list,ratings, neighborhood, similarities):
    params = []
    res = []
    for u in user_list:
        user_rating = ratings[u]
        params.append((u, user_rating, neighborhood, similarities ))
        # print(nei)
        # res.append(score_by_user((u, user_rating, neighborhood, similarities)))
    pool = multiprocessing.Pool(THREAD_NUM )
    res = pool.map(score_by_user, params)
    pool.close()
    predicted_matrix = np.zeros_like(ratings)
    for t in res:
        u,predicted_vec = t
        predicted_matrix[u] = predicted_vec
    return predicted_matrix


def score_by_user(x):
    u, user_rating, neighborhood, similarities = x
    # print("score by user", os.getpid())
    candinates = ((1-user_rating)>0.5).astype(int).nonzero()[0]
    predicted_rating = np.zeros_like(user_rating)
    for i in candinates:
        neighborhood_item = neighborhood[i]
        similarity_row = similarities[i]

        predicted_rating[i] = np.sum( user_rating[neighborhood_item]*similarity_row[neighborhood_item])
    return  u,predicted_rating


class IBCF():
    def __init__(self, sim):
        self.sim = sim


    def asymmetric_cosine(self, x, alpha=0.2):
        '''
        :param x:
        :return:
        '''
        return _asymcos(x,alpha=0.2)

    def compute_similarity(self, profile):
        '''

        :param profile: each row is a profile vector, r*c size
        :return: similarity matrix, r*r size
        '''
        self.item_num_ = profile.shape[0]
        self.similarities_ = np.zeros((self.item_num_, self.item_num_))
        if self.sim == 'dot':
            self.similarities_ = self.profile.dot(self.profile.T)
        elif self.sim == 'asymcos':
            self.similarities_ = self.asymmetric_cosine(self.profile,alpha=0.2)
        else:
            self.similarities_ = 1.0 - pairwise_distances(self.profile, metric=self.sim, n_jobs= THREAD_NUM)
        # set similarity to identity to 0
        self.similarities_ = np.multiply(   self.similarities_, (1-np.eye(self.similarities_.shape[0])))
        return self


    def find_neighbors(self):
        self.item_neighbors_ = dict()
        for item in range(self.item_num_):
            self.item_neighbors_[item] = np.argpartition(-1 * self.similarities_[item], self.topN)[:self.topN]
        return  self


    def compute_score2(self,input_ratings, topN, targets):
        # print("in class, find neighbors")

        self.input_ratings = input_ratings
        self.topN = topN
        self.item_neighbors_ = find_neighbors_mp(self.similarities_,range(self.item_num_),topN)
        # self.known_ratings = self.train_ratings + self.input_ratings
        self.targets = targets
        self.comsumed_num_ = {}
        self.candinates_num_ = {}

        self.predicted_score_ = score_mp(targets,input_ratings,self.item_neighbors_, self.similarities_)
        return self





    def score_pair_ib(self, user_id, item_id):
        neighborhood = self.item_neighbors_[item_id]
        score = self.input_ratings[user_id, neighborhood].dot(self.similarities_[item_id,neighborhood])
        return  score



    def fit(self,train_ratings, profile):
        '''
        :param train_ratings: user*item
        :param profile: item based, item*dim
        :return: self
        '''
        self.train_ratings = train_ratings
        self.profile = profile

        self.compute_similarity(profile)
        return self

    def compute_score(self,input_ratings, topN, targets):
        self.input_ratings = input_ratings
        self.topN = topN
        self.find_neighbors()
        self.known_ratings = self.train_ratings + self.input_ratings
        self.comsumed_num_ = {}
        self.candinates_num_ = {}

        self.predicted_score_ = np.zeros(self.train_ratings.shape)
        pool = multiprocessing.Pool(THREAD_NUM)

        # for u in targets:
        #     consumed = set(self.input_ratings[u, :].nonzero()[0])
        #     # self.comsumed_num_[u] = set()
        #     # import pdb; pdb.set_trace()
        #     candinates = list(set([i for k in consumed for i in self.item_neighbors_[k] ]) - (consumed))
        #     # for i in xrange(self.item_num_):
        #     for i in candinates:
        #         self.predicted_score_[u, i] = self.score_pair_ib(user_id=u, item_id=i)
        return self



    # def score_pair_ib(self, user_id, item_id):
    #     neighborhood = self.item_neighbors_[item_id]
    #     score = self.input_ratings[user_id, neighborhood].dot(self.similarities_[item_id,neighborhood])
    #     return  score


    def evaluate(self,train,test,targets,rec_len_list):
        self.rec =Rec()
        self.perf_ =  self.rec.evaluate2(train,test,targets,self.predicted_score_,rec_len_list=rec_len_list)
        return self.perf_

