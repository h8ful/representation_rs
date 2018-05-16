import numpy as np
import multiprocessing
import os
import warnings

# from joblib import Parallel, delayed
THREAD_NUM = 16
# all out of class method for pairwise auc, because multiprocessing bug does not support classmethod
def heaviside(x):
    ret = 0
    if x > 0:
        ret = 1
    return ret


def user_auc(x):
    rating_all, pred, recomendataion_items = x
    rated = recomendataion_items
    not_rated = (1-rating_all).nonzero()[0]
    auc_u = 0
    for i in rated:
        for j in not_rated:
            auc_u += heaviside(pred[i] - pred[j])
    return auc_u*1.0/(len(rated) * len(not_rated))

def pairwise_auc_multiprocessing(all_rating,test,targets,pred, recommendation,rec_len):
    rating_all = (((all_rating >0).astype(int)  + (test > 0).astype(int)) > 0).astype(int)
    user_profile = []
    # auc_of_targets_ = []
    for u in targets:
        # auc_of_targets_.append(user_auc(rating_all[u], pred[u],test[u].nonzero()[0]))
        user_profile.append((rating_all[u], pred[u],test[u].nonzero()[0]))
    p = multiprocessing.Pool(THREAD_NUM)
    auc_of_targets_ = p.map(user_auc,user_profile)
    p.close()
    # auc_of_targets_ = []
    # for p in user_profile:
    #     auc = user_auc(p)
    #     auc_of_targets_.append(auc)
    return (auc_of_targets_)

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

    # def evaluate(self, test, targets, rec_len):
    #     return self.precision_recall(test, targets, rec_len)


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

    def pairwise_auc(self,test,targets):
        all_rating = (((self.known_ratings > 0).astype(int) + (test > 0).astype(int))>0).astype(int)
        self.auc_of_targets_ = pairwise_auc_multiprocessing(all_rating, test, targets,self.raw_prediction_matrix, self.recommendation_,rec_len=5)
        return np.mean(self.auc_of_targets_)

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
        if "AUC" not in self.scores.keys():
            self.scores['AUC'] = self.pairwise_auc(test, targets)
        self.user_perf_name_ = ["precision","recall","ap", "hit",'F1']
        return self.scores
        #
        # for rec_len in rec_len_list:
        #     if rec_len == 5:
        #         pr = self.precision_recall_hr(test,targets,rec_len=5)
        #         self.scores['prec@5'] = pr[0]
        #         self.scores['reca@5'] = pr[1]
        #         self.scores['hr@5'] = pr[2]
        #     elif rec_len == 1:
        #         pr = self.precision_recall_hr(test,targets,rec_len=1)
        #         self.scores['prec@1'] = pr[0]
        #         self.scores['reca@1'] = pr[1]
        #         self.scores['hr@1'] = pr[2]
        #     else:
        #         print("only rec_len = 5 or rec_len = 1")
        # return self.scores
    
    
    
                    ########################
                    ## START OF IBCF PART ##
                    ########################


from sklearn.metrics.pairwise import pairwise_distances
# from numba import jit
# @jit
# def _asymcos(x,alpha=0.2):
#     numerator = x.dot(x.T)
#     dominator_a = np.power(numerator, alpha)
#     dominator_b = np.power(numerator, (1 - alpha))
#     sim = np.zeros_like(numerator).astype(np.float64)
#     for i in range(numerator.shape[0]):
#         for j in range(numerator.shape[0]):
#             if dominator_a[i, i] ==0 or dominator_b[j, j] == 0:
#                 sim[i,j] = 0
#             else:
#                 sim[i, j] = numerator[i, j] / dominator_a[i, i] / dominator_b[j, j]
#     return sim

def asymcos(x,y,alpha=0.2):
    xy = np.dot(x,y)
    if xy == 0:
        return 0
    else:
        xx = x.dot(x)
        yy = y.dot(y)
        sim = xy*1.0 / np.power(xx,alpha) / np.power(yy,(1-alpha))
            # print ("warning")
            # print('xx',xx)
            # print('yy',yy)
            # print('-'*80)

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
            # import pdb; pdb.set_trace()
            # print("in rec9 cosine, thread num",THREAD_NUM)
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

        # self.predicted_score_ = compute_all_score_multiprocessing(self.input_ratings, self.item_neighbors_, self.similarities_,self.targets)
        # self.predicted_score_ = np.zeros(self.train_ratings.shape)
        # for u in targets:
        #     consumed = set(self.input_ratings[u, :].nonzero()[0])
        #     # self.comsumed_num_[u] = set()
        #     # import pdb; pdb.set_trace()
        #     candinates = list(set([i for k in consumed for i in self.item_neighbors_[k] ]) - (consumed))
        #     # for i in xrange(self.item_num_):
        #     for i in candinates:
        #         self.predicted_score_[u, i] = self.score_pair_ib(user_id=u, item_id=i)
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




class UBCF():
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
        self.user_num_ = profile.shape[0]
        self.similarities_ = np.zeros((self.user_num_, self.user_num_))
        if self.sim == 'dot':
            self.similarities_ = self.profile.dot(self.profile.T)
        elif self.sim == 'asymcos':
            self.similarities_ = self.asymmetric_cosine(self.profile,alpha=0.2)
        else:
            # import pdb; pdb.set_trace()
            self.similarities_ = 1.0 - pairwise_distances(self.profile, metric=self.sim, n_jobs= 8)
        # set similarity to identity to 0
        self.similarities_ = np.multiply( self.similarities_, (1-np.eye(self.similarities_.shape[0])))
        return self


    def find_neighbors(self):
        self.user_neighbors_ = dict()
        for user in range(self.user_num_):
            self.user_neighbors_[user] = np.argpartition(-1 * self.similarities_[user], self.topN)[:self.topN]
        return  self

    def score_pair_ub(self, user_id, item_id):
        neighborhood = self.user_neighbors_[user_id]
        score = (self.similarities_[user_id,neighborhood]).dot(self.known_ratings[neighborhood, item_id])
        return  score

    def fit(self,train_ratings, profile):
        '''
        :param train_ratings: user*user
        :param profile: user based, user*dim
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
        for u in targets:
            consumed = set(self.input_ratings[u, :].nonzero()[0])
            neighbors = self.user_neighbors_[u]
            candinates = list(set(self.train_ratings[neighbors].nonzero()[1]) - consumed)
            # candinates = range(self.train_ratings.shape[1])
            for i in candinates:
                self.predicted_score_[u, i] = self.score_pair_ub(user_id=u, item_id=i)
        return self

    def produce_reclist(self,targets):
        # predict score first, ensure self.predicted_score_ exsit
        if True:
        # try:
            self.rec_ = Rec()
            self.rec_.set_prediction_matrix(self.train_ratings+self.input_ratings,self.predicted_score_)
            self.rec_.produce_rec_list(self.known_ratings,targets)
            self.recommendations_ = self.rec_.recommendation_
            return self
        # except Exception:
        #     raise Exception
    def evaluate(self, test, rec_len):
        self.test = test
        return self.rec_.evaluate(test=test, rec_len=rec_len)
