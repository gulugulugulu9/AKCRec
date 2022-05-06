import tensorflow as tf
import  random
import numpy as np
from sklearn.metrics import roc_auc_score

# 对每个用户随机从90-99随机取一个，跟指标的topk进行比对。
def shuffleConstant(length, k=100):
    index = random.randint(90, 99)
    y = tf.constant(index, shape=[1, k])
    for i in range(length - 1):
        index = random.randint(90, 99)
        x = tf.constant(index, shape=[1, k])
        y = tf.concat([x, y], axis=0)
    return y





def loss(rating,rate):
    err = tf.square(tf.subtract(rating,rate))
    los = tf.reduce_sum(err)
    return los

def auc(rate, negative, length): #user dim
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, 100).indices
    where = tf.where(tf.equal(topk, tf.constant(99,shape=[length,100])))
    auc = tf.split(where,num_or_size_splits=2,axis=1)[1]
    ran_auc = tf.Variable(tf.random_uniform(shape=[length, 1], minval=0, maxval=100, dtype=tf.int64))
    auc = tf.reduce_mean(tf.cast(tf.less(auc - ran_auc, 0), dtype=tf.float32))
    return auc



def aucChange_2(rate, negative, length): #user dim
    # y_scores = tf.reshape(tf.gather_nd(rate, negative),shape=(1, -1))  # 后10个是正例
    # y_trues = np.array([([0] * 90 + [1] * 10)] * length)
    # y_trues = y_trues.reshape(1, -1)
    # #auc = roc_auc_score(y_trues[0], y_scores[0].eval())
    
    
    # 用不到，先放这里，对后续没影响。
    su = tf.constant(1)
    return su


# 这个是negative固定10个正例的。
# def aucChange_1(rate, negative, length):  # user dim
#     # 要获得值的下标索引
#     test = tf.gather_nd(rate, negative) # 用索引获取值
#     topk = tf.nn.top_k(test, 100).indices
#     positiveSample = tf.random_uniform(shape=[length, 1], minval=90, maxval=100, dtype=tf.int32)
#     negativeSample = tf.random_uniform(shape=[length, 1], minval=0, maxval=90, dtype=tf.int32)
#     wherePositive = tf.where(tf.equal(topk, positiveSample))  # where函数将返回其中为true的元素的索引, ## 用值获取索引
#     whereNegative = tf.where(tf.equal(topk, negativeSample))
#     aucPositive = tf.split(wherePositive, num_or_size_splits=2, axis=1)[1]
#     aucNegative = tf.split(whereNegative, num_or_size_splits=2, axis=1)[1]
#     auc = tf.reduce_mean(tf.cast(tf.less(aucPositive - aucNegative, 0), dtype=tf.float32))
#     return auc


# 不一定是10

#不固定10个
def aucChange_1(rate, negative, length, negative_num):  # user dim
    # 要获得值的下标索引
    test = tf.gather_nd(rate, negative) # 用索引获取值
    topk = tf.nn.top_k(test, 100).indices
    # positiveSample = 随机从当前用户正例中选取一个正例。

    #  100个交互中，首先是负例，接下来是正例。
    # 随机一个正例索引 =            正例个数          *     (0到1 float)           +              负例个数
    positiveSample = tf.cast(tf.ceil((99 - negative_num) * tf.random_uniform(shape=[length, 1], minval=0, dtype=tf.float32) + negative_num), dtype=tf.int32)
    # 随机从0-negative_num中选取一个负例索引。
    negativeSample = tf.cast(tf.ceil(negative_num * tf.random_uniform(shape=[length, 1], minval=0, dtype=tf.float32)), dtype=tf.int32)
    #查找随机选取的正例，负例位置。
    wherePositive = tf.where(tf.equal(topk, positiveSample))  # where函数将返回其中为true的元素的索引, ## 用值获取索引
    whereNegative = tf.where(tf.equal(topk, negativeSample))
    #比如用户0随机后的选择：   正例是索引2，负例是索引3
    #                        [0 2]       [0 3]
    # split后就是                2           3
    # 即正例分数比负例分数高。tf.less(aucPositive - aucNegative, 0) = (2 - 3) < 0 = true;
    aucPositive = tf.split(wherePositive, num_or_size_splits=2, axis=1)[1]
    aucNegative = tf.split(whereNegative, num_or_size_splits=2, axis=1)[1]
    auc = tf.reduce_mean(tf.cast(tf.less(aucPositive - aucNegative, 0), dtype=tf.float32))
    return auc



def hr(rate, negative, length, k=5):
    # 只需要值是否在就OK
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, k).indices
    isIn = tf.cast(tf.equal(topk, 99), dtype=tf.float32)
    row = tf.reduce_sum(isIn, axis=1)
    all = tf.reduce_sum(row)
    return all/length

# 固定10个正例
# def hrNew(rate, negative, length, k=5):
#     test = tf.gather_nd(rate, negative)
#     topk = tf.nn.top_k(test, k).indices # 取了前K个
#     isIn = tf.cast(tf.greater(topk, 89), dtype=tf.float32)
#     # isIn = tf.cast(tf.greater(topk, 99-k), dtype=tf.float32)# 前K个中值大于99-K的个数，即命中。
#     row = tf.reduce_sum(isIn, axis=1) # [2, 0, 3, 1, 0]
#     none_zero = tf.count_nonzero(row, dtype=tf.int32)
#     return none_zero / length

# 不一定10个正例

# 不固定10个
def hrNew(rate, negative, length, negative_num, k=5):
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, k).indices # 取了前K个

    # topk中只要有索引大于negative_num，则就是正例。
    isIn = tf.cast(tf.greater(topk, negative_num), dtype=tf.float32)
    # 将用户所有命中正例加起来
    row = tf.reduce_sum(isIn, axis=1)
    # 非0的用户就是HR命中了。
    none_zero = tf.count_nonzero(row, dtype=tf.int32)
    return none_zero / length

# 原始mrr指标
# def mrr(rate, negative, length):
#     test = tf.gather_nd(rate, negative)
#     topk = tf.nn.top_k(test, 100).indices
#     mrr_ = tf.reduce_sum(1 / tf.add(tf.split(value=tf.where(tf.equal(topk, tf.constant(99, shape=[length, 100]))),
#                                              num_or_size_splits=2, axis=1)[1], 1))
#     mrr = mrr_/length
#     return mrr

# 固定10个
# def mrrChange_1(rate, negative, length):
#     test = tf.gather_nd(rate, negative)
#     topk = tf.nn.top_k(test, 100).indices
#     # 只要大于89，90-99之间都变为1
#     # [[0, 1, 1, 0, 0],
#     #  [1, 0, 1, 1, 0]]现在要得到每个用户第一个正例推荐的位置
#     # where:[[0,1],[0,2],[1,0],[1,2],[1,3]]
#     where = tf.where(tf.cast(tf.greater(topk, 89), dtype=tf.float32))
#     # 每个用户对应10个正例，都转化为一维矩阵，
#     where_index = tf.reduce_sum(tf.split(value=where, num_or_size_splits=2, axis=1)[1], axis=1)
#     mrr_ = tf.reduce_sum(1 / tf.add(where_index[::10], 1))
#     mrr = mrr_/length
#     return mrr


# 不固定10个
# def mrrChange_1(rate, negative, length, negative_num):
#     test = tf.gather_nd(rate, negative)
#     topk = tf.nn.top_k(test, 100).indices
#     # [[0, 1, 1, 0, 0],
#     #  [1, 0, 1, 1, 0]]现在要得到每个用户第一个正例推荐的位置
#
#     # where:[[0,1],[0,2],[1,0],[1,2],[1,3]]
#     where = tf.where(tf.cast(tf.greater(topk, negative_num), dtype=tf.float32))
#     # where_index = [1, 2, 0, 2, 3]
#     where_index = tf.reduce_sum(tf.split(value=where, num_or_size_splits=2, axis=1)[1], axis=1)
#
#     # 每个用户对应的正例，都转化为一维矩阵，
#     # [2, 3, 4]
#     positive_num = (99 - negative_num).flatten()
#     for i in range(len(positive_num)):
#         temp = positive_num[i]
#         positive_num[i] = nums
#         nums += temp
#     # [0, 2, 5]
#
#     mrr_ = tf.reduce_sum(1 / tf.add(tf.gather(where_index, positive_num), 1))
#     mrr = mrr_/length
#     return mrr




def ndcg(rate, negative, length, k=5):
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, k).indices
    n = tf.split(value=tf.where(tf.equal(topk, tf.constant(99, shape=[length, k]))), num_or_size_splits=2, axis=1)[1]

    ndcg = tf.reduce_sum(tf.log(2.0) / tf.log(tf.cast(tf.add(n, tf.constant(2, dtype=tf.int64)),
                                                      dtype=tf.float32)))/length

    return ndcg


# 固定10个正例
# def ndcgChange_1(rate, negative, length, k=5):
#     test = tf.gather_nd(rate, negative)
#     topk = tf.nn.top_k(test, k).indices
#     isIn = tf.cast(tf.greater(topk, 99 - k), dtype=tf.float32)# [1 0 1 1 0 ]
#     dcg = tf.reduce_sum((np.power(2, isIn)-1) / (np.log(np.arange(2, k+2))/np.log(2)), axis=1)
#     idcg = tf.reduce_sum((np.power(2, tf.sort(isIn, direction='DESCENDING'))-1) / (np.log(np.arange(2, k+2))/np.log(2)), axis=1)
#     ndcg_score = tf.divide(dcg, idcg)
#     # # 这里可能有0/0的情况，nan,将nan变为0就OK
#     ndcg_score = tf.reduce_sum(tf.where(tf.is_nan(ndcg_score), tf.zeros_like(ndcg_score), ndcg_score))
#     return ndcg_score/length


# 不固定10个正例
def ndcgChange_1(rate, negative, length, negative_num, k=5):
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, k).indices

    # 下述是基于这样一个考虑：
    # 若一个用户正例只有99，98，97即三个正例。
    # 则对这个用户在@5的时候就不能说索引大于94的就是正例，应该是大于96的是正例，这个96就是这个用户的negative_num。
    # 所以判断就是max(99 - k, negative_num)
    k_array = np.array([99 - k] * length).reshape(length, 1)

    # positiveIndex就是对每个用户来说在当前@K中，具体每个用户 大于什么数字才算做正例。
    positiveIndex = tf.where(tf.greater(negative_num, k_array), negative_num, k_array)

    # 大于positiveIndex的全部化作1，否则为0。 同上述例子，若一个用户TOP5是[98,87,42,99,97] => [1,0,0,1,1]
    isIn = tf.cast(tf.greater(topk, positiveIndex), dtype=tf.float32)

    # (np.log(np.arange(2, k+2))/np.log(2)) 是换底公式
    # np.arange(2, k+2)是因为公式是从i + 1开始，即就是从2开始。
    dcg = tf.reduce_sum((np.power(2, isIn)-1) / (np.log(np.arange(2, k+2))/np.log(2)), axis=1)

    # direction='DESCENDING'：同上述例子一个用户@5是[1,0,0,1,1], IDCG计算对象就是最理想化：[1,1,1,0,0]
    idcg = tf.reduce_sum((np.power(2, tf.sort(isIn, direction='DESCENDING'))-1) / (np.log(np.arange(2, k+2))/np.log(2)), axis=1)
    ndcg_score = tf.divide(dcg, idcg)

    # # 这里可能有0/0的情况，nan,将nan变为0就OK
    ndcg_score = tf.reduce_sum(tf.where(tf.is_nan(ndcg_score), tf.zeros_like(ndcg_score), ndcg_score))
    return ndcg_score/length



def recall(rate, negative, length, negative_num, k=5):
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, k).indices
    # 获取当前用户有多少个正例。
    positive_num = (99 - negative_num).flatten()
    k_array = np.array([99 - k] * length).reshape(length, 1)
    positiveIndex = tf.where(tf.greater(negative_num, k_array), negative_num, k_array)
    isIn = tf.cast(tf.greater(topk, positiveIndex), dtype=tf.float32)
    # row为每个用户命中多少个正例
    row = tf.reduce_sum(isIn, axis=1)
    recall = tf.reduce_sum(row / positive_num)
    return recall / length

def env(rate, negative, length):
    hrat1 = hr(rate,negative,length,k=1)
    hrat5 = hr(rate,negative,length,k=5)
    hrat10 = hr(rate,negative,length,k=10)
    hrat20 = hr(rate,negative,length,k=20)
    ndcg5 = ndcg(rate,negative,length,k=5)
    ndcg10 = ndcg(rate,negative,length,k=10)
    ndcg20 = ndcg(rate,negative,length,k=20)
    au = auc(rate,negative,length)
    return hrat1, hrat5, hrat10, hrat20, ndcg5, ndcg10, ndcg20, au
