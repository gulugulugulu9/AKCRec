from utils import *
import tensorflow as tf
from models import *
import time
import numpy as np



# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Set
learning_rate = 0.001
decay_rate = 1
global_steps = 10000
decay_steps = 100


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('output_dim', 64, 'Output_dim of user final embedding.')
flags.DEFINE_integer('latent_dim', 30,'Latent_dim of user&item.')


# Load data

support_string_user = ['ucu', 'uvu', 'uctcu', 'uku'] #
support_string_item = ['kuk']

rating, features_item, features_user, support_user, support_item, negative = load_data(user=support_string_user,
                                                                                       item=support_string_item)

user_dim = rating.shape[0]
item_dim = rating.shape[1]

# user_support
support_num_user = len(support_string_user)
# item_support
support_num_item = len(support_string_item)
# Define placeholders
placeholders = {
    'rating': tf.placeholder(dtype=tf.float32, shape=rating.shape, name="rating"),
    'features_user': tf.placeholder(dtype=tf.float32, shape=features_user.shape, name='features_user'),
    'features_item': tf.placeholder(dtype=tf.float32, shape=features_item.shape, name="features_item"),
    'support_user': [tf.placeholder(dtype=tf.float32, name='support'+str(_)) for _ in range(support_num_user)],
    'support_item': [tf.placeholder(dtype=tf.float32, name='support'+str(_)) for _ in range(support_num_item)],
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
    'negative': tf.placeholder(dtype=tf.int32, shape=negative.shape, name='negative')
}
global_ = tf.Variable(tf.constant(0))
learning = tf.train.exponential_decay(learning_rate,global_,decay_steps,decay_rate,staircase=False)
# Create Model
model = AGCNrec(placeholders, input_dim_user=features_user.shape[1], input_dim_item=features_item.shape[1],
                user_dim=user_dim, item_dim=item_dim, learning_rate=learning)

# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

# Train model
epoch = 0

maxList = [0]*14
maxEpoch = [0]*14


while epoch < global_steps:
    # Construct feed dictionary
    feed_dict = construct_feed_dict(placeholders, features_user, features_item, rating, support_user,
                                    support_item, negative)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({global_:epoch})

    _, los, HR1, HR5, HR10, NDCG5, NDCG10, AUC, user, item, result, auc1, auc2, ndcgNew5, ndcgNew10, \
                 hratNew1, hratNew5, hratNew10, recall5, recall10 = sess.run([model.train_op, model.los, model.hrat1,
                                                                              model.hrat5, model.hrat10, model.ndcg5, model.ndcg10,
                                                                             model.auc, model.user, model.item,
                                                                            model.result, model.auc1,model.auc2, model.ndcgNew5,
                                                                            model.ndcgNew10, model.hratNew1,model.hratNew5,model.hratNew10, model.recall5, model.recall10], feed_dict)

    if epoch%50 == 0:
        print("当前epoch的指标： ")
        aLine = 'Train'+str(epoch)+" Loss:"+str(los)+" AUC:"+str(AUC)+" HR@1:"+str(HR1)+" HR@5:"+str(HR5)+" HR@10:"+str(HR10)+\
                " nDCG@5:"+str(NDCG5)+" nDCG@10:"+str(NDCG10)
        aLine_new = 'Train'+str(epoch)+" Loss:"+str(los)+" AUC1:"+str(auc1)+" hr@1New:"+str(hratNew1)+" hr@5New:"+str(hratNew5)+\
                    " hr@10New:"+str(hratNew10)+" ndcgNew5:" + str(ndcgNew5) + " ndcgNew10:" + str(ndcgNew10) + " recall5:" + str(recall5) + " recall10：" + str(recall10)
        print(aLine)
        print(aLine_new)
        index = 0
        # 修改
        nowList = [AUC, HR1, HR5, HR10, NDCG5, NDCG10, auc1, hratNew1, hratNew5,hratNew10, ndcgNew5, ndcgNew10, recall5, recall10]
        for real in nowList:
            if real > maxList[index]:
                maxList[index] = real
                maxEpoch[index] = epoch
            index += 1
        ## [AUC, HR1, HR5, HR10, HR20, NDCG5, NDCG10, NDCG20, MRR,  auc1,  ndcgNew5, ndcgNew10, ndcgNew20, mrrNew, hratNew1, hratNew5,hratNew10, hratNew20]

        str1 = " AUC:"+str(maxList[0])+" HR@1:"+str(maxList[1])+" HR@5:"+str(maxList[2])+" HR@10:"+str(maxList[3])+" nDCG@5:"+str(maxList[4])+" nDCG@10:"+str(maxList[5])

        str2 = " AUC1:"+str(maxList[6])+" hr@1New:"+str(maxList[7])+" hr@5New:"+str(maxList[8])+" hr@10New:"+str(maxList[9])+" ndcgNew5:"+str(maxList[10])+" ndcgNew10:" + str(maxList[11])+" recall5:"+str(maxList[12])+" recall10:"+str(maxList[13])
        print("原始数据指标最大值如下：")
        print(str1)
        print("修改数据指标最大值如下：")
        print(str2)
        # [AUC, HR1, HR5, HR10, NDCG5, NDCG10,  auc1,  hratNew1, hratNew5,hratNew10, ndcgNew5, ndcgNew10, ndcgNew20, recall5,recall10]
        print("maxEpoch: ", maxEpoch)
        print("-------------------------------------------------")
    epoch += 1
