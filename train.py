import os, sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Reinforce_Model, DDPG
from utils import DataInput, evaluate, update_target_graph

#Note: this code must be run using tensorflow 1.4.0
tf.reset_default_graph()

random.seed(625)
np.random.seed(625)
tf.set_random_seed(625)
batch_size = 32
hidden_size = 128

data = pd.read_csv('sample_data.txt', low_memory=False)
userid = list(set(data['userId']))
itemid = list(set(data['itemId']))
user_count = len(userid)
item_count = len(itemid)

validate = 4 * len(data) // 5
train_data = data.loc[:validate,]
test_data = data.loc[validate:,]
train_set, test_set = [], []

for user in userid:
    train_user = train_data.loc[train_data['userId']==user]
    train_user = train_user.sort_values(['timestamp'])
    length = len(train_user)
    train_user.index = range(length)
    if length > 11:
        for i in range(length-11):
            train_set.append((user, list(train_user.loc[i:i+9,'itemId']), train_user.loc[i+10,'itemId'], train_user.loc[i+9,'timestamp']-train_user.loc[i+8,'timestamp'], float(train_user.loc[i+10,'reward']),float(train_user.loc[i+10,'objective1']),float(train_user.loc[i+10,'objective2'])))
    test_user = test_data.loc[test_data['userId']==user]
    test_user = test_user.sort_values(['timestamp'])
    length = len(test_user)
    test_user.index = range(length)
    if length > 11:
        for i in range(length-11):
            test_set.append((user, list(test_user.loc[i:i+9,'itemId']), test_user.loc[i+10,'itemId'], test_user.loc[i+9,'timestamp']-test_user.loc[i+8,'timestamp'], float(test_user.loc[i+10,'reward']),float(test_user.loc[i+10,'objective1']),float(test_user.loc[i+10,'objective2'])))
train_set = train_set[:len(train_set)//batch_size*batch_size]
test_set = test_set[:len(test_set)//batch_size*batch_size]

start_time = time.time()

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    primary_network = DDPG(hidden_size, 'primary_network')
    target_network = DDPG(hidden_size, 'target_network')
    model = Reinforce_Model(user_count, item_count, hidden_size, batch_size, primary_network, target_network)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    print('Objective1_Value: %.4f\t Objective2_Value: %.4f\t' % evaluate(sess, model, train_set))
    sys.stdout.flush()
    lr = 1
    start_time = time.time()
    last_auc = 0.0
    
    for epoch in range(100):
        random.shuffle(train_set)
        random.shuffle(test_set)
        epoch_size = round(len(train_set) / batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, batch_size):
            loss = model.train(sess, uij, lr)
            loss_sum += loss
        print('Epoch %d Train_Loss: %.4f' % (model.global_epoch_step.eval(), loss_sum))      
        print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time()-start_time))
        print('Objective1_Value: %.4f\t Objective2_Value: %.4f\t' % evaluate(sess, model, train_set))
        #print('Objective1_Value: %.4f\t Objective2_Value: %.4f\t' % evaluate(sess, model, test_set))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
        if epoch % 5 == 0:
            update_target_graph('primary_dqn','target_dqn')

end_time = time.time()