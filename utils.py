import random
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

batch_size = 32

class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        records = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        user, hist, next_hist, item, time, reward, objective1, objective2 = [], [], [], [], [], [], [], []
        for record in records:
            user.append(record[0])
            hist.append(record[1])
            next_hist.append(record[1][1:]+[record[2]])
            item.append(record[2])
            time.append(record[3])
            reward.append(record[4])
            objective1.append(record[5])
            objective2.append(record[6])
        return self.i, (user, hist, next_hist, item, time, reward, objective1, objective2)

def evaluate(sess, model, test_set):
    rec_objective1, rec_objective2, arr = [], [], []
    userid = list(set([x[0] for x in test_set]))
    for _, uij in DataInput(test_set, batch_size):
        score, objective1_predict, objective2_predict, reward, objective1, objective2, user = model.test(sess, uij)
        for index in range(len(score)):
            arr.append([score[index], objective1_predict[index], objective2_predict[index], reward[index], objective1[index], objective2[index], user[index]])

    for user in userid:
        arr_user = [x for x in arr if x[6]]
        arr_user = sorted(arr_user, key=lambda d:d[0], reverse = True)
        rec_objective1.append(arr_user[0][4])
        rec_objective2.append(arr_user[0][5])

    return np.mean(rec_objective1), np.mean(rec_objective2)

def update_target_graph(primary_network, target_network, tau=0.05):
    # Get the parameters of our Primary Network
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, primary_network)
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_network)
    op_holder = []
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign((1-tau)*from_var+tau*to_var))
    return op_holder