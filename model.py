import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

class Reinforce_Model(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, primary_network, target_network):
        hidden_size = 128
        slice_window = 10
        gamma = 0.95
        
        self.u = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.hist = tf.placeholder(tf.int32, [batch_size, slice_window]) # [B, T]
        self.next_hist = tf.placeholder(tf.int32, [batch_size, slice_window]) # [B, T]
        self.time = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.reward = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.objective1 = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.objective2 = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.lr = tf.placeholder(tf.float64, [])

        self.primary_network = primary_network
        self.target_network = target_network

        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_size // 2])
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_size // 2])
        item_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.i),tf.nn.embedding_lookup(user_emb_w, self.u),], axis=1)
        h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.hist),tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1), [1, slice_window, 1]),], axis=2)
        next_h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.next_hist),tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1), [1, slice_window, 1])], axis=2)

        # Current Consumer State & Next State
        with tf.variable_scope('time_lstm', reuse=tf.AUTO_REUSE):
        	output, _ = tf.nn.dynamic_rnn(LSTMCell(hidden_size), inputs=h_emb, dtype=tf.float32)
        	state, alphas = self.seq_attention(output, hidden_size, slice_window)
        	state = tf.nn.dropout(state, 0.1)

        	next_output, _ = tf.nn.dynamic_rnn(LSTMCell(hidden_size), inputs=next_h_emb, dtype=tf.float32)
        	next_state, next_alphas = self.seq_attention(next_output, hidden_size, slice_window)
        	next_state = tf.nn.dropout(next_state, 0.1)

        # Objective 1 Estimation
        with tf.variable_scope('objective1', reuse=tf.AUTO_REUSE):
            objective1 = tf.concat([state, item_emb], axis=1)
            objective1 = tf.layers.batch_normalization(inputs=objective1)
            objective1 = tf.layers.dense(objective1, 80, activation=tf.nn.sigmoid)
            objective1 = tf.layers.dense(objective1, 40, activation=tf.nn.sigmoid)
            objective1 = tf.layers.dense(objective1, 1, activation=None)
            self.objective1_predict = tf.reshape(objective1, [-1])
            self.objective1_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.objective1, predictions=self.objective1_predict))

        # Objective 2 Estimation
        with tf.variable_scope('objective2', reuse=tf.AUTO_REUSE):
            objective2 = tf.concat([state, item_emb], axis=1)
            objective2 = tf.layers.batch_normalization(inputs=objective2)
            objective2 = tf.layers.dense(objective2, 80, activation=tf.nn.sigmoid)
            objective2 = tf.layers.dense(objective2, 40, activation=tf.nn.sigmoid)
            objective2 = tf.layers.dense(objective2, 1, activation=None)
            self.objective2_predict = tf.reshape(objective2, [-1])
            self.objective2_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.objective2, predictions=self.objective2_predict))
            
        # Weight of Objective 1
        with tf.variable_scope('objective1_weight', reuse=tf.AUTO_REUSE):
            objective1_weight = tf.layers.dense(state, 1, activation=tf.nn.tanh, name='factor')
            next_objective1_weight = tf.layers.dense(next_state, 1, activation=tf.nn.tanh, name='factor')

        # Weight of Objective 2
        with tf.variable_scope('objective2_weight', reuse=tf.AUTO_REUSE):
            objective2_weight = tf.layers.dense(state, 1, activation=tf.nn.tanh, name='factor')
            next_objective2_weight = tf.layers.dense(next_state, 1, activation=tf.nn.tanh, name='factor')

        # DDPG Network
        with tf.variable_scope('ddpg', reuse=tf.AUTO_REUSE):
            q_value = self.primary_network.forward(state, objective1_weight, objective2_weight)
            next_q_value = self.target_network.forward(next_state, next_objective1_weight, next_objective2_weight)
            predict_q_value = self.reward + gamma*next_q_value
            self.ddpg_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=predict_q_value, predictions=q_value))
            self.objective1_weight = tf.reshape(objective1_weight, [-1])
            self.objective2_weight = tf.reshape(objective2_weight, [-1])

        # Aggregated Utility Function
        self.utility = self.objective1_weight * self.objective1_predict + self.objective2_weight * self.objective2_predict # [B]exp
        self.score = tf.sigmoid(self.utility)
        self.loss = self.ddpg_loss + self.objective1_loss + self.objective2_loss

        # Back-Propogation and Parameter Update
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.hist: uij[1],
                self.next_hist: uij[2],
                self.i: uij[3],
                self.time: uij[4],
                self.reward: uij[5],
                self.objective1: uij[6],
                self.objective2: uij[7],
                self.lr: lr
                })
        return loss

    def test(self, sess, uij):
        score, objective1_predict, objective2_predict = sess.run([self.score, self.objective1_predict, self.objective2_predict], feed_dict={
                self.u: uij[0],
                self.hist: uij[1],
                self.next_hist:uij[2],
                self.i: uij[3],
                self.time: uij[4],
                self.reward: uij[5],
                self.objective1: uij[6],
                self.objective2: uij[7],
                })
        return score, objective1_predict, objective2_predict, uij[4], uij[5], uij[6], uij[0]

    def seq_attention(self, inputs, hidden_size, attention_size):
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size]), 1, name="attention_embedding")
        return output, alphas

class DDPG(object):
    def __init__(self, hidden_size, scope_name):
        self.hidden_size = hidden_size
        self.scope_name = scope_name

    def forward(self, state, objective1_weight, objective2_weight):
        with tf.variable_scope(self.scope_name):
            feature = tf.layers.dense(state, self.hidden_size, activation=tf.nn.relu)
            feature = tf.layers.dense(feature, 1, activation=tf.nn.sigmoid)
            concat = tf.concat([objective1_weight, objective2_weight, feature], axis=1)
            q_value = tf.layers.dense(concat, 1, activation=tf.nn.tanh)
            q_value = tf.reshape(q_value, [-1])
            return q_value