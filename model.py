import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

class Reinforce_Model(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, primary_dqn, target_dqn):
        hidden_size = 128
        memory_window = 10
        gamma = 0.99
        
        self.u = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.y = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.hist = tf.placeholder(tf.int32, [batch_size, memory_window]) # [B, T]
        self.lr = tf.placeholder(tf.float64, [])
        self.next_hist = tf.placeholder(tf.int32, [batch_size, memory_window]) # [B, T]

        self.primary_dqn = primary_dqn
        self.target_dqn = target_dqn

        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_size // 2])
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_size // 2])
        user_b = tf.get_variable("user_b", [user_count], initializer=tf.constant_initializer(0.0),)
        item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))

        item_b = tf.gather(item_b, self.i)
        user_b = tf.gather(user_b, self.u)
        item_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.i),tf.nn.embedding_lookup(user_emb_w, self.u),], axis=1)
        h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.hist),tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1), [1, memory_window, 1]),], axis=2)
        next_h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.next_hist),tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1), [1, memory_window, 1])], axis=2)

        # Current State & Next State
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
        	output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float32)
        	state, alphas = self.seq_attention(output, hidden_size, memory_window)
        	#state = tf.nn.dropout(state, 0.1)

        	next_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=next_h_emb, dtype=tf.float32)
        	next_state, next_alphas = self.seq_attention(next_output, hidden_size, memory_window)
        	#next_state = tf.nn.dropout(next_state, 0.1)

        # Click-Through Rate Estimation
        concat = tf.concat([state, item_emb], axis=1)
        concat = tf.layers.batch_normalization(inputs=concat)
        concat = tf.layers.dense(concat, 80, activation=tf.nn.sigmoid, name='f1')
        concat = tf.layers.dense(concat, 40, activation=tf.nn.sigmoid, name='f2')
        concat = tf.layers.dense(concat, 1, activation=None, name='f3')
        concat = tf.reshape(concat, [-1])

        #Unexpectedness (with clustering of user interests)
        unexp = tf.reduce_mean(h_emb, axis=1)
        unexp = tf.norm(unexp-item_emb ,ord='euclidean', axis=1)
        self.unexp = unexp
        unexp = tf.exp(-1.0*unexp) * unexp #Unexpected Activation Function
        unexp = tf.stop_gradient(unexp)

        #Dueling Double DQN for Unexpected Factor
        with tf.variable_scope('unexp_factor', reuse=tf.AUTO_REUSE):
            unexp_factor = tf.layers.dense(state, 1, activation=tf.nn.tanh, name='factor')
            next_unexp_factor = tf.layers.dense(next_state, 1, activation=tf.nn.tanh, name='factor')

        self.q_value = self.primary_dqn.forward(state, unexp_factor)
        self.next_q_value = self.target_dqn.forward(next_state, next_unexp_factor)
        self.predict_q_value = self.y + gamma*self.next_q_value
        unexp_factor = tf.reshape(unexp_factor, [-1])

        #Estmation of user preference by combing different components
        self.logits = item_b + concat + user_b + unexp_factor*unexp # [B]exp
        self.score = tf.sigmoid(self.logits)

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.y)+tf.losses.mean_squared_error(labels=self.predict_q_value, predictions=self.q_value))
        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.y))
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
                self.y: uij[4],
                self.lr: lr
                })
        return loss

    def test(self, sess, uij):
        q_value = sess.run([self.q_value], feed_dict={
                self.u: uij[0],
                self.hist: uij[1],
                self.next_hist:uij[2],
                self.i: uij[3],
                self.y: uij[4]
                })
        return q_value, uij[4], uij[0], uij[3], []

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

class Primary_DQN(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def forward(self, state, unexp_factor):
        feature = tf.layers.dense(state, self.hidden_size, activation=tf.nn.relu)
        feature = tf.layers.dense(feature, 1, activation=tf.nn.relu)
        concat = tf.concat([unexp_factor, feature], axis=1)
        q_value = tf.layers.dense(concat, 1, activation=tf.nn.tanh)
        q_value = tf.reshape(q_value, [-1])
        return q_value

class Target_DQN(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def forward(self, state, unexp_factor):
        feature = tf.layers.dense(state, self.hidden_size, activation=tf.nn.relu)
        feature = tf.layers.dense(feature, 1, activation=tf.nn.relu)
        concat = tf.concat([unexp_factor, feature], axis=1)
        q_value = tf.layers.dense(concat, 1, activation=tf.nn.tanh)
        q_value = tf.reshape(q_value, [-1])
        return q_value