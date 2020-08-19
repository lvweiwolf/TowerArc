import numpy as np 
import tensorflow as tf


class ReinforceAgent(object):
    
    def __init__(self,
                 lr = 0.01,
                 gamma = 0.99,
                 state_size = 4,
                 action_size = 2,
                 scope = 'PolicyGradient'):
        super(ReinforceAgent, self).__init__()
        
        self.lr = lr
        self.gamma = gamma
        self.state_size = state_size        # 状态的尺寸, 网络的输入尺寸(车位置、车速、杆位置、杆速)
        self.action_size = action_size      # 动作的只存，网络输出尺寸(向左，向右)
        self.scope = scope
        
        self.n_hidden_1 = 128
        self.n_hidden_2 = 10
        
        self._build_policy_net()
        
        
    def _build_policy_net(self):
        ''' 构建策略神经网络
        '''
        with tf.variable_scope(self.scope): 
            # 输入张量
            self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
            # 输出张量
            self.action = tf.placeholder(tf.int32, [None])
            # 时间序列 t:T 的总回报
            self.target = tf.placeholder(tf.float32, [None])
            
            # 隐藏层
            hidden_layer = tf.layers.Dense(self.n_hidden_1, kernel_initializer='he_normal', activation=tf.nn.relu)(self.state_input)
            # hidden_layer = tf.layers.Dense(self.n_hidden_2, kernel_initializer='he_normal', activation=tf.nn.relu)(hidden_layer)
            
            # 输出层
            self.action_probs = tf.layers.Dense(self.action_size, kernel_initializer='he_normal', activation=tf.nn.softmax)(hidden_layer)
            self.value = tf.layers.Dense(1, kernel_initializer='he_normal')(hidden_layer)
            
            # self.action 由agent 随机取值
            action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
            self.action_value_preds = tf.reduce_sum(self.action_probs * action_mask, 1)

            self.value_loss = tf.reduce_mean(tf.square(self.target - self.value))
            self.policy_loss = tf.reduce_mean(-tf.log(self.action_value_preds) * (self.target - self.value))
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])     
            
            self.loss = self.policy_loss + 5 * self.value_loss + 0.002 * self.l2_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    
    def get_policy(self, state, sess):
        policy = sess.run(self.action_probs, feed_dict={self.state_input : [state]}) # [None, 0, 1]
        return policy[0] # [0, 1]

    def get_action(self, state, sess):
        pi = self.get_policy(state, sess)
        return np.random.choice(range(self.action_size), p=pi) # int32
    
    def train(self, episode, sess, train_epoch = 1):
        for t in range(len(episode)):
            # self.total_steps = self.total_steps + 1
            # 获取事件序列范围 t:T 的总回报
            target = sum([self.gamma**i * r for i, (s, a, s1, r, d) in enumerate(episode[t:])])
            state, action, state_next, reward, done = episode[t]
            feed_dict = {self.state_input : [state], self.target : [target], self.action : [action]}
            _, loss, v, pg_loss, v_a = sess.run([self.train_op, self.loss, self.value, self.policy_loss, self.action_value_preds], feed_dict)
        