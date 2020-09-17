import  matplotlib
from 	matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure()

import os
import gym
import custom_gym
import numpy as np
import tensorflow as tf

from collections import namedtuple

from tensorflow.python.keras.api import keras
from tensorflow.python.keras.api.keras import layers, losses, optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GAMMA = 0.98 # 激励衰减因子
EPSILON = 0.2 # PPO 误差超参数 0.8 ~ 1.2
BATCH_SIZE = 32 # batch size
NUM_EPOCH = 4000
NUM_EPISODES = 500

class Actor(keras.Model):
    ''' 策略网络
    '''
    def __init__(self, action_size):
        super(Actor, self).__init__()
        self.action_size = action_size
        # 策略网络，也叫Actor网络，输出为概率分布pi(a|s)
        self.fc1 = layers.Dense(100, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(action_size, kernel_initializer='he_normal')

    def call(self, inputs):
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.softmax(self.fc2(x), axis=1) # 转换为概率
        return x
    

class Critic(keras.Model):
    ''' 值估计网络, 用于计算采样策略与目标策略之间的距离D(KL)
    '''
    def __init__(self):
        super(Critic, self).__init__()
        # 偏置b的估值网络，也叫Critic网络，输出为v(s)
        self.fc1 = layers.Dense(100, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(1, kernel_initializer='he_normal')
        
    def call(self, inputs):
        x = tf.nn.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x
    

class PPO(object):
    # PPO 算法
    def __init__(self, action_size):
        super(PPO, self).__init__()
        self.actor = Actor(action_size)            # 创建Actor网络
        self.critic = Critic()                       # 创建Critic网络
        self.buffer = []                             # 数据缓冲区
        self.actor_optimizer = optimizers.Adam(1e-3) # Actor优化器
        self.critic_optimizer = optimizers.Adam(3e-3) # Critic优化器
        
    def get_action(self, s):
        # 送入状态向量，获取动作策略 shape: [4]
        s = tf.constant(s, dtype=tf.float32)
        # s: [4] => [1, 4]
        s = tf.expand_dims(s, axis=0)
        # 获取策略分布
        prob = self.actor(s)
        # 从策略分布中采样一个动作，shape: [1]
        a = tf.random.categorical(tf.math.log(prob), 1)[0]
        a = int(a) # warning: 需要eager_execution
        
        return a, float(prob[0][a]) # 返回动作及它的概率

    def get_value(self, s):
        # 总入值网络状态数据 shape: [4]
        s = tf.constant(s, dtype=tf.float32)
        # s: [4] => [1, 4]
        s = tf.expand_dims(s, axis=0)
        v = self.critic(s)[0]
        
        return float(v)
        
    def store_transition(self, transition):
        # 存储采样器数据
        self.buffer.append(transition)
        
    def optimize(self):
        # 优化网络主函数
        # 从缓存中取出样本数据， 转换成Tensor
        state = tf.constant([t.state for t in self.buffer], dtype=tf.float32)
        action = tf.constant([t.action for t in self.buffer], dtype=tf.int32)
        action = tf.reshape(action, [-1, 1])
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tf.constant([t.a_log_prob for t in self.buffer], dtype=tf.float32)
        old_action_log_prob = tf.reshape(old_action_log_prob, [-1, 1])
        
        # 计算总回报
        R = 0
        Rs = []
        for r in reward[::-1]:
            R = r + GAMMA * R
            Rs.insert(0, R)
        
        Rs = tf.constant(Rs, dtype=tf.float32)
        
        for _ in range(round(10 * len(self.buffer) / BATCH_SIZE)):
            # 随机从历史数据中采样batch size大小的样本
            index = np.random.choice(np.arange(len(self.buffer)), BATCH_SIZE, replace=False)    
            batch_state = tf.gather(state, index, axis=0)
            batch_action = tf.gather(action, index, axis=0) # [batch, 1]
            
            # 自动梯度计算环境
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                # 取出R(st), [batch_szie, 1]
                v_target = tf.expand_dims(tf.gather(Rs, index, axis=0), axis=1)
                # 计算v(s)预测值,
                v = self.critic(batch_state)
                delta = v_target - v # 计算优势值
                # 断开梯度连接, delta op之前的梯度不能BP
                advantage = tf.stop_gradient(delta) 
                # 预测batch个历史状态下的动作分布pi(a|st), shape:[batch, action_size]
                pi = self.actor(batch_state)
                # 构造索引张量，方便索引到动作概率值
                # [batch, ] => [bactch, 1]
                indices = tf.expand_dims(tf.range(batch_action.shape[0]), axis=1)
                # [batch, 1] => [batch, 2] 例如:[[0, 1], [2, 0]], shape[0]属于0~batch
                # shape[1] 属于 (0, 1) 0表示向左的动作, 1表示向右的动作
                indices = tf.concat([indices, batch_action], axis=1)
                # 获得batch个响应动作的概率, [batch,]
                pi_a = tf.gather_nd(pi, indices)
                # [batch,] => [batch, 1]
                pi_a = tf.expand_dims(pi_a, axis=1)
                # 重要性采样(IS), 当前从历史状态预测的动作与历史预测的动作的占比
                ratio = (pi_a / tf.gather(old_action_log_prob, index, axis=0))
                # 还需要考虑优势值
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON) * advantage
                # PPO 误差函数
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                # 对于偏置v来说，需要与MC估计的R(st)越接近越好
                value_loss = losses.MSE(v_target, v)

                # 1.优化策略网络
                grads = tape1.gradient(policy_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
                
                # 2.优化值网络
                grads = tape2.gradient(value_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        
        self.buffer = [] # 清空已训练数据
                
                

def tensorflow_init():
    tf.enable_eager_execution() # 开启立即执行模式
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=cfg))


def main():
    ENV = gym.make('OSG_TowerArc-v0') # 创建交互环境
    ENV.seed(2222)
    tf.random.set_random_seed(2222)
    np.random.seed(2222)
    
    # 结构体定义
    Transition = namedtuple('Transition', ['state',
                                           'action',
                                           'a_log_prob',
                                           'reward',
                                           'next_state'])
    
    # 定义智能体
    agent = PPO(action_size=16)
    returns = []    # 统计总回报
    total = 0       # 一段时间内的平均回报
    
    for i_epoch in range(NUM_EPOCH):
        state = ENV.reset()
        
        for t in range(NUM_EPISODES):
            # 通过最新策略与环境交互
            action, action_prob = agent.get_action(state)
            next_state, reward, done, _ = ENV.step(action)
            # 构建样本并存储
            trans = Transition(state, action, action_prob, reward, next_state)
            agent.store_transition(trans)
            state = next_state # 刷新状态
            total += reward # 累计回报
            
            ENV.render()
             
            if done:
                if len(agent.buffer) >= BATCH_SIZE:
                    agent.optimize() # 训练网络
                
                break
            
            
        if i_epoch % 20 == 0: # 每20回合统计一次平均回报
            returns.append(total / 20)
            total = 0
            # print(i_epoch, returns[-1])
            print(f"# of episode :{i_epoch}, avg reward : {returns[-1]}")

    # 正确的关闭环境
    ENV.close()
    
    print(np.array(returns))
    plt.figure()
    plt.plot(np.arange(len(returns))*20, np.array(returns))
    plt.plot(np.arange(len(returns))*20, np.array(returns), 's')
    plt.xlabel('回合数')
    plt.ylabel('总回报')
    plt.savefig('ppo-tf-towerarc.jpg')

if __name__ == '__main__':
    tensorflow_init()
    main()
    print('end')