import os
import gym # 导入 gym 游戏平台
import numpy as np
import tensorflow as tf
import  matplotlib
from 	matplotlib import pyplot as plt

matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False \

from tensorflow.python.keras.api.keras import layers,models,optimizers

tf.enable_eager_execution()
cfg = tf.ConfigProto(allow_soft_placement=True)
cfg.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=cfg))

LEARNING_RATE = 0.0002
GAMMA = 0.98

class Policy(models.Model):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = layers.Dense(128, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(2, kernel_initializer='he_normal')
          # 网络优化器
        self.optimizer = optimizers.Adam(lr=LEARNING_RATE)
        self.data = []
        
    def call(self, inputs, training=None):
        # 状态输入s的shape为向量：[4]
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.softmax(self.fc2(x), axis=1)
        return x

    def put_data(self, item):
        # 记录r,log_P(a|s)
        self.data.append(item)

    def train_net(self, tape):
        # 计算梯度并更新策略网络参数。tape为梯度记录器
        R = 0 # 终结状态的初始回报为0
        for r, log_prob in self.data[::-1]:#逆序取
            R = r + GAMMA * R # 计算每个时间戳上的回报
            # 每个时间戳都计算一次梯度
            # grad_R=-log_P*R*grad_theta
            loss = -log_prob * R
            with tape.stop_recording():
                # 优化策略网络
                grads = tape.gradient(loss, self.trainable_variables)
                # print(grads)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                
                
        self.data = [] # 清空轨迹


if __name__ == "__main__":
    env = gym.make("CartPole-v1") # 创建平衡杆游戏环境
    # env.seed(2333)
    # tf.random.set_random_seed(2333)
    # np.random.seed(2333)
    
    pi = Policy()       # 创建策略网路
    pi(tf.random.normal((4,4)))
    pi.summary()
    
    score = 0.0 # 计分
    print_interval = 20 # 打印间隔
    returns = []
    
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
        
    for n_epi in range(400):
        s = env.reset() # 回到游戏初始状态, 返回s0
        
        with tf.GradientTape(persistent=True) as tape:
            for t in range(501):
                env.render()
                # 送入当前时间戳状态， 获取策略
                s = tf.constant(s, dtype=tf.float32)
                # s: [4] => [1, 4]
                s = tf.expand_dims(s, axis=0)
                # 动作分布
                prob = pi(s)
                # 从类别分布中采样一个动作, shape: [1]
                a = tf.random.categorical(tf.math.log(prob), 1)[0]
                # a = sess.run(a).item()
                a = int(a) # Tenser转数字
                
                # 与环境交互，返回新的状态，奖励，是否结束标志，其他信息
                s_prime, r, done, info = env.step(a)
                # 记录新动作a与动作a产生的奖励
                pi.put_data((r, tf.math.log(prob[0][a])))
                s = s_prime # 刷新状态
                score += r
                
                # if n_epi > 1000:
                #     env.render()
                if done: # 当前episode终止
                    break
            
            # episode终止后，训练一次网络
            pi.train_net(tape)
        
        if n_epi%print_interval==0 and n_epi!=0:
            returns.append(score/print_interval)
            print(f"# of episode :{n_epi}, avg score : {score/print_interval}")
            score = 0.0
    
    env.close() # 关闭环境
    
    plt.plot(np.arange(len(returns))*print_interval, returns)
    plt.plot(np.arange(len(returns))*print_interval, returns, 's')
    plt.xlabel('回合数')
    plt.ylabel('总回报')
    plt.savefig('reinforce-tf-cartpole.jpg')