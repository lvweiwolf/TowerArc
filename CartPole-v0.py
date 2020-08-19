import os
import gym
import numpy as np
import tensorflow as tf

import matplotlib
from matplotlib import pyplot as plt
from agent.reinforce import ReinforceAgent


matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False \

cfg = tf.ConfigProto(allow_soft_placement=True)
cfg.gpu_options.allow_growth = True


NUM_EPISODES = 400
MAX_STEPS = 501
LEARNING_RATE = 0.0002 
DISCOUNT_FACTOR = 0.98
TRAIN_EVERY_NUM_EPISODES = 1
PRINT_INTERVAL = 20
EPOCH_SIZE = 1

RECORD = False
    
def train(agent, env, sess, num_episode = NUM_EPISODES):
    score = 0.0 # 计分
    returns = []
    
    for i in range(NUM_EPISODES):
        cur_state = env.reset()
        episode = []
        
        for t in range(MAX_STEPS):
            action = agent.get_action(cur_state, sess) # net.forward(cur_state) => random action
            next_state, reward, done, info = env.step(action)
            episode.append([cur_state, action, next_state, reward, done])
            cur_state = next_state
            score += reward
            env.render()
            
            
            if done: # 终结
                break
        
        if i % TRAIN_EVERY_NUM_EPISODES == 0:
            agent.train(episode, sess, EPOCH_SIZE) 
            
        if i % PRINT_INTERVAL==0 and i != 0:
            returns.append(score / PRINT_INTERVAL)
            print(f"# of episode :{i}, avg score : {score / PRINT_INTERVAL}")
            score = 0.0
    
    return returns

if __name__ == "__main__":
    env = gym.make("CartPole-v0") # 创建平衡杆游戏环境
    if RECORD:
        env = gym.wrappers.Monitor(env, './tmp/cartpole-experiment-2', force=True)
  
    agent = ReinforceAgent(lr = LEARNING_RATE,
                           gamma = DISCOUNT_FACTOR,
                           state_size = 4,
                           action_size = 2)
    
    
    
    with tf.Session(config=cfg) as sess:
        sess.run(tf.global_variables_initializer())
        returns = train(agent, env, sess)
    
    if RECORD:
        env.monitor.close()
  
    env.close() # 关闭环境
     
    plt.plot(np.arange(len(returns))*PRINT_INTERVAL, returns)
    plt.plot(np.arange(len(returns))*PRINT_INTERVAL, returns, 's')
    plt.xlabel('回合数')
    plt.ylabel('总回报')
    plt.savefig('reinforce-tf-cartpole.jpg')
    