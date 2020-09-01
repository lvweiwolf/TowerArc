import gym # 导入 gym 游戏平台
import custom_gym
import numpy as np

env = gym.make("TowerArc-v0") # 创建平衡杆游戏环境
env.reset()
total_reward = 0

for i in range(1000):
    env.render()
    a1 = np.random.choice(range(3), p=[0.2, 0.4, 0.4])
    a2 = np.random.choice(range(3), p=[0.2, 0.4, 0.4])
    s, r, done, info = env.step([a1, a2])
    total_reward += r
    
    if i % 20 == 0 or done:
        print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
        print("step {} total_reward {:+0.2f}".format(i, total_reward))
    
    if done:
        break    

env.close() # 销毁游戏环境


# env = gym.make("LunarLander-v2") # 创建平衡杆游戏环境
# observation = env.reset() # 复位游戏， 回到初始状态

# for _ in range(1000): # 循环交互 1000 次
#     env.render() # 显示当前时间戳的游戏画面
#     action = env.action_space.sample() # 随机生成一个动作
#     # 与环境交互，返回新的状态，奖励，是否结束标志，其他信息
#     observation, reward, done, info = env.step(action)
#     if done:#游戏回合结束，复位状态
#         observation = env.reset()
        
# env.close() # 销毁游戏环境