import pickle
import gym
import custom_gym
import numpy as np
import tensorflow as tf

from gym import spaces
from tensorflow.python.ops.gen_math_ops import mod

# tf.enable_eager_execution() # 开启立即执行模式
cfg = tf.ConfigProto(allow_soft_placement=True)
cfg.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=cfg))

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, callbacks

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


ENV_NAME_3D = 'OSG_TowerArc-v0'
ENV_NAME_2D = 'TowerArc-v0'
ENV_NAME_CARTPOLE = 'CartPole-v1'

def tensorflow_init():
    # tf.enable_eager_execution() # 开启立即执行模式
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=cfg))

def simple_model(env):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(256))
    model.add(Activation('relu')),
    model.add(Dense(256))
    model.add(Activation('relu')),
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    print(model.summary())
    
    return model

def medium_model(env):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    print(model.summary())
    
    return model

def high_model(env):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    print(model.summary())
    
    return model

def train(env_name):
    #env_name = 'MountainCar-v0'
    
    env = gym.make(env_name)
    # np.random.seed(1024)
    # env.seed(1024)
    model = simple_model(env)
    
    # 计算Q值策略
    policy = EpsGreedyQPolicy(eps=0.1)
    # policy = BoltzmannQPolicy()
    
    # relayBuffer 和 policy 记录
    try:
        memory = pickle.load(open('tmp/dqn_{}_weights.mdl'.format(env_name), "rb"))
    except (FileNotFoundError, EOFError):
        memory = SequentialMemory(limit=50000, window_length=1)
    
    # dqn agent模型权重
    dqn = DQNAgent(model=model,
                   enable_double_dqn=True,
                   nb_actions=env.action_space.n,
                   memory=memory,
                   nb_steps_warmup=10,
                   target_model_update=1e-2,
                   policy=policy)
    
    dqn.compile(Adam(lr=1e-4), metrics=['mae'])

    try:
        dqn.load_weights('tmp/dqn_{}_weights.h5f'.format(env_name))
    except (OSError):
        print("no weights file, train from the beginning.")
    
    dqn.fit(env, nb_steps=100000, visualize=True, verbose=2)
    dqn.test(env, nb_episodes=5, visualize=True)
    
    # 保存权重
    dqn.save_weights('tmp/dqn_{}_weights.h5f'.format(env_name), overwrite=True)
    # Save memory
    pickle.dump(memory, open('tmp/dqn_{}_weights.mdl'.format(env_name), "wb"))
    
def eval(env_name):
    #env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    # np.random.seed(1024)
    # env.seed(1024)
    model = simple_model(env)
    
    # 计算Q值策略
    policy = EpsGreedyQPolicy(eps=0.1)
    # policy = BoltzmannQPolicy()
    
    # relayBuffer 和 policy 记录
    # try:
    #     memory = pickle.load(open('tmp/dqn_{}_weights.mdl'.format(env_name), "rb"))
    # except (FileNotFoundError, EOFError):
    memory = SequentialMemory(limit=50000, window_length=1)
    
    # dqn agent模型权重
    dqn = DQNAgent(model=model,
                   enable_double_dqn=True,
                   nb_actions=env.action_space.n,
                   memory=memory,
                   nb_steps_warmup=10,
                   target_model_update=1e-2,
                   policy=policy)
    
    dqn.compile(Adam(lr=1e-4), metrics=['mae'])

    try:
        dqn.load_weights('tmp/dqn_{}_weights.h5f'.format(env_name))
    except (OSError):
        print("no weights file, train from the beginning.")
 
    dqn.test(env, nb_episodes=5, visualize=True)
    

if __name__ == '__main__':
    env_name = ENV_NAME_3D
    tensorflow_init()
    # train(env_name)
    eval(env_name) # 训练结果测试