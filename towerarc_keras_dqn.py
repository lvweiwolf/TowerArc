import os
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
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME_3D = 'OSG_TowerArc-v0'
ENV_NAME_2D = 'TowerArc-v0'
ENV_NAME_CARTPOLE = 'CartPole-v1'

def tensorflow_init():
    # tf.enable_eager_execution() # 开启立即执行模式
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=cfg))


def build_model(env):
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

def build_model2(env):
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

if __name__ == '__main__':
    tensorflow_init()
    
    #env_name = 'MountainCar-v0'
    env_name = ENV_NAME_3D
    env = gym.make(env_name)
    # np.random.seed(1024)
    # env.seed(1024)
    model = build_model(env)
    
    pretrained_file = 'tmp/dqn_{}_weights.h5f'.format(env_name)
    if os.path.exists(pretrained_file):
        model.load_weights(pretrained_file)
    
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   memory=SequentialMemory(limit=50000, window_length=1),
                   nb_steps_warmup=10,
                   target_model_update=1e-2,
                   policy=None)
    
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    check_point = ModelCheckpoint(pretrained_file,
                                  monitor='episode_reward',
                                  mode='max',
                                  save_best_only=True,
                                  save_weights_only=True)
    
    # dqn.fit(env, nb_steps=500000, visualize=False, verbose=2, callbacks=[check_point])
    dqn.fit(env, nb_steps=500000, visualize=False, verbose=2, callbacks=[check_point])
    dqn.test(env, nb_episodes=5, visualize=True)