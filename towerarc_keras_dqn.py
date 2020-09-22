import numpy as np
import gym
from tensorflow.python.ops.gen_math_ops import mod
import custom_gym
import tensorflow as tf

# tf.enable_eager_execution() # 开启立即执行模式
cfg = tf.ConfigProto(allow_soft_placement=True)
cfg.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=cfg))

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME_3D = 'OSG_TowerArc-v0'
ENV_NAME_2D = 'TowerArc-v0'

def tensorflow_init():
    # tf.enable_eager_execution() # 开启立即执行模式
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=cfg))


def train_3d():
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME_3D)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    check_point = ModelCheckpoint('tmp/dqn_{}_weights.h5f'.format(ENV_NAME_3D),
                                  monitor='episode_reward',
                                  mode='max',
                                  save_best_only=True,
                                  save_weights_only=True)
     

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=150000, visualize=True, verbose=2)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)

def train_2d():
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME_2D)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    check_point = ModelCheckpoint('tmp/dqn_{}_weights.h5f'.format(ENV_NAME_2D),
                                  monitor='episode_reward',
                                  mode='max',
                                  save_best_only=True,
                                  save_weights_only=True)
     

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2, callbacks=[check_point])

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    tensorflow_init()
    train_3d()