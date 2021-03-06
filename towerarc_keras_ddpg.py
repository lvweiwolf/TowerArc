import gym
import custom_gym
import pickle
import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess



def tensorflow_init():
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=cfg))

def actor_model(env):
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]
    
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())
    
    return actor

def critic_model(env):
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]
    
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    return action_input, critic

def train(env_name):
    # Get the environment and extract the number of actions.
    env = gym.make(env_name)
    np.random.seed(123)
    env.seed(123)

    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]
    # Next, we build a very simple model.
    
    actor = actor_model(env)
    action_input, critic = critic_model(env)

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    try:
        memory = pickle.load(open('tmp/ddpg_{}_weights.mdl'.format(env_name), "rb"))
    except (FileNotFoundError, EOFError):
        memory = SequentialMemory(limit=100000, window_length=1)
    
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, 
                                              theta=.15, 
                                              mu=0., 
                                              sigma=.3)
    
    agent = DDPGAgent(nb_actions=nb_actions, 
                      actor=actor, 
                      critic=critic, 
                      critic_action_input=action_input,
                      memory=memory, 
                      nb_steps_warmup_critic=100, 
                      nb_steps_warmup_actor=100,
                      random_process=random_process, 
                      gamma=.99, 
                      target_model_update=1e-2)
    # optimizer
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    
    # load weights
    try:
        agent.load_weights('tmp/ddpg_{}_weights.h5f'.format(env_name))
    except (OSError):
        print("no weights file, train from the beginning.")

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    agent.fit(env, nb_steps=1000000, visualize=True, verbose=1, nb_max_episode_steps=200)
    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=50, visualize=True, nb_max_episode_steps=200)
    
    # After training is done, we save the final weights.
    agent.save_weights('tmp/ddpg_{}_weights.h5f'.format(env_name), overwrite=True)
    # Save memory
    pickle.dump(memory, open('tmp/ddpg_{}_weights.mdl'.format(env_name), "wb"))
    
def eval(env_name):
    env = gym.make(env_name)
    random_seed = 123
    np.random.seed(random_seed)
    tf.random.set_random_seed(random_seed)
    env.seed(random_seed)
    
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]
    
    actor = actor_model(env)
    action_input, critic = critic_model(env)
    memory = SequentialMemory(limit=100000, window_length=1)
    
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, 
                                              theta=.15, 
                                              mu=0., 
                                              sigma=.3)
    
    agent = DDPGAgent(nb_actions=nb_actions, 
                      actor=actor, 
                      critic=critic, 
                      critic_action_input=action_input,
                      memory=memory, 
                      nb_steps_warmup_critic=100, 
                      nb_steps_warmup_actor=100,
                      random_process=random_process, 
                      gamma=.99, 
                      target_model_update=1e-2)
    
    # optimizer
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    
    try:
        agent.load_weights('tmp/ddpg_{}_weights.h5f'.format(env_name))
    except (OSError):
        print("no weights file, train from the beginning.")
        
    agent.test(env, nb_episodes=50, visualize=True)
     
if __name__ == '__main__':
    env_name = 'Continuous_OSG_TowerArc-v0'
    # tensorflow_init()
    train(env_name)
    # eval(env_name) # 训练结果测试