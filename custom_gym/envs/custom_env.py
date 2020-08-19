import sys, math
import numpy as np

import Box2D

import gym
from gym import spaces
from gym.utils import seeding, EzPickle


class TowerArcEnv(gym.Env):
    
    def __init__(self):
        super(TowerArcEnv, self).__init__()
        
    def step(self):
        print('Custom eviroment \'step\' called.')
        
    def reset(self):
        print('Custom eviroment \'reset\' called.')