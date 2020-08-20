import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle


FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well


VIEWPORT_W = 600     # 窗口宽度
VIEWPORT_H = 400     # 窗口高度

class TowerArcEnv(gym.Env):
    
    continuous = False
    
    def __init__(self):
        super(TowerArcEnv, self).__init__()
        self.seed()
        self.viewer = None
      
        self.world = Box2D.b2World()
        self.terrain = None
        self.tower1 = None
        self.tower2 = None
        self.arcline = None
        
        self.pre_reward = None
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        
        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)
            
            
          
        self.reset()
        
    def _destroy(self):
        self.world.contactListener = None
        
        if self.terrain:
            self.world.DestroyBody(self.terrain)
            self.terrain = None
        
        if self.tower1:
            self.world.DestroyBody(self.tower1)
            self.tower1 = None
        
        if self.tower2:
            self.world.DestroyBody(self.tower2)
            self.tower2 = None
            
        if self.arcline:
            self.world.DestroyBody(self.arcline)
            self.arcline = None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        self._destroy()
        # self.world.contactListener_keepref = ContactDetector(self)
        # self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        
        # 地形
        CHUNKS = 31
        # 生成随机高度值
        heights = self.np_random.uniform(0, H/2., size=(CHUNKS+1, ))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        smooth_y = [0.33*(heights[i-1] + heights[i+0] + heights[i+1]) for i in range(CHUNKS)]
        
        self.terrain = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0,0), (W, 0)]))
        self.terrain_polys = []
        
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            # 为地形刚体填充边缘顶点
            self.terrain.CreateEdgeFixture(vertices=[p1,p2],  density=0, friction=0.1)
            # 为地形绘制顶点数据填充顶点
            self.terrain_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        
        self.terrain.color1 = (0.0, 0.0, 0.0)
        self.terrain.color2 = (0.0, 0.0, 0.0)
        
        # 创建杆塔
        self.tower1 = self.world.CreateDynamicBody()
        
        
        # return self.step(np.array([0, 0]) if self.continuous else 0)[0]
        return None

    def step(self, action):
        print('Custom eviroment \'step\' called.')
        
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
        
        
    def render(self):
        from gym.envs.classic_control import rendering
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)
        
        for p in self.terrain_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))
        
        return self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None