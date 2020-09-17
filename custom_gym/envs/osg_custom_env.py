import math
import re
import numpy as np
import osgGymEnv as env

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from osgGymEnv import World

DEM_TIFF = "C:\\Users\\lvwei\\Desktop\\export\\dem.tif"
DOM_TIFF = "C:\\Users\\lvwei\\Desktop\\export\\dom2.tif"
TOWER_MODEL_OBJ = "C:\\Users\\lvwei\\Desktop\\export\\tower.obj"

class OSG_TowerArcEnv(gym.Env):
    """
    说明:
        两基杆塔立在崎岖的地形上，每根杆塔根据提供的状态可以向左、向右、
        加速移动或停止。
        
    状态 (State):
        Type: ndarray[4]
        序号      状态          最小值          最大值
        1         杆1 x位置     extents.min_x   extents.max_x
        2         杆1 y位置     extents.min_y   extents.max_y
        3         杆2 x位置     extents.min_x   extents.max_x
        4         杆2 y位置     extents.min_y   extents.max_y
        5         杆1 x速度     -0.5            0.5
        6         杆1 y速度     -0.5            0.5
        7         杆2 x速度     -0.5            0.5
        7         杆2 y速度     -0.5            0.5
        
    动作 (Action):
        Type: ndarray[16]
        序号       动作
        
        0          杆1 +x, +y  杆2 +x, +y
        1          杆1 +x, +y  杆2 +x, -y
        2          杆1 +x, +y  杆2 -x, +y
        3          杆1 +x, +y  杆2 -x, -y
        4          杆1 +x, -y  杆2 +x, +y
        5          杆1 +x, -y  杆2 +x, -y
        6          杆1 +x, -y  杆2 -x, +y
        7          杆1 +x, -y  杆2 -x, -y
        8          杆1 -x, +y  杆2 +x, +y
        9          杆1 -x, +y  杆2 +x, -y
        10         杆1 -x, +y  杆2 -x, +y
        11         杆1 -x, +y  杆2 -x, -y
        12         杆1 -x, -y  杆2 +x, +y
        13         杆1 -x, -y  杆2 +x, -y
        14         杆1 -x, -y  杆2 -x, +y
        15         杆1 -x, -y  杆2 -x, -y
        
    奖励 (Reward):
        假设有杆1挂点与杆2挂点按照悬链线方程计算的弧垂挂点，最低弧垂点
        在地形下方，奖励为'-'; 最低弧垂点在地形上方奖励为'+', 奖励值为
        最低点到地形的垂直距离.

    初始状态:
        杆1的X轴坐标被分配到[0.0, VIEWPORT_W / SCALE / 3.0]的随机位置
        杆2的X轴坐标被分配到[VIEWPORT_W / SCALE / 3.0, VIEWPORT_W / SCALE]的随机位置
        杆1、杆2的起始速度为0     
    """
    
    def __init__(self):
        super(OSG_TowerArcEnv, self).__init__()
    
        self.K = 7e-5
        self.tau = 1.0
        self.max_speed = 20
            
        self.world = env.World()
        self.viewer = self.world.GetViewer()
        
        self.terrain = self.world.CreateTerrainBody(DEM_TIFF, DOM_TIFF)
        # 创建杆塔 1
        self.tower1 = self.world.CreateTowerBody(0, 0, TOWER_MODEL_OBJ)
        # 创建杆塔2
        self.tower2 = self.world.CreateTowerBody(0, 0, TOWER_MODEL_OBJ)
        # 创建弧垂
        self.arcline = self.world.CreateArclineBody(self.tower1, self.tower2, self.K, 5)
        
        
        self.seed()
        self.reset()
        
    def _destroy(self):
        if self.terrain:
            self.world.DeleteTerrainBody(self.terrain)
            self.terrain = None
        
        if self.tower1:
            self.world.DeleteTowerBody(self.tower1)
            self.tower1 = None
            
        if self.tower2:
            self.world.DeleteTowerBody(self.tower2)
            self.tower2 = None
            
        if self.arcline:
            self.world.DeleteArclineBody(self.arcline)
            self.arcline = None
    
    def _get_accelerate(self, action):
        v = 10.0
        acc = [v, v, v, v]        
        
        # 将action转化为二进制
        bitstr = '{:04b}'.format(action)
        
        for i in range(len(acc)):
            if bitstr[i] == '0':
                acc[i] = -acc[i]
                
        return acc
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]          
    
    def reset(self):
        # if self.tower1:
        #     self.world.DeleteTowerBody(self.tower1)
        #     self.tower1 = None
            
        # if self.tower2:
        #     self.world.DeleteTowerBody(self.tower2)
        #     self.tower2 = None
            
        # if self.arcline:
        #     self.world.DeleteArclineBody(self.arcline)
        #     self.arcline = None
        extents = self.terrain.extent
        W = extents.max_x - extents.min_x
        H = extents.max_y - extents.min_y
      
        # 初始化杆1的位置
        self.tower1.x = self.np_random.uniform(extents.min_x, extents.max_x)
        self.tower1.y = self.np_random.uniform(extents.min_y, extents.max_y)
        
        # 初始化杆2的位置
        self.tower2.x = self.np_random.uniform(extents.min_x, extents.max_x)
        self.tower2.y = self.np_random.uniform(extents.min_y, extents.max_y)
       
        # 更新弧垂
        self.world.UpdateArcline(self.arcline)
        
        self.state = np.array([(self.tower1.x - extents.min_x) / W,
                               (self.tower1.y - extents.min_y) / H,
                               (self.tower2.x - extents.min_x) / W,
                               (self.tower2.y - extents.min_y) / H,
                               self.np_random.uniform(-self.max_speed, self.max_speed),
                               self.np_random.uniform(-self.max_speed, self.max_speed),
                               self.np_random.uniform(-self.max_speed, self.max_speed),
                               self.np_random.uniform(-self.max_speed, self.max_speed)])
        
        return np.array(self.state)
    

    def step(self, action):
        extents = self.terrain.extent
        W = extents.max_x - extents.min_x
        H = extents.max_y - extents.min_y
        Z = extents.max_z - extents.min_z
        
        # 获取当前状态
        tower1_x = self.state[0] * W + extents.min_x
        tower1_y = self.state[1] * H + extents.min_y
        tower2_x = self.state[2] * W + extents.min_x
        tower2_y = self.state[3] * H + extents.min_y
        tower1_vx = self.state[4]
        tower1_vy = self.state[5]
        tower2_vx = self.state[6]
        tower2_vy = self.state[7]
        
        # 根据当前状态中的各分量速度，更新坐标
        tower1_x += self.tau * tower1_vx
        tower1_y += self.tau * tower1_vy
        tower2_x += self.tau * tower2_vx
        tower2_y += self.tau * tower2_vy
        
        tower1_vx_acc, tower1_vy_acc, tower2_vx_acc, tower2_vy_acc = \
            self._get_accelerate(action)
        # 根据加速度更新新的速度值
        tower1_vx += self.tau * tower1_vx_acc
        tower1_vy += self.tau * tower1_vy_acc
        tower2_vx += self.tau * tower2_vx_acc
        tower2_vy += self.tau * tower2_vy_acc
        
        # 更新杆塔位置和弧垂
        self.tower1.x = tower1_x
        self.tower1.y = tower1_y
        self.tower2.x = tower2_x
        self.tower2.y = tower2_y
        
        # 设置新的状态
        self.state = [
            (self.tower1.x - extents.min_x) / W,
            (self.tower1.y - extents.min_y) / H,
            (self.tower2.x - extents.min_x) / W,
            (self.tower2.y - extents.min_y) / H,
            tower1_vx,
            tower1_vy,
            tower2_vx,
            tower2_vy
        ]
        
        reward = 0
        lowestPnt = env.Point3D()
        # 计算弧垂前，更新弧垂点
        start_tower_x = self.arcline.startTower.x
        self.world.UpdateArcline(self.arcline)
        min_distance_to_terrain = self.world.CalcLowestDistance(self.arcline,
                                                                lowestPnt)

        self.endPnt = lowestPnt
        self.startPnt = env.Point3DCopy(self.endPnt)
        self.startPnt.z -= min_distance_to_terrain
        
        tower1_is_out = (self.tower1.x < extents.min_x 
                        or self.tower1.x > extents.max_x
                        or self.tower1.y < extents.min_y
                        or self.tower1.y > extents.max_y)
        
        tower2_is_out = (self.tower2.x < extents.min_x 
                        or self.tower2.x > extents.max_x
                        or self.tower2.y < extents.min_y
                        or self.tower2.y > extents.max_y)
        
        done = (tower1_is_out or 
                tower2_is_out or
                min_distance_to_terrain < 0.0 or
                min_distance_to_terrain == math.inf )
        
        if not done:
            reward = min_distance_to_terrain / Z
            distance_between_tower = self.tower1.DistanceTo(self.tower2)
            if distance_between_tower < 50.0:
                reward = 0.0
        else:
            reward = 0.0
        
        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    
    def render(self):
        if self.viewer is None:
            return
        
        if self.arcline:
            self.viewer.DrawArcline(self.arcline)
            
            if self.startPnt and self.endPnt: 
                self.viewer.DrawReferenceLine(self.startPnt, self.endPnt)
        
        if self.tower1:
            self.viewer.DrawTower(self.tower1)
        
        if self.tower2:
            self.viewer.DrawTower(self.tower2)
            
    def close(self):
        
        if self.terrain:
            self.world.DeleteTerrainBody(self.terrain)
            self.terrain = None
        
        self._destroy()
        
            