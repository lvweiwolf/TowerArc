import math
from math import fabs
import re
import numpy as np
from tensorflow.python.framework.tensor_shape import vector
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
        1         杆1 高度      5               30
        2         杆2 高度      5               30
        ...
        ...         杆1与杆2间K值 0               7e-4
       
        
        
    动作 (Action):
        Type: ndarray[16]
        序号       动作
        
        1          杆1 +h      杆2 +h       +k
        2          杆1 +h      杆2 +h       -k
        3          杆1 +h      杆2 -h       +k
        4          杆1 +h      杆2 -h       -k
        5          杆1 -h      杆2 +h       +k
        6          杆1 -h      杆2 +h       -k
        7          杆1 -h      杆2 -h       +k
        8          杆1 -h      杆2 -h       -k
        ...
        
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
        self.max_speed = 10
        self.min_height = 20
        self.max_height = 80
        
        self.towers = []
        self.tower_velocity = []
        self.arclines = []
        
        # towerPositions = [(-239.130, -1078.551), 
        #                   (-466.500, -647.039),
        #                   (-629.240, -116.453),
        #                   (-1017.519, 229.423),
        #                   (-753.508, 772.095)]
        towerPositions = [(-239.130, -1078.551), 
                          (-466.500, -647.039),
                          (-629.240, -116.453)]
        
        self.seed()
        self.world = env.World()
        self.viewer = self.world.GetViewer()
        
        # 创建地形
        self.terrain = self.world.CreateTerrainBody(DEM_TIFF, DOM_TIFF)
        extents = self.terrain.extent

        # 创建杆塔
        for x, y in towerPositions:
            tower = self.world.CreateTowerBody(x, y, self.max_height, TOWER_MODEL_OBJ)
            self.towers.append(tower)
            self.tower_velocity.append(0.0)
        
        # 创建弧垂
        for i in range(len(self.towers) - 1):
            curTower = self.towers[i]
            nextTower = self.towers[i + 1]
            arcline = self.world.CreateArclineBody(curTower, nextTower, self.K, 5)
            self.arclines.append(arcline)
    
        low = np.array([0.0 for _ in self.towers] + [-1.0 for _ in self.tower_velocity])
        high = np.array([1.0 for _ in self.towers] + [1.0 for _ in self.tower_velocity])
        
        # 暂时固定K值，只考虑杆塔高度
        self.action_space = spaces.Discrete(int(math.pow(2, len(self.towers))))
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.reset()
        
    def _destroy(self):
        if self.terrain:
            self.world.DeleteTerrainBody(self.terrain)
            self.terrain = None
        
        for tower in self.towers:
            self.world.DeleteTowerBody(tower)
        
        self.towers = None

        for arcline in self.arclines:
            self.world.DeleteArclineBody(arcline)
            
        self.arclines = None

    def _get_accelerate(self, action):
        acc = [0.5 for _ in self.towers]
        
        # 将action转化为二进制
        format_str = '{' + f':0{len(acc)}b' + '}'
        bitstr = format_str.format(action)
        
        for i in range(len(acc)):
            if bitstr[i] == '0':
                acc[i] = -acc[i]
                
        return acc
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]          
    
    def reset(self):
        extents = self.terrain.extent
        W = extents.max_x - extents.min_x
        H = extents.max_y - extents.min_y
      
        for tower in self.towers:
            tower.height = self.max_height
            
        for velocity in self.tower_velocity:
            velocity = self.np_random.uniform(-self.max_speed, self.max_speed)
       
        # 更新弧垂
        for arcline in self.arclines:
            self.world.UpdateArcline(arcline)
      
        # 杆塔高度 + 杆塔速度
        self.state = np.array([(tower.height - self.min_height) / (self.max_height - self.min_height) 
                     for tower in self.towers] + [velocity / self.max_speed for velocity in self.tower_velocity])
        
        return np.array(self.state)
    

    def step(self, action):
        extents = self.terrain.extent
        W = extents.max_x - extents.min_x
        H = extents.max_y - extents.min_y
        Z = extents.max_z - extents.min_z
        
        tower_states = []
        velocity_states = []
        velocity_acc = self._get_accelerate(action)
        
        for i in range(len(self.towers)):
            tower_h = self.state[i] * (self.max_height - self.min_height) + self.min_height
            tower_v = self.state[len(self.towers) + i] * self.max_speed
            tower_h += self.tau * tower_v

            self.towers[i].height = tower_h
            tower_states.append((self.towers[i].height - self.min_height) / (self.max_height - self.min_height))
            
            tower_v += self.tau * velocity_acc[i]
            tower_v = min(self.max_speed, max(tower_v, -self.max_speed))
            velocity_states.append( tower_v / self.max_speed)
            
        # 设置新的状态
        self.state = np.array(tower_states + velocity_states)
      
        has_invalid_tower = False
        for tower in self.towers:
            if tower.height < self.min_height or tower.height > self.max_height:
                has_invalid_tower = True
                break
      
        self.references = []
        min_distance_to_terrain = math.inf
        
        # 计算弧垂前，更新弧垂点
        for arcline in self.arclines:
            lowestPnt = env.Point3D()
            self.world.UpdateArcline(arcline)
            distance_to_terrain = self.world.CalcLowestDistance(arcline, lowestPnt)

            endPnt = lowestPnt
            startPnt = env.Point3DCopy(endPnt)
            startPnt.z -= distance_to_terrain
            self.references.append((startPnt, endPnt))
            
            if distance_to_terrain < min_distance_to_terrain:
                min_distance_to_terrain = distance_to_terrain
            
        # tower1_is_out = (self.tower1.x < extents.min_x 
        #                 or self.tower1.x > extents.max_x
        #                 or self.tower1.y < extents.min_y
        #                 or self.tower1.y > extents.max_y)
        
        # tower2_is_out = (self.tower2.x < extents.min_x 
        #                 or self.tower2.x > extents.max_x  
        #                 or self.tower2.y < extents.min_y
        #                 or self.tower2.y > extents.max_y)
        
        done = (min_distance_to_terrain < 0.0 or 
                min_distance_to_terrain == math.inf or
                has_invalid_tower)
        
        reward = 0.0
        
        if not done:
            reward = min_distance_to_terrain / Z
        else:
            reward = 0.0
        
        # for tower in self.towers:
        #     tower_cost = ((tower.height - self.min_height) / self.max_height) / len(self.towers)
        #     reward -= 0.4 * 100.0 * tower_cost
        
        # for arcline in self.arclines:
        #     arcline_cost = (1.0 - (arcline.K - self.min_K) / self.max_K) / len(self.arclines)
        #     reward -= 0.6 * 100.0 * arcline_cost
        
        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    
    def render(self, mode='human'):
        if self.viewer is None:
            return
        
        if self.arclines:
            for arcline in self.arclines:
                self.viewer.DrawArcline(arcline)
            
        if self.references and len(self.references):
            for startPnt, endPnt in self.references:
                self.viewer.DrawReferenceLine(startPnt, endPnt)
        
        if self.towers:
            for tower in self.towers:
                self.viewer.DrawTower(tower)
            
    def close(self):
        
        if self.terrain:
            self.world.DeleteTerrainBody(self.terrain)
            self.terrain = None
        
        self._destroy()
        
            