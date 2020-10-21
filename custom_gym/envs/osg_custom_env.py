import gym
import math
import numpy as np
import osgGymEnv as env

from gym import spaces
from gym.utils import seeding

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
        n + 1     杆1 速度      -max_speed      max_speed
        n + 2     杆2 速度      -max_speed      max_speed
        
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
       
        self.min_height = 20
        self.max_height = 80
        self.max_speed = 2
        
        self.towers = []
        self.tower_velocity = []
        self.arclines = []
        self.referenceLines = []
        
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
            
            # 创建对地参考线
            referenceLine = self.world.CreateLineBody(env.Point3D(), env.Point3D())
            self.referenceLines.append(referenceLine)
            
    
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
        
        for referenceLine in self.referenceLines:
            self.world.DeleteLineBody(referenceLine)
            
        self.referenceLines = None

    def _get_accelerate(self, action):
        acc = [0.5 for _ in self.towers]
        
        # 将action转化为二进制
        format_str = '{' + f':0{len(acc)}b' + '}'
        bitstr = format_str.format(action)
        
        for i in range(len(acc)):
            if bitstr[i] == '0':
                acc[i] = -acc[i]
                
        return acc



    # def _unnorm_velocity(self, velocity):
    #     return velocity * self.max_speed
    
    def _normalize(self, value, min, max):
        return (value - min) / (max - min)
    
    def _unnormalize(self, value, min, max):
        return value * (max - min) + min
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]          
    
    def reset(self):
        extents = self.terrain.extent
        W = extents.max_x - extents.min_x
        H = extents.max_y - extents.min_y
      
        for tower in self.towers:
            tower.height = self._unnormalize(0.5, self.min_height, self.max_height)
            
        for i in range(len(self.tower_velocity)):
            self.tower_velocity[i] = 0.0 # 将杆塔高度初始速度设置为0
            # self.tower_velocity[i] = self.np_random.uniform(-self.max_speed/2, self.max_speed/2)
       
        # 更新弧垂
        for arcline in self.arclines:
            self.world.UpdateArcline(arcline)
      
        # 杆塔高度 + 杆塔速度
        self.state = np.array([self._normalize(tower.height, self.min_height, self.max_height) 
                               for tower in self.towers] + 
                              [self._normalize(velocity, -self.max_speed, self.max_speed) 
                               for velocity in self.tower_velocity])
        
        return np.array(self.state)
    

    def step(self, action):
        extents = self.terrain.extent
        W = extents.max_x - extents.min_x
        H = extents.max_y - extents.min_y
        
        tower_states = []
        velocity_states = []
        velocity_acc = self._get_accelerate(action)
        
        for i in range(len(self.towers)):
            tower_h = self._unnormalize(self.state[i], 
                                        self.min_height, self.max_height)
            tower_v = self._unnormalize(self.state[len(self.towers) + i],
                                        -self.max_speed, 
                                        self.max_speed)
            tower_h += self.tau * tower_v

            # 先更新位置，后更新速度
            self.towers[i].height = tower_h
            tower_states.append(self._normalize(self.towers[i].height, 
                                                self.min_height, self.max_height))
            
            tower_v += self.tau * velocity_acc[i]
            tower_v = np.clip(tower_v, -self.max_speed, self.max_speed)
            velocity_states.append(self._normalize(tower_v, 
                                                   -self.max_speed, 
                                                   self.max_speed))
            
        # 设置新的状态
        self.state = np.array(tower_states + velocity_states)
      
        has_invalid_tower = False
        for tower in self.towers:
            if tower.height < self.min_height or tower.height > self.max_height:
                has_invalid_tower = True
                break
      
        min_distance_to_terrain = math.inf
        
        # 计算弧垂前，更新弧垂点
        for i in range(len(self.arclines)):
            arcline = self.arclines[i]
            referenceLine = self.referenceLines[i]
            
            lowestPnt = env.Point3D()
            self.world.UpdateArcline(arcline)
            distance_to_terrain = self.world.CalcLowestDistance(arcline, lowestPnt)

            endPnt = lowestPnt
            startPnt = env.Point3DCopy(endPnt)
            startPnt.z -= distance_to_terrain
            #self.references.append((startPnt, endPnt))
            referenceLine.startPnt = startPnt
            referenceLine.endPnt = endPnt
            
            if distance_to_terrain < min_distance_to_terrain:
                min_distance_to_terrain = distance_to_terrain
                
        done = (min_distance_to_terrain < 0.0 or 
                min_distance_to_terrain == math.inf or
                has_invalid_tower)
        
        reward = 0.0
        
        if not done:
            reward = -1.0 + 2.0 * min_distance_to_terrain / self.max_height
        else:
            reward = -1.0
   
        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    
    def render(self, mode='human'):
        if self.viewer is None:
            return
        
        if self.arclines:
            for arcline in self.arclines:
                self.viewer.RecordArcline(arcline)
            
        if self.referenceLines:
            for referenceLine in self.referenceLines:
                self.viewer.RecordLine(referenceLine)
        
        if self.towers:
            for tower in self.towers:
                self.viewer.RecordTower(tower)
                
        # 统一绘制
        self.viewer.DrawRecords()
            
    def close(self):
        
        if self.terrain:
            self.world.DeleteTerrainBody(self.terrain)
            self.terrain = None
        
        self._destroy()
        
            