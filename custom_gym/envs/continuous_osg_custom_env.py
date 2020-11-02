import gym
import math
import numpy as np
import osgGymEnv as env

from gym import spaces
from gym.utils import seeding

DEM_TIFF = "C:\\Users\\lvwei\\Desktop\\export\\dem.tif"
DOM_TIFF = "C:\\Users\\lvwei\\Desktop\\export\\dom2.tif"
TOWER_MODEL_OBJ = "C:\\Users\\lvwei\\Desktop\\export\\tower.obj"

CARE_ABOUT_ARCLINE = True

class Continuous_OSG_TowerArcEnv(gym.Env):
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
        Type: ndarray[4]
        序号       动作
        
        1          杆1速度，        +v, -v
        2          杆2速度，        +v, -v
        3          杆3速度，        +v, -v
        4          K值变化速度      +k, -k
        
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
        super(Continuous_OSG_TowerArcEnv, self).__init__()
        self.action_cost_weight = 0.1
        self.max_acc = 2.0
        self.dt = 0.5
        
        self.min_height = 20
        self.max_height = 80
        self.max_speed = 2
        
        self.min_K = 7e-6
        self.max_K = 7e-4        
        self.max_K_speed = 5e-6
        
        self.towers = []
        self.tower_velocity = []
        self.arclines = []
        self.K_velocity = []
        self.referenceLines = []
        
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
            arcline = self.world.CreateArclineBody(curTower, nextTower, 7e-5, 5)
            self.arclines.append(arcline)
            self.K_velocity.append(0.0)
            
            # 创建对地参考线
            referenceLine = self.world.CreateLineBody(env.Point3D(), env.Point3D())
            self.referenceLines.append(referenceLine)
        
        if CARE_ABOUT_ARCLINE:        
            low = np.array([0.0 for _ in self.towers] +               # 杆塔最小高度
                           [0.0 for _ in self.arclines] +             # K 最小值
                           [-1.0 for _ in self.tower_velocity] +       # 杆塔最小速度
                           [-1.0 for _ in self.K_velocity])            # K 最小速度
            
            high = np.array([1.0 for _ in self.towers] +            # 杆塔最大高度    
                            [1.0 for _ in self.arclines] +          # K 最大值
                            [1.0 for _ in self.tower_velocity] +    # 杆塔最大速度
                            [1.0 for _ in self.K_velocity])         # K 最大速度
        else:
            low = np.array([0.0 for _ in self.towers] +             # 杆塔最小高度
                           [-1.0 for _ in self.tower_velocity])     # 杆塔最小速度
                 
            high = np.array([1.0 for _ in self.towers] +            # 杆塔最大高度
                            [1.0 for _ in self.tower_velocity])     # 杆塔最大速度
        
        # state 维度空间
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        if CARE_ABOUT_ARCLINE:
            low = np.array([-self.max_acc for _ in self.tower_velocity] +       # 杆塔最小加速度
                           [-self.max_acc for _ in self.K_velocity])          # K 最小加速度
            
            high = np.array([self.max_acc for _ in self.tower_velocity] +    # 杆塔最大加速度
                            [self.max_acc for _ in self.K_velocity])       # K 最大加速度
        else:
            low = np.array([-self.max_acc for _ in self.tower_velocity])     # 杆塔最小加速度
            high = np.array([self.max_acc for _ in self.tower_velocity])     # 杆塔最大加速度
        
        # action 维度空间
        self.action_space = spaces.Box(low, high, dtype=np.float32)

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
        tower_accs = action[:len(self.tower_velocity)]
        K_accs = action[len(self.tower_velocity):]
        
        return tower_accs, K_accs
    
    def _action_cost(self, action):
        cost = -self.action_cost_weight * np.sum(np.square(action))
        return cost
    
    def _get_observation(self):
        def norm(value, min, max):
            return (value - min) / (max - min)
        
        if CARE_ABOUT_ARCLINE:
            tower_heights, K_values, tower_velocity, K_velocity = self.state
            observation = [norm(np.array(tower_heights, dtype=np.float32), self.min_height, self.max_height),
                           norm(np.array(K_values, dtype=np.float32), self.min_K, self.max_K),
                           np.array(tower_velocity, dtype=np.float32) / self.max_speed,
                           np.array(K_velocity, dtype=np.float32) / self.max_K_speed]
        else:
            tower_heights, tower_velocity = self.state
            observation = [norm(np.array(tower_heights, dtype=np.float32), self.min_height, self.max_height),
                           np.array(tower_velocity, dtype=np.float32) / self.max_speed]
        
        observation = np.hstack(observation)
        
        return observation
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]          
    
    def reset(self):
        # 初始化杆塔高度
        for tower in self.towers:
            tower.height = self.min_height + (self.max_height - self.min_height) / 2.
            
        # 初始化杆塔速度
        for i in range(len(self.tower_velocity)):
            #self.tower_velocity[i] = 0.0 # 将杆塔初始速度设置为0
            self.tower_velocity[i] = self.np_random.uniform(-self.max_speed/2, self.max_speed/2)

        # 初始化弧垂K值, 并计算弧垂点
        for i in range(len(self.arclines)):
            self.arclines[i].K = 7e-5
            self.world.UpdateArcline(self.arclines[i])
            
        # 初始化K值速度
        for i in range(len(self.K_velocity)):
            self.K_velocity[i] = self.np_random.uniform(-self.max_K_speed/2, self.max_K_speed/2)
      
        if CARE_ABOUT_ARCLINE:
            # 杆塔高度 + K值 + 杆塔速度 + K值速度
            self.state = [
                [tower.height for tower in self.towers],        # 杆塔高度
                [arcline.K for arcline in self.arclines],       # K值
                [velocity for velocity in self.tower_velocity], # 杆塔速度
                [velocity for velocity in self.K_velocity]      # K值速度
            ]
        else:
            self.state = [
                [tower.height for tower in self.towers],        # 杆塔高度
                [velocity for velocity in self.tower_velocity]  # 杆塔速度
            ]
        
        return self._get_observation()
    
    def step(self, action):
        tower_accs, K_accs = self._get_accelerate(action)
        
        if CARE_ABOUT_ARCLINE:
            tower_heights, K_values, tower_velocity, K_velocity = self.state
            
            assert len(tower_accs) == len(self.towers), "length between tower_accs and self.towers must equal."
            assert len(K_accs) == len(self.arclines), "length between K_accs and self.arclines must equal."
            assert len(tower_heights) == len(tower_velocity), "length between tower_heights and tower_velocity must equal."
            assert len(tower_heights) == len(self.towers), "length between tower_heights and self.towers must equal."
            assert len(K_values) == len(K_velocity), "length between K_values and K_velocity must equal."
            assert len(K_values) == len(self.arclines), "length between K_values and self.arclines must equal."
            
            for i in range(len(self.towers)):
                tower_h = tower_heights[i]
                tower_v = tower_velocity[i]
                
                # 加速度
                acc = np.clip(tower_accs[i], -self.max_acc, self.max_acc)
                # 根据加速度更新速度
                tower_v += acc * self.dt
                # 限制最大速度
                tower_v = np.clip(tower_v, -self.max_speed, self.max_speed)
                # 更新杆塔高度
                tower_h += tower_v

                self.towers[i].height = tower_h
                
                # 裁剪杆塔高度
                tower_h = np.clip(tower_h, self.min_height, self.max_height)
                
                if tower_h == self.min_height and tower_v < 0:
                    tower_v = 0
                    
                tower_heights[i] = tower_h
                tower_velocity[i] = tower_v
                
            for i in range(len(self.arclines)):
                k = K_values[i]
                k_v = K_velocity[i]

                # K值变化加速度
                acc = np.clip(K_accs[i], -self.max_acc, self.max_acc)
                # 根据加速度更新K变化速度
                k_v += acc * self.dt * 1e-6
                # 限制最大速度
                k_v = np.clip(k_v, -self.max_K_speed, self.max_K_speed)
                # 更新K值
                k += k_v
                
                self.arclines[i].K = k
                
                # 裁剪K值
                k = np.clip(k, self.min_K, self.max_K)
                
                if k == self.min_K and k_v < 0:
                    k_v = 0
                
                K_values[i] = k
                K_velocity[i] = k_v

            # 设置新的状态
            self.state = [tower_heights, K_values, tower_velocity, K_velocity]
        else:
            tower_heights, tower_velocity = self.state
        
            assert len(tower_accs) == len(self.towers), "length between tower_accs and self.towers must equal."
            assert len(tower_heights) == len(tower_velocity), "length between tower_heights and tower_velocity must equal."
            assert len(tower_heights) == len(self.towers), "length between tower_heights and self.towers must equal."
        
            for i in range(len(self.towers)):
                tower_h = tower_heights[i]
                tower_v = tower_velocity[i]
                
                # 加速度
                acc = np.clip(tower_accs[i], -self.max_acc, self.max_acc)
                # 根据加速度更新速度
                tower_v += acc * self.dt
                # 限制最大速度
                tower_v = np.clip(tower_v, -self.max_speed, self.max_speed)
                # 更新杆塔高度
                tower_h += tower_v
                
                self.towers[i].height = tower_h
                
                # 裁剪杆塔高度
                tower_h = np.clip(tower_h, self.min_height, self.max_height)
                
                if tower_h == self.min_height and tower_v < 0:
                    tower_v = 0
                    
                tower_heights[i] = tower_h
                tower_velocity[i] = tower_v
                
            self.state = [tower_heights, tower_velocity]

        has_invalid_tower = False
        has_invalid_K = False
        min_distance_to_terrain = math.inf
        max_k = -math.inf
        
        for tower in self.towers:
            # 是否存在无效杆塔
            if tower.height < self.min_height or tower.height > self.max_height:
                has_invalid_tower = True
                break
        
        # 计算弧垂前，更新弧垂点 
        for i in range(len(self.arclines)):
            arcline = self.arclines[i]
            referenceLine = self.referenceLines[i]
            
            # 是否存在无效弧垂
            if not has_invalid_K:
                has_invalid_K = arcline.K < self.min_K or arcline.K > self.max_K
            
            lowestPnt = env.Point3D()
            self.world.UpdateArcline(arcline)
            distance_to_terrain = self.world.CalcLowestDistance(arcline, lowestPnt)

            # 最低点参考线referenced line
            endPnt = lowestPnt
            startPnt = env.Point3DCopy(endPnt)
            startPnt.z -= distance_to_terrain
            referenceLine.startPnt = startPnt
            referenceLine.endPnt = endPnt
            
            if distance_to_terrain < min_distance_to_terrain:
                min_distance_to_terrain = distance_to_terrain
                
            if arcline.K > max_k:
                max_k = arcline.K
            

        if CARE_ABOUT_ARCLINE:
            done = (min_distance_to_terrain < 10.0 or 
                    min_distance_to_terrain == math.inf or
                    has_invalid_tower or
                    has_invalid_K)
        else:
            done = (min_distance_to_terrain < 10.0 or 
                    min_distance_to_terrain == math.inf or
                    has_invalid_tower)
        
        reward = -1.0
        
        if not done:
            # 选择合适的回报比例分配，有助于快速收敛
            # 弧垂对地距离reward
            distance_reward = -1.0 + 2.0 * min_distance_to_terrain / self.max_height
            # K值reward
            K_reward = (max_k - self.min_K) / (self.max_K - self.min_K)
            
            reward =  0.05*distance_reward + 0.95*K_reward
        else:
            reward = -1.0
         
         
         
        # print(f'reward: {reward}, action: {action}.')   
        reward += self._action_cost(action)
       
        return self._get_observation(), reward, done, {}
    
    
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
        
            