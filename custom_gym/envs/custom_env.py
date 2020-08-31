from math import tan
import os, math
from gym.envs.classic_control.rendering import LineWidth
import numpy as np
import Box2D
from Box2D.Box2D import (b2CircleShape, b2EdgeShape, 
                         b2PolygonShape, 
                         b2FixtureDef,
                         b2Vec2, 
                         b2ContactListener, 
                         b2RopeJointDef, b2Vec3)


import gym
from gym import spaces
from gym.utils import seeding, EzPickle


FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

VIEWPORT_W = 600     # 窗口宽度
VIEWPORT_H = 400     # 窗口高度
NUM_ARCS = 30

BODY_W = 0.1
BODY_H = 5.0
AUTO_POINT_NUM = 20

class TowerArcEnv(gym.Env):
    """
    说明:
        两基杆塔立在崎岖的地形上，每根杆塔根据提供的状态可以向左、向右、
        加速移动或停止。
        
    状态 (State):
        Type: ndarray[4]
        序号      状态          最小值      最大值
        1         杆1位置       0.0         VIEWPORT_W / SCALE
        2         杆2位置       0.0         VIEWPORT_W / SCALE
        3         杆1速度       -0.5        0.5
        4         杆2速度       -0.5        0.5
        
    动作 (Action):
        Type: Box(2, 3)
        序号      动作
        0         向左加速
        1         无加速
        2         向右加速
        
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
        super(TowerArcEnv, self).__init__()
        self.seed()
        self.viewer = None
      
        self.world = Box2D.b2World()
        self.terrain = None
        self.tower1 = None
        self.tower2 = None
        self.arcline = None
        self.max_speed = 0.5
        self.min_x = 0.0
        self.max_x = VIEWPORT_W / SCALE
        
        self.pre_reward = None
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(0, 3, (2,), dtype=np.int32)
          
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
            
    def _calc_heights(self, heights, chunk_x, x):
        assert len(heights) == len(chunk_x), "heights's length must equal to chunk_x's length" 
        
        pi = None
        ni = None
        
        for i in range(len(chunk_x)):
            if i != len(chunk_x) - 1:
                prev = chunk_x[i]
                next = chunk_x[i + 1]
                
                if prev <= x and x <= next:
                    pi = i
                    ni = i + 1
    
        if pi and ni:
            h = ((x - chunk_x[pi]) / (chunk_x[ni] - chunk_x[pi])
                 ) * (heights[ni] - heights[pi]) + heights[pi]

            return h
        else:
            return 0
        
    def _calc_arcpnts(self, p1, p2, k, step):
        assert len(p1) == 3, "point dim must equal 3."
        assert len(p2) == 3, "point dim must equal 3."
        assert k > 0.0, "K must greator than 0.0."
        assert step > 0.0, "step must greator than 0.0"
        
        p1x, p1y, p1z = p1
        p2x, p2y, p2z = p2
        
        # 计算档距
        span = math.sqrt((p2x - p1x)*(p2x - p1x) +
                         (p2y - p1y)*(p2y - p1y) +
                         (p2z - p1z)*(p2z - p1z))
        # 计算高差
        dh = p2z - p1z
        # 计算高角差
        angle = math.atan(dh / span)
        
        # 计算垂直档距
        lv1 = span / 2 - math.sin(angle) / (8 * k) 
        lv2 = span / 2 + math.sin(angle) / (8 * k)
        
        if step < 1e-5:
            step = span / AUTO_POINT_NUM
        
        # 弧垂点数
        num = math.floor(span / step)
        pntsInSpan = []
        pntsOutSpan = []
        
        for i in range(num):
            px, py, pz = (0.0, 0.0, 0.0)  # x, y, z
            xrec = i * step  # 与起点的距离
            x = p1x + (p2x - p1x)*xrec / span
            y = p1y + (p2y - p1y)*xrec / span
            yrec = xrec * math.tan(angle) - 4*k*xrec*(span - xrec)/math.cos(angle)
            # 斜抛公式
            z = p1z + yrec
            pntsInSpan.append((x, y, z))
            
        # 弧垂最低点在起始点左侧
        if lv1 < 0:
            # 计算档距外弧垂最低点坐标
            lowestx = p1x + (p1x - p2x)*math.fabs(lv1) / span
            lowesty = p1y + (p1y - p2y)*math.fabs(lv1) / span
            lowrec = lv1 * math.tan(angle) - 4*k*lv1*(span-lv1) / math.cos(angle)
            lowestz = p1z + lowrec
            
            # 计算档距外分布点坐标
            num = math.floor(math.fabs(lv1) / step)
            for i in range(num):
                px, py, pz = (0.0, 0.0, 0.0)  # x, y, z
                xrec = 0.0 - i*step
                x = p1x + (p1x - p2x) * math.fabs(xrec) / span
                y = p1y + (p1y - p2y) * math.fabs(xrec) / span
                yrec = xrec * math.tan(angle) - 4*k*xrec(span - xrec) / math.cos(angle)
                z = p1z + yrec
                pntsOutSpan.append((x, y, z))
            
            pntsOutSpan.append((lowestx, lowesty, lowestz))
            pntsOutSpan.reverse()
         
        elif lv2 < 0:
            # 计算档距外弧垂最低点坐标
            lowestx = p2x + (p2x - p1x)*math.fabs(lv2) / span
            lowesty = p2y + (p2y - p1y)*math.fabs(lv2) / span
            lowrec = lv1 * math.tan(angle) - 4*k*lv1*(span-lv1) / math.cos(angle)
            lowestz = p1z + lowrec
            
            # 计算档距外分布点坐标
            num = math.floor(math.fabs(lv2) / step)
            for i in range(num):
                px, py, pz = (0.0, 0.0, 0.0)  # x, y, z
                xrec = span + i*step
                x = p2x + (p2x - p1x) * i * step / span
                y = p2y + (p2y - p1y) * i * step / span
                yrec = xrec * math.tan(angle) - 4*k*xrec(span - xrec) / math.cos(angle)
                z = p1z + yrec
                pntsOutSpan.append((x, y, z))
            
            pntsOutSpan.append((lowestx, lowesty, lowestz))
            
        elif lv1 > 0 and lv2 < 0:
            # 弧垂最低点在档距内部
            lowestx = p1x + (p2x - p1x)* lv1 / span
            lowesty = p1y + (p2y - p1y)* lv1 / span
            lowrec = lv1 * math.tan(angle) - 4*k*lv1*(span-lv1) / math.cos(angle)
            lowestz = p1z + lowrec
            
        return pntsInSpan, pntsOutSpan, (lowestx, lowesty, lowestz)
                
    
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
        self.chunks = 31
        # 生成随机高度值
        self.heights = self.np_random.uniform(0, H/2., size=(self.chunks+1, ))
        self.chunk_x = [W / (self.chunks - 1) * i for i in range(self.chunks)]
        self.smooth_y = [0.33*(self.heights[i-1] + self.heights[i+0] + self.heights[i+1]) 
                         for i in range(self.chunks)]
        
        self.terrain = self.world.CreateStaticBody(shapes=b2EdgeShape(vertices=[(0,0), (W, 0)]))
        self.terrain_polys = []
        
        for i in range(self.chunks - 1):
            p1 = (self.chunk_x[i], self.smooth_y[i])
            p2 = (self.chunk_x[i+1], self.smooth_y[i+1])
            # 为地形刚体填充边缘顶点
            self.terrain.CreateEdgeFixture(vertices=[p1,p2],  density=0, friction=0.1)
            # 为地形绘制顶点数据填充顶点
            self.terrain_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        
        self.terrain.color1 = (0.0, 0.0, 0.0)
        self.terrain.color2 = (0.0, 0.0, 0.0)
        
        # 创建杆塔
        body_shape = b2PolygonShape(box = (BODY_W / 2, BODY_H / 2))
        fd = b2FixtureDef(shape = body_shape,
                          friction=0.1,
                          density=5.0,
                          categoryBits=0x0001,
                          maskBits = 0x001,
                          restitution = 0.0)
        
        pos1x = W * 3 / 10
        pos1y = self._calc_heights(self.smooth_y, self.chunk_x, pos1x)
        self.tower1 = self.world.CreateStaticBody(
            position=(pos1x, pos1y + BODY_H / 2),
            fixtures=fd)
        
        self.tower1.color1 = (0.5, 0.4, 0.9)
        self.tower1.color2 = (0.3, 0.3, 0.5)
        
        pos2x = W * 7 / 10
        pos2y = self._calc_heights(self.smooth_y, self.chunk_x, pos2x)
        self.tower2 = self.world.CreateStaticBody(
            position=(pos2x, pos2y + BODY_H / 2),
            fixtures=fd)
        
        self.tower2.color1 = (0.5, 0.4, 0.9)
        self.tower2.color2 = (0.3, 0.3, 0.5)

        
        # 弧垂导线
        WIRE_LENGTH = 10
        pellet_w = 0.05
        pellet_h = 0.05
        num_pellets = int(WIRE_LENGTH / pellet_w )
        pr_w = (pos2x - pos1x) / num_pellets
        pr_h = (pos2y - pos1y) / num_pellets
        
        
        pellet = b2FixtureDef(shape=b2PolygonShape(box=(pellet_w, pellet_h)),
                          friction=0.2,
                          density=5.0,
                          categoryBits=0x0001,
                          maskBits = 0x001)
        
        
        pellets = []
        prevBody = self.tower1
        
        for i in range(num_pellets):
            body = self.world.CreateDynamicBody(
                position=(pos1x + pr_w * i, pos1y + BODY_H + pr_h * i),
                fixtures=pellet
            ) 
            body.color1 = (0.0, 1.0, 0.0)
            body.color2 = (0.0, 1.0, 0.0)     
                
            self.world.CreateRevoluteJoint(
                bodyA=prevBody,
                bodyB=body,
                anchor=(pos1x + pr_w * i, pos1y + BODY_H + pr_h * i)
            )
            
            pellets.append(body)            
            prevBody = body
            
            if(i == num_pellets - 1):
                #   追加一个ground用于绑定最后一块动态块
                self.world.CreateRevoluteJoint(
                    bodyA=body,
                    bodyB=self.tower2,
                    anchor=(pos1x + pr_w * i, pos1y + BODY_H + pr_h * i)
                )
                
        self.drawlist = [self.tower1, self.tower2] + pellets
        self.state = np.array([])
        
        return self.step(np.array([0, 0]))[0]

    def step(self, action):
        
        self.world.Step(1.0/FPS, 6*30, 2*30)
        
        # 横向偏移
        pos = self.tower1.position
        posx = pos.x + 0.1
        posy = self._calc_heights(self.smooth_y, self.chunk_x, posx)
        self.tower1.position = b2Vec2(posx, posy + BODY_H / 2)
        
        state = [
            self.tower1.position.x,
            self.tower2.position.x,
            0.0,
            0.0
        ]
        reward = 0
        done = False
        
        return np.array(state, dtype=np.float32), reward, done, {}
        
    def render(self):
        from gym.envs.classic_control import rendering
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)
        
        for p in self.terrain_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))
            
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        
        return self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
