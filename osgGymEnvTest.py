import os
import time
import random
import osgGymEnv as env

default_dem = "C:\\Users\\lvwei\\Desktop\\export\\dem.tif"
default_dom = "C:\\Users\\lvwei\\Desktop\\export\\dom2.tif"
default_tower = "C:\\Users\\lvwei\\Desktop\\export\\tower.obj"
    
def main():
    try:
        # 创建场景环境
        world = env.World()
        
        # 创建地形
        terrain = world.CreateTerrainBody(default_dem, default_dom)
        # os.system("pause")
         
        # 获得地形范围
        extent = terrain.extent
        
        # 创建杆塔及弧垂
        # 杆塔1
        x = random.uniform(extent.min_x, extent.max_x)
        y = random.uniform(extent.min_y, extent.max_y)
        tower1 = world.CreateTowerBody(x, y, default_tower)
        
        # 杆塔2
        x = random.uniform(extent.min_x, extent.max_x )
        y = random.uniform(extent.min_y, extent.max_y)
        tower2 = world.CreateTowerBody(x, y, default_tower)
        
        # 创建弧垂
        arcline = world.CreateArclineBody(tower1, tower2, 7e-5, 5)
        
        os.system("pause")
        
        # 渲染视图
        viewer = world.GetViewer()
       
        for i in range(50):
            tower1.x += 10
            viewer.DrawTower(tower1)
            
            world.UpdateArcline(arcline)
            viewer.DrawArcline(arcline)
            endPnt = env.Point3D()
            
            minDistance = world.CalcLowestDistance(arcline, endPnt)
            startPnt = env.Point3DCopy(endPnt)
            startPnt.z -= minDistance
            viewer.DrawReferenceLine(startPnt, endPnt)
                    
        os.system("pause")
        
        world.DeleteArclineBody(arcline)
        world.DeleteTowerBody(tower1)
        world.DeleteTowerBody(tower2)
        # world.DeleteTerrainMesh(1)
        os.system("pause")
        
        
    except Exception as e:
        import traceback
        traceback.print_exc() 
        
if __name__ == "__main__":
    main()