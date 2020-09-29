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
        world2 = env.World()
        # 创建地形
        terrain = world.CreateTerrainBody(default_dem, default_dom)
        terrain2 = world2.CreateTerrainBody(default_dem, default_dom)
        
        
        os.system("pause")
      
        
    except Exception as e:
        import traceback
        traceback.print_exc() 
        
if __name__ == "__main__":
    main()