import numpy as np
from scipy.spatial.transform import Rotation
import math
import json
import keyboard

from macro import Vector2, Vector3, Workspace, Cuboid, Color

def load_cuboids(file_path:str):
    with open(file_path) as f:
        a = json.load(f)
    cuboids:list[Cuboid] = []
    for v in a:
        cuboids.append(Cuboid(
            Vector3(*v['position']), 
            Rotation.from_euler('xyz', v['rotation']), 
            Vector3(*v['scale']), 
            Color(*v['color']) if v['color'] else None
            ))
    return cuboids


workspace = Workspace() 
cuboids = load_cuboids(r'scene.json')

print('press "-" to init')
keyboard.wait('-')
workspace.init()

for cuboid in cuboids:
    print('press "-" to continue')
    keyboard.wait('-')
    workspace.build_block(cuboid)