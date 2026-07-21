import numpy as np
from scipy.spatial.transform import Rotation
import math
import json
import keyboard
from tqdm import tqdm

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
    cuboids.sort(key=lambda v: v.position.x)
    cuboids.sort(key=lambda v: v.position.z)
    cuboids.sort(key=lambda v: v.position.y)
    return cuboids


stop_program = False

def stop_trigger():
    global stop_program
    print("\n[Stop signal received] Finishing up or stopping before the next block...")
    stop_program = True

# This listens in the background. As soon as "=" is pressed, it calls stop_trigger
keyboard.add_hotkey('f7', stop_trigger)



workspace = Workspace() 
cuboids = load_cuboids(r'files/cirno.json')

progress = -1
with open('progress.txt') as f:
    progress = int(f.read())

print(f'press "f6" to init and start, f7 to stop. progress: {progress}')
keyboard.wait('f6')
workspace.init()


for i in tqdm(range(progress+1, len(cuboids)), initial=progress+1, total=len(cuboids)):
    cuboid = cuboids[i]
 
    # Check if the stop flag was set before starting the next block
    if stop_program:
        print("Program stopped by user.")
        break

    # print('press "-" to continue')
    # keyboard.wait('-')
    workspace.build_block(cuboid)

    with open('progress.txt', 'w') as f:
        f.write(str(i))