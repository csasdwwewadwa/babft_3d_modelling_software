import pyautogui as pag
from ahk import AHK
ahk = AHK()
ahk.set_coord_mode('Mouse', "Screen")

import time
import math
import numpy as np
from scipy.spatial.transform import Rotation

from typing import Optional, Literal

from core import Vector3, look_at
from config import SCREEN_RESOLUTION


class Camera:

    #    _______   ______
    #   |       | [_____/
    #   |       |]=V*‾‾‾
    #   |_______|    ^
    #       ^        | actual view origin
    #       | <---->
    #       | offset_z
    #       |
    #     origin
    


    #        |\
    #        | \
    #        |__\     <-- measurement
    #     ^  |5.2\
    #     |  |    \
    # 100 |  |     \
    #     |  |      \
    #     v  |_______\     <-- measurement
    #        |  70.8  \
    #        |         \
    #        |          \
    #        |           \
    #        |            \
    #        |             \
    #        |______________\     <-- screen pixel
    #              506


    def __init__(self, position:Vector3, rotation:Rotation, hotkey:str=None):
        self.pos = position
        self.rot = rotation
        self.rot_inv = rotation.inv()

        self.hotkey = hotkey

        self.offset_z = 126/41   #  ~3.07    |  111 - (100/(1-5.2/70.8))
        self.offset = Vector3(0, 0, -self.offset_z)
        self.screen_z = 31625/41 #  ~771.34  |  506/70.8 * (100/(1-5.2/70.8))

    def to_screen(self, position:Vector3):
        local_pos = self._world_to_local(position)
        screen_pos = self._local_to_screen(local_pos)
        return (round(screen_pos.x), round(screen_pos.y))
    
    def _local_to_screen(self, local_pos:Vector3) -> Vector3:
        screen_pos = local_pos * (self.screen_z / local_pos.z)
        screen_pos.y *= -1
        screen_pos += SCREEN_RESOLUTION / 2
        screen_pos.y += 0.5 # offset
        return screen_pos

    def _world_to_local(self, pos:Vector3) -> Vector3:
        local_pos = (pos - self.pos).rotate(self.rot_inv) + self.offset
        return local_pos
    

class Workspace:
    def __init__(self):
        self.workspace_size = Vector3(84, 84, 84)
        self.center = Vector3(0, 42, 0)
        self.block_placement_pos = Vector3(-54, 0, 54)

        self.DEBOUNCE = 0.1 # seconds

        self.cameras:list[Camera] = self._get_cameras()
        self.current_camera:Camera = None
        # for cam in self.cameras:
        #     print(cam.hotkey, cam.pos)

    # ---- the sauce ----
    def place(position:Vector3, rotation:Rotation, scale: Vector3):
        pass

    # ---- actions submacro ----
    def click_world(self, position:Vector3, direction:Optional[Literal['U','D','Up','Down']]=None):
        screen_coord = self.current_camera.to_screen(position)
        ahk.click(*screen_coord, direction=direction)
        time.sleep(self.DEBOUNCE)

    def mouse_move_world(self, position:Vector3):
        screen_coord = self.current_camera.to_screen(position)
        ahk.mouse_move(*screen_coord)
        time.sleep(self.DEBOUNCE)

    def drag_world(self, position_from:Vector3, position_to:Vector3):
        self.click_world(position_from, 'Down')
        self.click_world(position_to, "Up")

    def switch_camera(self, camera:Camera):
        if self.current_camera is None:
            ahk.key_press('m')
            time.sleep(self.DEBOUNCE)

        if camera == self.current_camera:
            return
        
        self.current_camera = camera
        ahk.key_press(camera.hotkey)
        time.sleep(self.DEBOUNCE)

    # ---- init ----
    def _get_cameras(self):
        cameras:list[Camera] = []
        hotkey_seq = self._hotkey_seq()
        base_pos = Vector3(0, 42, -122)
        z_offsets = [0, 4, 18, 28, 18, 4, 0]
        for i in range(7):
            deg = i*15
            rot = Rotation.from_euler('y', -deg, degrees=True)
            pos = base_pos + Vector3(0, 0, -z_offsets[i])
            pos = self.center + (pos - self.center).rotate(rot)
            camera = Camera(pos, rot, next(hotkey_seq))
            cameras.append(camera)
        for i in range(1, 7):
            deg = i*15
            rot = Rotation.from_euler('x', deg, degrees=True)
            pos = base_pos + Vector3(0, 0, -z_offsets[i])
            pos = self.center + (pos - self.center).rotate(rot)
            camera = Camera(pos, rot, next(hotkey_seq))
            cameras.append(camera)
        for i in range(1, 6):
            deg = i*15
            rot = Rotation.from_euler('xy', [deg, -90], degrees=True)
            pos = base_pos + Vector3(0, 0, -z_offsets[i])
            pos = self.center + (pos - self.center).rotate(rot)
            # rot = look_at(pos, self.center)
            camera = Camera(pos, rot, next(hotkey_seq))
            cameras.append(camera)
        return cameras
    
    def _hotkey_seq(self):
        for k in 'qertyupghjklzxcvbnm':
            yield k

if __name__ == '__main__':
    workspace = Workspace()

    print('starting in 3 secs..')
    time.sleep(3)

    for camera in workspace.cameras:
        workspace.switch_camera(camera)
        p = workspace.block_placement_pos
        workspace.mouse_move_world(p)
        time.sleep(1)