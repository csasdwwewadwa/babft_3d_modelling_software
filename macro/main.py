import pyautogui as pag
from ahk import AHK
ahk = AHK()
ahk.set_coord_mode('Mouse', "Screen")

import time
import math
import enum
import keyboard
import random

import numpy as np
from scipy.spatial.transform import Rotation

from typing import Optional, Literal

from core import Vector3, Color
from helper import *
from config import SCREEN_RESOLUTION

INF = float('inf')


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

    def to_screen(self, position:Vector3) -> Vector2:
        local_pos = self._world_to_local(position)
        screen_pos = self._local_to_screen(local_pos)
        return Vector2(screen_pos.x, screen_pos.y)
    
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
        # system constants
        self.workspace_size = Vector3(84, 84, 84)
        self.center = Vector3(0, 42, 0)

        # inits
        self.cameras:list[Camera] = self._get_cameras()
        self.current_camera:Camera = None
        # self.current_color = None # TODO

        # configs
        self.DEBOUNCE = 0.05 # seconds

        # more constants
        self.block_placement_pos = Vector3(-54, 0, 54)
        self.camera_top_down:Camera = next((cam for cam in self.cameras if cam.pos == Vector3(30, 164, 0)))
        self.camera_left_right:Camera = next((cam for cam in self.cameras if cam.pos == Vector3(122, 42, 30)))


    # ---- the sauce ----
    def build_block(self, position:Vector3, rotation:Rotation, scale: Vector3, color:Color=None):
        # --- delete block at block placement pos if there is
        #region
        self.switch_camera(self.camera_top_down)
        if pag.pixel(*(map(round, self.to_screen(self.block_placement_pos + Vector3(0, 1, 0))))) != (4, 4, 4):
            self.switch_tool('Delete')
            self.click_world(self.block_placement_pos + Vector3(0, 1, 0))
            print('WARNING: Exists remnant block in block_placement_pos')
        #endregion

        # --- rotate and place block at block placement pos
        #region
        # select block plastic
        self.switch_tool('Build')
        if pag.pixel(172, 257) != (239, 239, 239):
            ahk.click(172, 257)
            time.sleep(0.2)
            ahk.key_down('ctrl')
            time.sleep(self.DEBOUNCE)
            ahk.key_press('a')
            time.sleep(self.DEBOUNCE)
            ahk.key_up('ctrl')
            time.sleep(self.DEBOUNCE)
            ahk.type('plastic')
            time.sleep(self.DEBOUNCE)
            ahk.key_press('enter')
            time.sleep(0.2)
            ahk.click(50, 369)
        # open advanced panel
        if pag.pixel(518, 268) != (239, 239, 239):
            ahk.click(518, 268)
            time.sleep(0.5)
        rot_cmd_seq = rotation_to_rotate_commands(rotation)
        for k, v in rot_cmd_seq:
            if v >= 90:
                self._set_build_tool_rotate_value(v)
                while v >= 90:
                    v -= 90
                    ahk.key_press(k)
                    time.sleep(self.DEBOUNCE)
            if v != 0:
                self._set_build_tool_rotate_value(v)
                ahk.key_press(k)
                time.sleep(self.DEBOUNCE)
        self.switch_camera(self.camera_top_down)
        self.click_world(self.block_placement_pos)
        #endregion

        # --- color block if set
        #region
        self.switch_camera(self.camera_top_down)
        if color is not None:
            self.switch_tool('Color')
            if pag.pixel(234, 461) != (97, 97, 97):
                ahk.click(336, 794)
                time.sleep(0.5)
            for i, v in enumerate(color):
                ahk.click(46, 676 + 61*i)
                time.sleep(0.2)
                ahk.key_down('ctrl')
                time.sleep(self.DEBOUNCE)
                ahk.key_press('a')
                time.sleep(self.DEBOUNCE)
                ahk.key_up('ctrl')
                time.sleep(self.DEBOUNCE)
                ahk.type(str(v))
                time.sleep(self.DEBOUNCE)
                ahk.key_press('enter')
                time.sleep(self.DEBOUNCE)
            self.click_world(self.block_placement_pos + Vector3(0, 1, 0))
        #endregion

        # --- move and scale block to designated position
        #region
        # init block position value
        current_block_position = self.block_placement_pos
        current_block_scale = Vector3(2, 2, 2)
        rot_matrix = rotation.as_matrix()
        current_block_position.y = np.sum(np.abs(rot_matrix[1, :]))
        # select block with scale tool
        self.switch_tool('Scale')
        self.click_world(current_block_position)
        # compute knobs position
        def _to_local(pos:Vector3):
            return (pos - current_block_position).rotate(rotation.inv())
        def _to_global(pos:Vector3):
            return pos.rotate(rotation) + current_block_position
        # for each knob, find the best camera such that the knob on screen is furthest from the nearest knob on screen
        # scale front knobs
        local_target_position = _to_local(position)
        for axis_i in range(3):
            # front knobs are to be scaled first, then to back knobs.
            local_knobs_position = [Vector3.zero() for _ in range(6)]
            for i in range(3):
                # first 3 indices: front knobs
                local_knobs_position[i][i] = math.copysign(current_block_scale[i]+2, local_target_position[i])
                # last 3 indices: back knobs
                local_knobs_position[i+3][i] = -math.copysign(current_block_scale[i]+2, local_target_position[i])
            global_knobs_position = [_to_global(v) for v in local_knobs_position]
            best_camera = None
            best_distance_sq = 0
            for camera in self.cameras:
                screen_knobs_position = [camera.to_screen(global_knobs_position[i]) for i in range(6)]
                d = min((screen_knobs_position[i] - screen_knobs_position[axis_i]).length_sq() for i in range(6) if i != axis_i)
                if d > best_distance_sq:
                    best_distance_sq = d
                    best_camera = camera
            # scale block and update saved block position and scale
            self.switch_camera(best_camera)
            print(local_target_position)
            offset_from = 7 # px
            offset_to = 12  # px
            scale_amount = abs(local_target_position[axis_i]) - current_block_scale[axis_i]/2 + scale[axis_i]/2 # TODO the scale amount was right, idk why it fucks up though
            _temp_vector = Vector3.zero()
            _temp_vector[axis_i] = math.copysign(scale_amount, local_target_position[axis_i])
            screen_from = self.to_screen(global_knobs_position[axis_i])
            screen_to = self.to_screen(_to_global(local_knobs_position[axis_i] + _temp_vector))
            screen_from += (screen_to - screen_from).normalized() * offset_from
            screen_to += (screen_to - screen_from).normalized() * offset_to

            current_block_scale[axis_i] += scale_amount
            current_block_position = _to_global(_to_local(current_block_position) + local_knobs_position[axis_i].normalized()*scale_amount/2)

            # print(current_block_position, current_block_scale)
            # keyboard.wait('-')

            # change scale tool scale
            if pag.pixel(37, 450) != (239, 239, 239):
                ahk.click(37, 450)
                time.sleep(0.5)
            ahk.click(124, 453)
            time.sleep(self.DEBOUNCE)
            ahk.type(str(scale_amount))
            time.sleep(self.DEBOUNCE)
            ahk.key_press('enter')
            time.sleep(self.DEBOUNCE)
            ahk.mouse_drag(*screen_to, from_position=screen_from)
            time.sleep(self.DEBOUNCE)
        #endregion

    # ---- actions submacro ----
    def click_world(self, position:Vector3, direction:Optional[Literal['U','D','Up','Down']]=None):
        screen_coord = self.to_screen(position)
        ahk.click(*screen_coord, direction=direction)
        time.sleep(self.DEBOUNCE)

    def mouse_move_world(self, position:Vector3):
        screen_coord = self.to_screen(position)
        ahk.mouse_move(*screen_coord)
        time.sleep(self.DEBOUNCE)

    def drag_world(self, position_from:Vector3, position_to:Vector3):
        self.click_world(position_from, 'Down')
        self.click_world(position_to, "Up")

    def to_screen(self, position:Vector3) -> Vector2:
        if self.current_camera is None:
            raise Exception("current_camera is not set.")
        screen_coord = self.current_camera.to_screen(position)
        return screen_coord

    def switch_camera(self, camera:Camera):
        if self.current_camera is None:
            ahk.key_press('m') # change to portal cam
            time.sleep(self.DEBOUNCE)
        if self.current_camera and self.current_camera.hotkey == camera.hotkey:
            ahk.key_press('m') # change to portal cam
            time.sleep(self.DEBOUNCE)
        
        self.current_camera = camera
        ahk.key_press(camera.hotkey)
        time.sleep(self.DEBOUNCE)
    
    def switch_tool(self, tool:Literal['Delete', 'Build', 'Color', 'Bind', 'Scale', 'Screwdriver', 'Trowel']):
        tool_idx = ['Delete', 'Build', 'Color', 'Bind', 'Scale', 'Screwdriver', 'Trowel'].index(tool)
        is_equipped = pag.pixel(738+65*tool_idx, 1071) == (90, 142, 233)
        # unequip if is currently equipped (this also resets the rotation on the build tool)
        if is_equipped:
            ahk.key_press(str(tool_idx+1))
            time.sleep(self.DEBOUNCE)
        # equip
        ahk.key_press(str(tool_idx+1))
        time.sleep(self.DEBOUNCE)

    def _set_build_tool_rotate_value(self, value:float):
        value = f'{value:.2f}'
        ahk.click(493, 621)
        time.sleep(self.DEBOUNCE)
        ahk.type(value)
        time.sleep(self.DEBOUNCE)
        ahk.key_press('enter')

    # ---- init ----
    def _get_cameras(self):
        cameras:list[Camera] = []
        hotkey_seq = self._hotkey_seq()
        base_pos = Vector3(30, 42, -122)
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
    # input('closed chat? press Enter to start.')
    print('starting in 3 secs..')
    time.sleep(3)

    workspace.build_block(Vector3(0, 0, 0), Rotation.from_euler('xyz', (123, 456, 789)), Vector3(1, 1, 1), color=None)
    # workspace.build_block(Vector3(0, 0, 0), Rotation.random(), Vector3(1, 1, 1), color=None)

    # a = [
    #     Vector3(-42, 0, -42),
    #     Vector3(-42, 0, 42),
    #     Vector3(42, 0, 42),
    #     Vector3(42, 0, -42),
    # ]
    # for camera in workspace.cameras:
    #     workspace.switch_camera(camera)
    #     for v in a:
    #         workspace.mouse_move_world(v)
    #     time.sleep(0.2)