import numpy as np
from scipy.spatial.transform import Rotation

from core import Vector3, Camera


class Workspace:
    def __init__(self):
        self.workspace_size = Vector3(84, 84, 84)
        self.center = Vector3(0, 42, 0)

        self.cameras:list[Camera] = self._get_cameras()

    def _get_cameras(self):
        cameras:list[Camera] = []
        hotkey_seq = self._hotkey_seq()
        base_pos = Vector3(0, 42, -122)
        for deg in range(0, 91, 15):
            rot = Rotation.from_euler('y', -deg, degrees=True)
            pos = self.center + (base_pos - self.center).rotate(rot)
            camera = Camera(pos, rot, next(hotkey_seq))
            cameras.append(camera)
        return cameras
    
    def _hotkey_seq(self):
        for k in 'qertyuiopghjklzxcvbnm':
            yield k

if __name__ == '__main__':
    workspace = Workspace()
