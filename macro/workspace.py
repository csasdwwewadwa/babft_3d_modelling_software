import numpy as np
from scipy.spatial.transform import Rotation

from core import Vector3, Camera


class Workspace:
    def __init__(self):
        self.cameras:list[Camera] = []
        for angle in range(0, 90, 15):
            rad = np.deg2rad(angle)
            pos = Vector3(0, 42, -122)