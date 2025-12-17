import numpy as np
from scipy.spatial.transform import Rotation

from core import Vector3, Camera


class Workspace:
    def __init__(self):
        self.cameras:list[Camera] = []
        for angle in range(0, 90, 15):
            np.deg2rad(angle)
            