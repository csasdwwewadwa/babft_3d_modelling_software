import numpy as np
from scipy.spatial.transform import Rotation


class Vector3:
    __slots__ = ("_v",)

    def __init__(self, x, y=None, z=None):
        if y is None and z is None:
            arr = np.asarray(x, dtype=float)
            if arr.shape != (3,):
                raise ValueError("Vector3 requires exactly 3 elements")
            self._v = arr
        else:
            self._v = np.array([x, y, z], dtype=float)

    # ---- basic access ----
    @property
    def x(self): return self._v[0]
    @property
    def y(self): return self._v[1]
    @property
    def z(self): return self._v[2]

    def as_array(self):
        return self._v

    # ---- math ops ----
    def __add__(self, other):
        return Vector3(self._v + other._v)

    def __sub__(self, other):
        return Vector3(self._v - other._v)

    def __mul__(self, scalar):
        return Vector3(self._v * scalar)

    __rmul__ = __mul__

    def dot(self, other):
        return float(np.dot(self._v, other._v))

    def cross(self, other):
        return Vector3(np.cross(self._v, other._v))

    def norm(self):
        return float(np.linalg.norm(self._v))

    def normalized(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        return Vector3(self._v / n)

    # ---- rotation (SciPy-native) ----
    def rotate(self, rotation: Rotation):
        return Vector3(rotation.apply(self._v))

    # ---- numpy interoperability ----
    def __array__(self, dtype=None):
        return np.asarray(self._v, dtype=dtype)

    def __repr__(self):
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

class Camera:

    #    _______   ______
    #   |       | [_____/
    #   |       |]=V*‾‾‾
    #   |_______|    ^
    #       ^        | Camera focal point
    #       | <---->
    #       |offset_z
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


    def __init__(self, pos:Vector3, rot:Rotation, hotkey:str=None):
        self.pos = pos
        self.rot = rot
        self.rot_inv = rot.inv()

        self.hotkey = hotkey

        self.offset_z = 126/41   #  ~3.07    |  111 - (100/(1-5.2/70.8))
        self.screen_z = 31625/41 #  ~771.34  |  506/70.8 * (100/(1-5.2/70.8))
    
    def _local_to_screen(self, local_pos:Vector3) -> Vector3:
        screen_pos = local_pos / (local_pos.z*self.screen_z)
        return screen_pos

    def _world_to_local(self, pos:Vector3) -> Vector3:
        local_pos = self.rot_inv.apply(pos) - self.offset_z
        return local_pos
    
    def to_screen(self, pos:Vector3):
        local_pos = self._world_to_local(pos)
        screen_pos = self._local_to_screen(local_pos)
        return (screen_pos.x, screen_pos.y)