import numpy as np
from scipy.spatial.transform import Rotation as R


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
    def rotate(self, rotation: R):
        return Vector3(rotation.apply(self._v))

    # ---- numpy interoperability ----
    def __array__(self, dtype=None):
        return np.asarray(self._v, dtype=dtype)

    def __repr__(self):
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


class Camera:
    '''
    ```
    ._______   ______
    |       | [_____/
    |       |]=V*‾‾‾
    |_______|

        ^        ^
        |        | Camera focal point
        | Camera origin ('pos')
    ```
    '''
    def __init__(self, pos, rot):