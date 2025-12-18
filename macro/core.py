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
    def x(self) -> float: return self._v[0]
    @property
    def y(self) -> float: return self._v[1]
    @property
    def z(self) -> float: return self._v[2]


    @x.setter
    def x(self, value):
        self._v[0] = float(value)

    @y.setter
    def y(self, value):
        self._v[1] = float(value)

    @z.setter
    def z(self, value):
        self._v[2] = float(value)


    def as_array(self):
        return self._v

    # ---- math ops ----
    def __add__(self, other):
        return Vector3(self._v + other._v)

    def __sub__(self, other):
        return Vector3(self._v - other._v)

    def __mul__(self, scalar):
        return Vector3(self._v * scalar)
    
    def __truediv__(self, scalar):
        if scalar == 0:
            raise ZeroDivisionError("division by zero")
        return Vector3(self._v / scalar)
    
    def __rtruediv__(self, other):
        raise NotImplementedError

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
            raise ZeroDivisionError("cannot normalize zero vector")
        return Vector3(self._v / n)

    # ---- rotation (SciPy-native) ----
    def rotate(self, rotation: Rotation):
        return Vector3(rotation.apply(self._v))

    # ---- numpy interoperability ----
    def __array__(self, dtype=None):
        return np.asarray(self._v, dtype=dtype)

    def __repr__(self):
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"
    

def look_at(from_pos, to_pos, up=Vector3(0, 1, 0)):
    f = np.asarray(to_pos) - np.asarray(from_pos)
    f = f / np.linalg.norm(f)

    u = np.asarray(up)
    u = u / np.linalg.norm(u)

    # right vector
    r = np.cross(u, f)
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        raise ValueError("up vector is parallel to view direction")
    r /= r_norm

    # recompute orthonormal up
    u = np.cross(f, r)

    # rotation matrix (columns = basis vectors)
    R = np.stack([r, u, f], axis=1)

    return Rotation.from_matrix(R)