import numpy as np
from scipy.spatial.transform import Rotation
import math


class Vector3:
    __slots__ = ("_v",)

    def __init__(self, x, y=None, z=None):
        """
        Initialize with x, y, z or an array/list/tuple of 3 elements.
        Example: Vector3(1, 2, 3) or Vector3([1, 2, 3])
        """
        if y is None and z is None:
            arr = np.asarray(x, dtype=float)
            if arr.shape != (3,):
                raise ValueError("Vector3 requires exactly 3 elements")
            self._v = arr
        else:
            self._v = np.array([x, y, z], dtype=float)

    # ---- basic access ----
    @property
    def x(self) -> float: return float(self._v[0])
    @property
    def y(self) -> float: return float(self._v[1])
    @property
    def z(self) -> float: return float(self._v[2])

    @x.setter
    def x(self, value): self._v[0] = value
    @y.setter
    def y(self, value): self._v[1] = value
    @z.setter
    def z(self, value): self._v[2] = value

    def as_array(self):
        """Returns the internal numpy array (mutable)."""
        return self._v

    # ---- Comparisons ----
    def __eq__(self, other):
        """
        Checks equality. returns True if vectors are approximately equal 
        within standard numpy float tolerance.
        """
        if isinstance(other, Vector3):
            return np.allclose(self._v, other._v)
        # Fallback to handle comparison with raw arrays or lists
        if hasattr(other, '__iter__') and len(other) == 3:
            return np.allclose(self._v, np.asarray(other, dtype=float))
        return NotImplemented

    def approx_eq(self, other: 'Vector3', atol=1e-8):
        """
        Explicit approximate equality with custom tolerance.
        """
        if not isinstance(other, Vector3):
            raise TypeError("approx_eq expects a Vector3")
        return np.allclose(self._v, other._v, atol=atol)

    def __ne__(self, other):
        return not self.__eq__(other)

    # ---- math ops ----
    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self._v + other._v)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self._v - other._v)
        return Vector3(self._v - float(other))

    def __mul__(self, other):
        # Support scalar multiplication
        if isinstance(other, (int, float, np.number)):
            return Vector3(self._v * other)
        # Support element-wise multiplication if passing another vector
        if isinstance(other, Vector3):
            return Vector3(self._v * other._v)
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float, np.number)):
            if scalar == 0:
                raise ZeroDivisionError("division by zero")
            return Vector3(self._v / scalar)
        return NotImplemented
    
    def __rtruediv__(self, other):
        raise NotImplementedError("Cannot divide scalar by Vector3")

    __rmul__ = __mul__

    def __neg__(self):
        """Unary minus (-v)"""
        return Vector3(-self._v)

    def __abs__(self):
        """abs(v) returns the length (norm)"""
        return self.norm()

    # ---- vector products & norms ----
    def dot(self, other: 'Vector3') -> float:
        return float(np.dot(self._v, other._v))

    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(np.cross(self._v, other._v))

    def norm(self) -> float:
        return float(np.linalg.norm(self._v))
    
    def length(self) -> float:
        """Alias for norm()"""
        return self.norm()

    def length_sq(self) -> float:
        """Faster than length() as it avoids sqrt."""
        return float(np.dot(self._v, self._v))

    def normalized(self) -> 'Vector3':
        """Returns a new Vector3 that is the unit vector of self."""
        n = self.norm()
        if n == 0:
            # Depending on use case, might want to return zero vector or error
            raise ZeroDivisionError("cannot normalize zero vector")
        return Vector3(self._v / n)

    # ---- Python protocols ----
    def __iter__(self):
        """Allows unpacking: x, y, z = v"""
        yield from self._v

    def __getitem__(self, index):
        """Allows access via v[0], v[1], v[2]"""
        return self._v[index]

    def __setitem__(self, index, value):
        self._v[index] = value

    def copy(self):
        return Vector3(self._v.copy())

    # ---- rotation (SciPy-native) ----
    def rotated(self, rotation: Rotation):
        """Applies a scipy Rotation object to this vector."""
        return Vector3(rotation.apply(self._v))

    # ---- numpy interoperability ----
    def __array__(self, dtype=None):
        return np.asarray(self._v, dtype=dtype)

    def __repr__(self):
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"
    
    # ---- Static Constructors ----
    @classmethod
    def zero(cls): return cls(0, 0, 0)
    
    @classmethod
    def one(cls): return cls(1, 1, 1)

    @classmethod
    def up(cls): return cls(0, 1, 0)
    
    @classmethod
    def right(cls): return cls(1, 0, 0)
    
    @classmethod
    def forward(cls): return cls(0, 0, 1)

class Vector2:
    __slots__ = ("_v",)

    def __init__(self, x, y=None):
        """
        Initialize with x, y or an array/list/tuple of 2 elements.
        Example: Vector2(1, 2) or Vector2([1, 2])
        """
        if y is None:
            arr = np.asarray(x, dtype=float)
            if arr.shape != (2,):
                raise ValueError("Vector2 requires exactly 2 elements")
            self._v = arr
        else:
            self._v = np.array([x, y], dtype=float)

    # ---- basic access ----
    @property
    def x(self) -> float: return float(self._v[0])
    @property
    def y(self) -> float: return float(self._v[1])

    @x.setter
    def x(self, value): self._v[0] = value
    @y.setter
    def y(self, value): self._v[1] = value

    def as_array(self):
        """Returns the internal numpy array (mutable)."""
        return self._v

    # ---- Comparisons ----
    def __eq__(self, other):
        """
        Checks equality. returns True if vectors are approximately equal 
        within standard numpy float tolerance.
        """
        if isinstance(other, Vector2):
            return np.allclose(self._v, other._v)
        # Fallback to handle comparison with raw arrays or lists
        if hasattr(other, '__iter__') and len(other) == 2:
            return np.allclose(self._v, np.asarray(other, dtype=float))
        return NotImplemented

    def approx_eq(self, other: 'Vector2', atol=1e-8):
        """
        Explicit approximate equality with custom tolerance.
        """
        if not isinstance(other, Vector2):
            raise TypeError("approx_eq expects a Vector2")
        return np.allclose(self._v, other._v, atol=atol)

    def __ne__(self, other):
        return not self.__eq__(other)

    # ---- math ops ----
    def __add__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self._v + other._v)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self._v - other._v)
        return NotImplemented

    def __mul__(self, other):
        # Support scalar multiplication
        if isinstance(other, (int, float, np.number)):
            return Vector2(self._v * other)
        # Support element-wise multiplication if passing another vector
        if isinstance(other, Vector2):
            return Vector2(self._v * other._v)
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float, np.number)):
            if scalar == 0:
                raise ZeroDivisionError("division by zero")
            return Vector2(self._v / scalar)
        return NotImplemented
    
    def __rtruediv__(self, other):
        raise NotImplementedError("Cannot divide scalar by Vector2")

    __rmul__ = __mul__

    def __neg__(self):
        """Unary minus (-v)"""
        return Vector2(-self._v)

    def __abs__(self):
        """abs(v) returns the length (norm)"""
        return self.norm()

    # ---- vector products & norms ----
    def dot(self, other: 'Vector2') -> float:
        return float(np.dot(self._v, other._v))

    def cross(self, other: 'Vector2') -> float:
        """
        In 2D, the cross product is a scalar (the Z component of the 3D cross product).
        Returns: float (x1*y2 - y1*x2)
        """
        return float(np.cross(self._v, other._v))

    def norm(self) -> float:
        return float(np.linalg.norm(self._v))
    
    def length(self) -> float:
        """Alias for norm()"""
        return self.norm()

    def length_sq(self) -> float:
        """Faster than length() as it avoids sqrt."""
        return float(np.dot(self._v, self._v))

    def normalized(self) -> 'Vector2':
        """Returns a new Vector2 that is the unit vector of self."""
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("cannot normalize zero vector")
        return Vector2(self._v / n)

    # ---- Python protocols ----
    def __iter__(self):
        """Allows unpacking: x, y = v"""
        yield from self._v

    def __getitem__(self, index):
        """Allows access via v[0], v[1]"""
        return self._v[index]

    def __setitem__(self, index, value):
        self._v[index] = value

    def copy(self):
        return Vector2(self._v.copy())

    # ---- rotation ----
    def rotated(self, angle_rad: float) -> 'Vector2':
        """
        Rotates the vector by angle_rad (in radians) counter-clockwise.
        Replaces the 3D scipy Rotation method.
        """
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        # Standard 2D rotation matrix:
        # | cos -sin |
        # | sin  cos |
        x_new = self.x * c - self.y * s
        y_new = self.x * s + self.y * c
        return Vector2(x_new, y_new)

    # ---- numpy interoperability ----
    def __array__(self, dtype=None):
        return np.asarray(self._v, dtype=dtype)

    def __repr__(self):
        return f"Vector2({self.x:.4f}, {self.y:.4f})"
    
    # ---- Static Constructors ----
    @classmethod
    def zero(cls): return cls(0, 0)
    
    @classmethod
    def one(cls): return cls(1, 1)

    @classmethod
    def up(cls): return cls(0, 1)
    
    @classmethod
    def right(cls): return cls(1, 0)

class Color:
    def __init__(self, r:int, g:int, b:int):
        self.value = (r, g, b)

    # ---- Comparisons ----
    def __eq__(self, other:'Color'):
        return self.value == other.value
    
    # ---- Python protocols ----
    def __iter__(self):
        """Allows unpacking: r, g, b = v"""
        yield from self.value

    def __getitem__(self, index):
        """Allows access via v[0], v[1], v[2]"""
        return self.value[index]

    def __setitem__(self, index, value):
        self._v[index] = value

class Cuboid:
    '''
    Container for Cuboid data
    '''
    def __init__(self, position:Vector3, rotation:Rotation, scale:Vector3, color:Color):
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.color = color
