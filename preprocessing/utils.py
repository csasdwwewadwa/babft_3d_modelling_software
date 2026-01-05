import math
from itertools import product
import numpy as np
# from ..macro.core import Vector2


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




import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_shapes(polygon_vertices, rectangles, other_polygon_vertices=None):
    """
    Visualizes a convex polygon and a list of rotated rectangles.
    
    :param polygon_vertices: List of (x, y) tuples/array for the main polygon.
    :param rectangles: List of dicts: {'center': (x, y), 'size': (w, h), 'rotation': degrees}
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Draw the Convex Polygon
    # We use a Polygon patch. 'closed=True' ensures the last point connects to the first.
    poly_patch = patches.Polygon(polygon_vertices, closed=True, 
                                 linewidth=2, edgecolor='blue', 
                                 facecolor='blue', alpha=0.3, label='Convex Polygon')
    ax.add_patch(poly_patch)

      

    # 2. Draw the Rectangles
    for i, rect in enumerate(rectangles):
        cx, cy = rect['pos']
        w, h = rect['size']
        angle = rect['rot']
        
        # Define the 4 corners of the rectangle relative to the center (0,0)
        # Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left
        corners = np.array([
            [-w/2, -h/2],
            [ w/2, -h/2],
            [ w/2,  h/2],
            [-w/2,  h/2]
        ])
        
        # Rotation Matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        
        # Rotate corners and then translate to the center (cx, cy)
        rotated_corners = np.dot(corners, rotation_matrix.T) + [cx, cy]
        
        # Create a polygon patch for the rectangle
        rect_patch = patches.Polygon(rotated_corners, closed=True, 
                                     linewidth=1.5, edgecolor='red', 
                                     facecolor='red', alpha=0.5, 
                                     label=f'Rect {i+1}' if i == 0 else "")
        ax.add_patch(rect_patch)

    if other_polygon_vertices:
        other_poly_patch = patches.Polygon(other_polygon_vertices, closed=True, 
                                 linewidth=2, edgecolor='yellow', 
                                 facecolor='yellow', alpha=0.3, label='more Convex Polygon')
        ax.add_patch(other_poly_patch)

    # Auto-scale the plot limits
    all_points = np.vstack([polygon_vertices] + [
        np.dot(np.array([[-w/2, -h/2], [w/2, h/2]]), np.array([[np.cos((r['rot'])), -np.sin((r['rot']))], [np.sin((r['rot'])), np.cos((r['rot']))]]).T) + r['pos'] 
        for r in rectangles
    ])
    
    # Set plot aesthetics
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale_view()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.title("Polygon and Rotated Rectangles Visualization")
    plt.show()


def intersect_lines(from_1:Vector2, to_1:Vector2, from_2:Vector2, to_2:Vector2, eps=1e-9):
    """
    Intersection of two infinite lines.
    Returns Vector2 or None.
    """
    r = to_1 - from_1
    s = to_2 - from_2

    denom = r.cross(s)
    if abs(denom) < eps:
        return None  # Parallel or coincident
    
    t = (from_2 - from_1).cross(s) / denom
    return from_1 + r * t

def intersect_segments(from_1:Vector2, to_1:Vector2, from_2:Vector2, to_2:Vector2, eps=1e-9):
    """
    Intersection of two line segments.
    Returns Vector2 or None.
    """

    r = to_1 - from_1
    s = to_2 - from_2

    denom = r.cross(s)
    if abs(denom) < eps:
        return None  # Parallel, collinear, or overlapping

    qp = from_2 - from_1
    t = qp.cross(s) / denom
    u = qp.cross(r) / denom

    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        return from_1 + r * t

    return None

def intersect_ray_to_segment(ray_origin: Vector2, ray_direction: Vector2, seg_start: Vector2, seg_end: Vector2, eps=1e-9):
    """
    Intersection of an infinite ray and a finite line segment.
    ray_direction does not need to be normalized.
    Returns Vector2 or None.
    """
    # r = ray direction
    # s = segment vector
    r = ray_direction
    s = seg_end - seg_start

    denom = r.cross(s)
    if abs(denom) < eps:
        # Ray and segment are parallel
        return None

    qp = seg_start - ray_origin
    
    # t is the parameter for the ray: ray_origin + r * t
    # u is the parameter for the segment: seg_start + s * u
    t = qp.cross(s) / denom
    u = qp.cross(r) / denom

    # For a ray, t must be >= 0 (it extends infinitely in one direction)
    # For a segment, u must be between 0 and 1
    if t >= -eps and -eps <= u <= 1 + eps:
        return ray_origin + r * t

    return None


def angle_between_three(v1:Vector2, v2:Vector2, v3:Vector2, eps=1e-9):
    """
    Returns the angle (in radians) formed by v1 - v2 - v3.
    The angle is measured at v2.
    """

    a = v1 - v2
    b = v3 - v2

    dot = a.dot(b)
    mag_a = a.length()
    mag_b = b.length()

    if mag_a < eps or mag_b < eps:
        return 0.0  # Degenerate case

    # Clamp to avoid floating-point drift
    cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.acos(cos_theta)

def look_rotation(from_vec: Vector2, to_vec: Vector2) -> float:
    direction = to_vec - from_vec
    return math.atan2(direction.y, direction.x)


    
class Transform:
    def __init__(self, translation: Vector2, rotation: float):
        self.translation = translation
        self.rotation = rotation
        
    def apply(self, vector: Vector2):
        return (vector + self.translation).rotated(self.rotation)
        
    def undo(self, vector: Vector2):
        return vector.rotated(-self.rotation) - self.translation
        
    def __add__(self, other: 'Transform'):
        new_translation = self.translation + other.translation.rotated(-self.rotation)
        new_rotation = self.rotation + other.rotation
        return Transform(new_translation, new_rotation)
    

def fill_polygon(vertices:list[Vector2], min_length:float=0.05, offset:float=0.01):
    n = len(vertices)
    outer = vertices.copy()
    inner = vertices.copy()
    trans_cum = Transform(Vector2.zero(), 0)
    rect_list = []

    # outer border with min_length width
    for i in range(n):
        # indices
        next_i = (i+1) % n
        next_i2 = (i+2) % n
        prev_i = (i-1) % n
        # to local
        translation = -outer[i]
        rotation = -look_rotation(outer[i], outer[next_i])
        trans = Transform(translation, rotation)
        trans_cum += trans
        for j in range(n):
            outer[j] = trans.apply(outer[j])
            inner[j] = trans.apply(inner[j])
        # generate rect
        prev_ang = angle_between_three(outer[next_i], outer[i], outer[prev_i])
        next_ang = angle_between_three(outer[i], outer[next_i], outer[next_i2])
        
        rect_x_from = max(outer[i].x, (min_length + offset) / math.tan(prev_ang))
        rect_x_to = min(outer[next_i].x, outer[next_i].x - (min_length + offset) / math.tan(next_ang))

        rect_pos = Vector2((rect_x_from+rect_x_to)/2, -min_length/2)
        rect_pos = trans_cum.undo(rect_pos)
        rect_rot = -trans_cum.rotation
        rect_size = Vector2(rect_x_to-rect_x_from, min_length)

        rect_list.append({
            'pos': rect_pos.copy(),
            'rot': rect_rot,
            'size': rect_size.copy()
        })
        # update inner
        _inner_i = intersect_lines(
            inner[i] + Vector2(0, -min_length),
            inner[next_i] + Vector2(0, -min_length),
            inner[i],
            inner[prev_i]
        )
        _inner_next_i = intersect_lines(
            inner[i] + Vector2(0, -min_length),
            inner[next_i] + Vector2(0, -min_length),
            inner[next_i],
            inner[next_i2]
        )
        inner[i] = _inner_i
        inner[next_i] = _inner_next_i
        

    # infill
    next_inner = []
    done = False
    for _ in range(20):
        for i in range(n):
            next_i = (i+1) % n
            next_i2 = (i+2) % n
            prev_i = (i-1) % n
            translation = -inner[i]
            rotation = -look_rotation(inner[i], inner[next_i])
            trans = Transform(translation, rotation)
            trans_cum += trans
            for j in range(n):
                outer[j] = trans.apply(outer[j])
                inner[j] = trans.apply(inner[j])
            
            rect_x_from = inner[i].x - offset
            rect_x_to = inner[next_i].x + offset
            rect_y_from = offset
            rect_y_to = max(
                res.y for res in (
                    intersect_ray_to_segment(Vector2(x, 0), Vector2(0, -1), outer[j], outer[(j+1)%n])
                    for x, j in product([rect_x_from, rect_x_to], range(n))
                ) if res is not None
            ) + offset
            
            rect_pos = Vector2((rect_x_from+rect_x_to)/2, (rect_y_from+rect_y_to)/2)
            rect_pos = trans_cum.undo(rect_pos)
            rect_rot = -trans_cum.rotation
            rect_size = Vector2(rect_x_to-rect_x_from, rect_y_to-rect_y_from)
            
            is_update_inner_i = angle_between_three(inner[prev_i], inner[i], inner[next_i]) < math.pi/2
            is_update_inner_next_i = angle_between_three(inner[i], inner[next_i], inner[next_i2]) < math.pi/2

            # do not add rect if no inner is updated
            if is_update_inner_i or is_update_inner_next_i:
                rect_list.append({
                    'pos': rect_pos.copy(),
                    'rot': rect_rot,
                    'size': rect_size.copy()
                })

            # update inner
            _inner_i = None
            _inner_next_i = None

            if is_update_inner_i:
                _inner_i = intersect_lines(
                    inner[i] + Vector2(0, rect_y_to),
                    inner[next_i] + Vector2(0, rect_y_to),
                    inner[i],
                    inner[prev_i]
                )
            # visualize_shapes(vertices, rect_list, [trans_cum.undo(v) for v in inner])
                
            if is_update_inner_next_i:
                _inner_next_i = intersect_lines(
                    inner[i] + Vector2(0, rect_y_to),
                    inner[next_i] + Vector2(0, rect_y_to),
                    inner[next_i],
                    inner[next_i2]
                )
                check_ray_hit = None
                for j in range(n):
                    yes = intersect_ray_to_segment(Vector2(_inner_next_i.x, 99), Vector2(0, -1), inner[j], inner[(j+1)%n])
                    if yes:
                        check_ray_hit = yes
                        break
                if check_ray_hit.y < rect_y_to:
                    _inner_next_i = intersect_lines(
                        inner[next_i],
                        inner[next_i] + Vector2(0, rect_y_to),
                        inner[next_i],
                        inner[next_i2]
                    )
                    
            if _inner_i:
                inner[i] = _inner_i
            if _inner_next_i:
                inner[next_i] = _inner_next_i

            visualize_shapes(vertices, rect_list, [trans_cum.undo(v) for v in inner])

            # check if done
            if rect_y_to < min(v.y for v in inner):
                done = True
                break
                
        if done:
            break
    
    else:
        raise Exception('too many iterations! might be infinite loop or that the input polygon was too big.')

    return rect_list



vertices = [
    Vector2(0, 0.5),
    Vector2(1.5, 1),
    Vector2(1, 0),
    Vector2(0.9, 0)
]
rect_list = fill_polygon(vertices)
print(len(rect_list))

visualize_shapes(vertices, rect_list)
