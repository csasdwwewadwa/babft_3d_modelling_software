import numpy as np
from scipy.spatial.transform import Rotation
from core import *

def look_at(from_pos: Vector3, to_pos: Vector3, up=Vector3(0, 1, 0)) -> Rotation:
    """
    Creates a SciPy Rotation object representing a rotation matrix that looks 
    from 'from_pos' towards 'to_pos'.
    
    Note: This creates a matrix where Z is the forward vector (from -> to).
    """
    f = np.asarray(to_pos) - np.asarray(from_pos)
    len_f = np.linalg.norm(f)
    if len_f == 0:
         return Rotation.identity() # or raise Error
    f = f / len_f

    u = np.asarray(up)
    u = u / np.linalg.norm(u)

    # right vector (Cross product of Up and Forward)
    r = np.cross(u, f)
    r_norm = np.linalg.norm(r)
    
    if r_norm == 0:
        # This happens if 'up' is parallel to 'forward'.
        # Fallback: choose an arbitrary axis (e.g. X) to cross with.
        r = np.cross(np.array([1, 0, 0]), f)
        r_norm = np.linalg.norm(r)
        if r_norm == 0:
             r = np.cross(np.array([0, 1, 0]), f)
             r_norm = np.linalg.norm(r)
    
    r /= r_norm

    # recompute orthonormal up
    u = np.cross(f, r)

    # rotation matrix (columns = basis vectors [Right, Up, Forward])
    R = np.stack([r, u, f], axis=1)

    return Rotation.from_matrix(R)

def rotation_to_rotate_commands(rotation: Rotation) -> list:
    """
    Converts a scipy Rotation instance into a list of "r" (local Y) and "t" (local X) 
    commands to replicate the orientation.
    """
    epsilon = 1e-6  # Threshold to filter out negligible rotations

    def get_yxy_commands(rot):
        # Decompose into Intrinsic Euler angles: Local Y -> Local X -> Local Y
        y1, x, y2 = rot.as_euler('YXY', degrees=True)
        cmds = []
        
        # Command ("r", a) rotates Y by -a. We want rotation y1, so a = -y1.
        if abs(y1) > epsilon:
            cmds.append(("r", -y1))
            
        # Command ("t", a) rotates X by a. We want rotation x, so a = x.
        if abs(x) > epsilon:
            cmds.append(("t", x))
            
        # Command ("r", a) rotates Y by -a. We want rotation y2, so a = -y2.
        if abs(y2) > epsilon:
            cmds.append(("r", -y2))
            
        return cmds

    def get_xyx_commands(rot):
        # Decompose into Intrinsic Euler angles: Local X -> Local Y -> Local X
        x1, y, x2 = rot.as_euler('XYX', degrees=True)
        cmds = []
        
        # Command ("t", a) rotates X by a.
        if abs(x1) > epsilon:
            cmds.append(("t", x1))
            
        # Command ("r", a) rotates Y by -a. We want rotation y, so a = -y.
        if abs(y) > epsilon:
            cmds.append(("r", -y))
            
        # Command ("t", a) rotates X by a.
        if abs(x2) > epsilon:
            cmds.append(("t", x2))
            
        return cmds

    # Calculate both possibilities
    commands_yxy = get_yxy_commands(rotation)
    commands_xyx = get_xyx_commands(rotation)

    # Return the sequence with fewer commands to be "as fast as possible"
    if len(commands_yxy) <= len(commands_xyx):
        commands = commands_yxy
    else:
        commands = commands_xyx
    
    if commands:
        commands = [(c, v%360) for c, v in commands]
        commands[-1] = (commands[-1][0], commands[-1][1] % 90)
    
    return commands