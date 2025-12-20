import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from helper import *

def get_cuboid_vertices(center, sizes):
    """
    Generate the 8 vertices of a cuboid.
    sizes = (width_x, height_y, depth_z)
    """
    cx, cy, cz = center
    wx, hy, dz = sizes 
    
    # Half-dimensions
    hw, hh, hd = wx / 2, hy / 2, dz / 2
    
    # Define vertices in the local (x, y, z) space
    # 0-3: Bottom face (y is low), 4-7: Top face (y is high)
    vertices = np.array([
        [cx - hw, cy - hh, cz - hd], # 0: Left-Bottom-Back
        [cx + hw, cy - hh, cz - hd], # 1: Right-Bottom-Back
        [cx + hw, cy - hh, cz + hd], # 2: Right-Bottom-Front
        [cx - hw, cy - hh, cz + hd], # 3: Left-Bottom-Front
        [cx - hw, cy + hh, cz - hd], # 4: Left-Top-Back
        [cx + hw, cy + hh, cz - hd], # 5: Right-Top-Back
        [cx + hw, cy + hh, cz + hd], # 6: Right-Top-Front
        [cx - hw, cy + hh, cz + hd]  # 7: Left-Top-Front
    ])
    return vertices

def map_to_plot_coords(vertices):
    """
    Maps logical (x, y, z) to Matplotlib plotting (x_plot, y_plot, z_plot).
    Logic: Y is Up, Z is Forward.
    Matplotlib: Z is Up, Y is Depth.
    
    Mapping:
    User X -> Plot X
    User Y -> Plot Z (Height)
    User Z -> Plot Y (Depth)
    """
    # Create a copy to ensure we don't modify original data
    v_plot = np.zeros_like(vertices)
    v_plot[:, 0] = vertices[:, 0] # X -> X
    v_plot[:, 1] = vertices[:, 2] # Z -> Y (Plot Depth)
    v_plot[:, 2] = vertices[:, 1] # Y -> Z (Plot Height)
    return v_plot

def plot_rotated_cuboid(rotation_obj, center=(0,0,0), sizes=(1,2,1)):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Generate Vertices in Local Space
    vertices = get_cuboid_vertices(center, sizes)

    # 2. Apply Rotation (Standard Scipy logic on x,y,z)
    vertices_centered = vertices - np.array(center)
    rotated_vertices_centered = rotation_obj.apply(vertices_centered)
    rotated_vertices = rotated_vertices_centered + np.array(center)

    # 3. Map to Matplotlib Coordinate System (Y-Up logic)
    plot_vertices = map_to_plot_coords(rotated_vertices)

    # 4. Define Faces
    # Indices based on the vertex order in get_cuboid_vertices
    faces_indices = [
        [0, 1, 2, 3], # Bottom (y-)
        [4, 5, 6, 7], # Top (y+)
        [0, 1, 5, 4], # Back (z-)
        [2, 3, 7, 6], # Front (z+)
        [1, 2, 6, 5], # Right (x+)
        [4, 7, 3, 0]  # Left (x-)
    ]
    
    poly_faces = [[plot_vertices[i] for i in face] for face in faces_indices]

    # 5. Plot Cuboid
    cuboid_poly = Poly3DCollection(poly_faces, alpha=0.6, linewidths=1, edgecolors='k')
    # Color faces: Top/Bottom=Cyan, Front/Back=Magenta, Left/Right=Yellow
    cuboid_poly.set_facecolor(['cyan', 'cyan', 'magenta', 'magenta', 'yellow', 'yellow'])
    ax.add_collection3d(cuboid_poly)

    # 6. Plot Local Axes Arrows (visualize orientation)
    origin = np.array(center)
    axis_vectors = np.eye(3) # Unit vectors [1,0,0], [0,1,0], [0,0,1]
    rotated_axes = rotation_obj.apply(axis_vectors)
    
    # Map origins and vectors to plot coordinates
    origin_plot = map_to_plot_coords(origin.reshape(1,3)).flatten()
    vectors_plot = map_to_plot_coords(rotated_axes)

    scale = max(sizes) * 0.8
    colors = ['r', 'g', 'b'] 
    labels = ['x', 'y (up)', 'z (fwd)']
    
    for i in range(3):
        ax.quiver(origin_plot[0], origin_plot[1], origin_plot[2], 
                  vectors_plot[i,0], vectors_plot[i,1], vectors_plot[i,2], 
                  length=scale, color=colors[i], linewidth=2)
        
        # Label position
        lab_pos = origin_plot + vectors_plot[i] * scale
        ax.text(lab_pos[0], lab_pos[1], lab_pos[2], labels[i], color=colors[i], fontsize=11, fontweight='bold')

    # 7. Setup Axes Limits and Labels
    # Use the plot_vertices to determine limits
    all_x = plot_vertices[:, 0]
    all_y = plot_vertices[:, 1] # This is actually Z data
    all_z = plot_vertices[:, 2] # This is actually Y data

    max_range = np.array([
        all_x.max()-all_x.min(), 
        all_y.max()-all_y.min(), 
        all_z.max()-all_z.min()
    ]).max() / 2.0

    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Custom Labels to match user requirement
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Forward)') # Matplotlib Y axis represents Depth
    ax.set_zlabel('Y (Up)')      # Matplotlib Z axis represents Height
    
    ax.set_title(f"Cuboid Rotation (Y-Up System)\nEuler: {rotation_obj.as_euler('xyz', degrees=True).round(1)}")
    
    # Adjust view angle to see "Forward" clearly
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1,1,1])

    plt.show()

if __name__ == "__main__":
    # Example: Rotate 45 degrees around the vertical axis (Y in this system)
    # and 30 degrees around the forward axis (Z)
    rot = R.from_euler('xyz', (123, 456, 789))
    
    # Size: Width=2, Height=4 (Tall), Depth=1
    print(rotation_to_rotate_commands(rot))
    plot_rotated_cuboid(rot, sizes=(2, 2, 2))