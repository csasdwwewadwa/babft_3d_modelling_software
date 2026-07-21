import numpy as np
from scipy.spatial import KDTree
import trimesh

def get_pixel_color_from_uv(mesh, face_indices, pts):
    """Maps 3D sample coordinates back to UV texture space using precise barycentric coordinates."""
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None or len(mesh.visual.uv) == 0:
        return None

    # SURGICAL FIX: Force deep reference tracking of this specific mesh's isolated material profile
    material = mesh.visual.material
    img = None
    # for attr in ['baseColorTexture', 'emissiveTexture', 'image', 'main_texture']:
    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
        img = material.baseColorTexture
    elif hasattr(material, 'image') and material.image is not None:
        img = material.image
    elif hasattr(material, 'emissiveTexture') and material.emissiveTexture is not None:
        img = material.emissiveTexture
        
    if img is None:
        return None
        
    if hasattr(img, 'image'):
        img = img.image
        
    if hasattr(img, 'convert'):
        img = img.convert('RGB')
        
    img_w, img_h = img.size
    img_arr = np.array(img, dtype=np.float32) / 255.0

    try:
        triangles = mesh.triangles[face_indices] # Shape: (N, 3, 3)
        bary = trimesh.triangles.points_to_barycentric(triangles, pts) # Shape: (N, 3)
        
        # Pull the exact UV coordinates for the 3 corners of each face
        uv_corners = mesh.visual.uv[mesh.faces[face_indices]] # Shape: (N, 3, 2)
        
        # Interpolate the UVs using the barycentric weights
        exact_uvs = np.sum(uv_corners * bary[:, :, np.newaxis], axis=1) # Shape: (N, 2)
        
        u = np.mod(exact_uvs[:, 0], 1.0)
        v = np.mod(exact_uvs[:, 1], 1.0)
        
        # Keep the original 1.0 - v inversion that worked perfectly for the main body parts
        px = np.clip((u * img_w).astype(int), 0, img_w - 1)
        py = np.clip(((1.0 - v) * img_h).astype(int), 0, img_h - 1)
        
        return img_arr[py, px, :3]
    except Exception as e:
        return None

def extract_global_scene_data(scene, target_total_points=500000):
    all_points = []
    all_normals = []
    all_colors = []
    
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)

    nodes = list(scene.graph.nodes_geometry)
    total_area = sum(scene.geometry[scene.graph[node][1]].area for node in nodes if scene.graph[node][1] in scene.geometry)
    if total_area == 0: total_area = 1.0

    print("Baking world space coordinates and extracting true texture maps...")
    for node in nodes:
        transform, geom_name = scene.graph[node]
        if geom_name not in scene.geometry: continue
        
        # Create an explicit unlinked copy so material dimensions don't leak between sub-meshes
        mesh = scene.geometry[geom_name].copy()
        if len(mesh.vertices) == 0: continue
        
        node_pts_count = int((mesh.area / total_area) * target_total_points)
        if node_pts_count < 500: node_pts_count = 500
        
        pts, face_indices = trimesh.sample.sample_surface(mesh, node_pts_count)
        norms = mesh.face_normals[face_indices]
        
        # Direct UV image texture sampling
        face_colors = get_pixel_color_from_uv(mesh, face_indices, pts)
        
        if face_colors is None:
            face_colors = np.ones((len(pts), 3), dtype=np.float32) * 0.75
            if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
                v_colors = np.array(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
                face_colors = v_colors[mesh.faces[face_indices]].mean(axis=1)

        # Bake transforms into world coordinates
        homo_pts = np.hstack((pts, np.ones((len(pts), 1))))
        world_pts = np.dot(homo_pts, transform.T)[:, :3]
        
        inv_trans_3x3 = np.linalg.inv(transform[:3, :3]).T
        world_norms = np.dot(norms, inv_trans_3x3.T)
        world_norms /= np.reshape(np.linalg.norm(world_norms, axis=1), (-1, 1))
        
        all_points.append(world_pts)
        all_normals.append(world_norms)
        all_colors.append(face_colors)

    return (np.vstack(all_points).astype(np.float32), 
            np.vstack(all_normals).astype(np.float32), 
            np.vstack(all_colors).astype(np.float32))

def build_accurate_flow_cuboids(points, normals, colors):
    num_pts = len(points)
    visited = np.zeros(num_pts, dtype=bool)
    tree = KDTree(points)
    
    mesh_scale = np.max(np.ptp(points, axis=0))
    step_r = mesh_scale * 0.015
    
    cuboids = []
    print("Generating curvature-aligned flow elements...")
    
    for idx in range(num_pts):
        if visited[idx]: continue
        
        seed_pt = points[idx]
        seed_normal = normals[idx]
        seed_color = colors[idx]
        
        indices = tree.query_ball_point(seed_pt, r=step_r)
        valid = np.array([i for i in indices if not visited[i]])
        
        if len(valid) < 3:
            visited[idx] = True
            continue
            
        norm_dots = np.abs(np.dot(normals[valid], seed_normal))
        color_diffs = np.linalg.norm(colors[valid] - seed_color, axis=1)
        
        cluster_mask = (norm_dots >= 0.9) & (color_diffs <= 0.06)
        final_cluster = valid[cluster_mask]
        
        if len(final_cluster) < 3:
            visited[idx] = True
            continue
            
        cell_points = points[final_cluster]
        center = np.mean(cell_points, axis=0)
        avg_color = np.mean(colors[final_cluster], axis=0)
        
        # 1. Fixed Normal Z-Axis Vector
        z_axis = seed_normal / np.linalg.norm(seed_normal)
        
        # 2. PCA Tangent Alignment Calculation
        centered_pts = cell_points - center
        projected_pts = centered_pts - np.outer(np.dot(centered_pts, z_axis), z_axis)
        
        if len(projected_pts) > 2:
            # Derive geometric surface flow direction via covariance eigenvalues
            cov = np.cov(projected_pts.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            x_axis = eigenvectors[:, np.argmax(eigenvalues)]
            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis # Orthogonalize
            x_axis /= np.linalg.norm(x_axis)
        else:
            ortho = np.array([1.0, 0.0, 0.0]) if np.abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(ortho, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        
        local_pts = np.dot(cell_points - center, rotation)
        min_b = np.min(local_pts, axis=0)
        max_b = np.max(local_pts, axis=0)
        extents = (max_b - min_b)
        
        # extents[2] = max(extents[2], mesh_scale * 0.002)
        extents[:2] *= 1.04

        # min size: 0.1
        extents = np.maximum(extents, 0.1)
        
        cuboids.append({
            'center': center.astype(np.float32),
            'rotation': rotation.astype(np.float32),
            'extents': extents.astype(np.float32),
            'color': avg_color.astype(np.float32)
        })
        
        visited[final_cluster] = True

    # remove small cuboids
    cuboids = [c for c in cuboids if not np.all(c['extents'] <= 0.15)]
        
    return cuboids

if __name__ == "__main__":
    scene = trimesh.load("glb/cirno.glb")
    pts, norms, cols = extract_global_scene_data(scene, target_total_points=500000)
    optimized_cuboids = build_accurate_flow_cuboids(pts, norms, cols)
    
    np.save("cuboids_data.npy", optimized_cuboids, allow_pickle=True)
    print(f"Generated {len(optimized_cuboids)} elements for the baseline population.")