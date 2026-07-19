import numpy as np
from scipy.spatial import KDTree
import trimesh

def generate_point_cloud_with_exact_textures(mesh, num_points=200000):
    print(f"Sampling {num_points} points from surface...")
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    colors = np.ones((len(points), 3)) * 0.5 
    
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and hasattr(mesh.visual, 'material'):
        material = mesh.visual.material
        if hasattr(material, 'image') and material.image is not None:
            img = material.image.convert('RGB')
            img_w, img_h = img.size
            img_array = np.array(img) / 255.0
            
            uvs = mesh.visual.uv
            face_uvs = uvs[mesh.faces[face_indices]]
            
            bary = trimesh.triangles.points_to_barycentric(mesh.vertices[mesh.faces[face_indices]], points)
            bary = np.clip(bary, 0, 1)
            bary /= np.sum(bary, axis=1, keepdims=True)
            
            exact_uvs = np.sum(face_uvs * bary[:, :, None], axis=1)
            
            pixel_x = np.clip((exact_uvs[:, 0] * img_w).astype(int), 0, img_w - 1)
            pixel_y = np.clip(((1.0 - exact_uvs[:, 1]) * img_h).astype(int), 0, img_h - 1)
            
            colors = img_array[pixel_y, pixel_x, :3]
            return points, colors

    return points, colors

def get_local_orientation(local_points):
    center = np.mean(local_points, axis=0)
    centered = local_points - center
    cov = np.dot(centered.T, centered) / len(local_points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, idx]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return center, rotation

def decompose_mesh_to_cuboids(mesh, max_cuboids=8000, color_threshold=0.08, initial_k=60):
    # Higher initial_k allows blocks to start larger and flatter
    points, colors = generate_point_cloud_with_exact_textures(mesh, num_points=200000)
    tree = KDTree(points)
    
    num_pts = len(points)
    visited = np.zeros(num_pts, dtype=bool)
    cuboids = []
    
    mesh_max_dimension = np.max(mesh.extents)
    # Allow blocks to start much wider on flat areas
    max_single_block_size = mesh_max_dimension * 0.35 

    print("Extracting adaptive structural surface cuboids...")
    
    for _ in range(max_cuboids):
        unvisited_indices = np.where(~visited)[0]
        if len(unvisited_indices) == 0:
            break
            
        seed_idx = unvisited_indices[0]
        seed_pt = points[seed_idx]
        seed_color = colors[seed_idx]
        
        _, neighborhood = tree.query(seed_pt, k=initial_k)
        valid_neighbors = neighborhood[~visited[neighborhood]]
        
        if len(valid_neighbors) < 8:
            visited[seed_idx] = True
            continue
            
        neighbor_colors = colors[valid_neighbors]
        color_dists = np.linalg.norm(neighbor_colors - seed_color, axis=1)
        valid_neighbors = valid_neighbors[color_dists < color_threshold]
        
        if len(valid_neighbors) < 6:
            visited[seed_idx] = True
            continue

        center, rotation = get_local_orientation(points[valid_neighbors])
        cluster_local_pts = np.dot(points[valid_neighbors] - center, rotation)
        
        min_bounds = np.min(cluster_local_pts, axis=0)
        max_bounds = np.max(cluster_local_pts, axis=0)
        extents = (max_bounds - min_bounds)
        
        local_center_offset = (min_bounds + max_bounds) / 2.0
        center = center + np.dot(rotation, local_center_offset)
        
        # Keep thin parts clean, but don't limit flat surface expansion
        min_thickness = np.max(extents) * 0.15
        extents[2] = max(extents[2], min_thickness) 
        extents = np.clip(extents, 0.005, max_single_block_size)
        
        local_all_pts = np.dot(points - center, rotation)
        inside_mask = np.all(np.abs(local_all_pts) <= (extents / 2.0), axis=1)
        
        actual_captured = np.where(inside_mask & ~visited)[0]
        if len(actual_captured) == 0:
            visited[seed_idx] = True
            continue
            
        visited[inside_mask] = True
        
        cuboids.append({
            'center': center,
            'rotation': rotation,
            'extents': extents,
            'color': seed_color,
            'active': True
        })
        
    return cuboids

def merge_compatible_cuboids_fast(cuboids, angle_tolerance=0.94):
    print("Merging blocks into flat sheets and geometric shapes...")
    active_cuboids = [c for c in cuboids if c['active']]
    
    for pass_num in range(5):
        centers = np.array([c['center'] for c in active_cuboids if c['active']])
        if len(centers) < 2:
            break
            
        cuboid_tree = KDTree(centers)
        mapping = [i for i, c in enumerate(active_cuboids) if c['active']]
        merged_in_pass = 0
        
        for idx, current_idx in enumerate(mapping):
            c1 = active_cuboids[current_idx]
            if not c1['active']:
                continue
                
            max_search_k = min(16, len(centers))
            dists, neighbors = cuboid_tree.query(c1['center'], k=max_search_k)
            
            if max_search_k == 1:
                neighbors = [neighbors]
                
            for neighbor_map_idx in neighbors:
                target_idx = mapping[neighbor_map_idx]
                if target_idx == current_idx:
                    continue
                    
                c2 = active_cuboids[target_idx]
                if not c2['active']:
                    continue
                    
                # Quantized color matching
                if np.linalg.norm(c1['color'] - c2['color']) > 0.05:
                    continue
                    
                # Alignment check
                alignment = abs(np.dot(c1['rotation'][:, 0], c2['rotation'][:, 0]))
                if alignment < angle_tolerance:
                    continue
                    
                center_vector = c2['center'] - c1['center']
                center_dist = np.linalg.norm(center_vector)
                
                # Relaxed touch condition to allow flattening large sheets like the cape
                max_contact_bound = (np.max(c1['extents']) + np.max(c2['extents'])) * 0.75
                
                if center_dist <= max_contact_bound:
                    combined_center = (c1['center'] + c2['center']) / 2.0
                    combined_extents = np.maximum(c1['extents'], c2['extents'])
                    
                    offset_axis = np.argmax(np.abs(np.dot(c1['rotation'].T, center_vector)))
                    combined_extents[offset_axis] += center_dist * 0.5
                    
                    # ASPECT RATIO: Bumped up to 15.0 to allow beautiful flat panels for capes/hair
                    if np.max(combined_extents) / np.min(combined_extents) > 15.0:
                        continue
                        
                    c1['center'] = combined_center
                    c1['extents'] = combined_extents
                    c2['active'] = False
                    merged_in_pass += 1
                    
        if merged_in_pass == 0:
            break
            
    return [c for c in active_cuboids if c['active']]

def save_gltf_multi_material(cuboids, output_path="flat_shaded_paimon.glb"):
    """
    Groups cuboids by color and assigns discrete PBR materials.
    Guarantees instant vibrant colors inside Blender's Material Preview.
    """
    print("Grouping by color nodes and inflating blocks to seamless bounds...")
    
    # Simple color quantization to avoid creating thousands of identical materials
    def quantize_color(rgb):
        return tuple((np.round(rgb * 20) / 20).tolist())

    color_groups = {}
    for box in cuboids:
        q_key = quantize_color(box['color'])
        if q_key not in color_groups:
            color_groups[q_key] = []
        color_groups[q_key].append(box)

    scene = trimesh.Scene()
    base_cube = trimesh.creation.box(extents=[1, 1, 1])
    cube_verts = base_cube.vertices
    cube_faces = base_cube.faces

    for rgb_key, boxes in color_groups.items():
        all_vertices = []
        all_faces = []
        vertex_count = 0
        
        for box_data in boxes:
            # GAP SOLUTION: Overlap padding factor (1.06) fills out the ridges seamlessly
            padded_extents = box_data['extents'] * 1.06
            
            scaled_verts = cube_verts * padded_extents
            rotated_verts = np.dot(scaled_verts, box_data['rotation'].T)
            transformed_verts = rotated_verts + box_data['center']
            
            all_vertices.append(transformed_verts)
            all_faces.append(cube_faces + vertex_count)
            vertex_count += len(cube_verts)
            
        group_vertices = np.vstack(all_vertices)
        group_faces = np.vstack(all_faces)
        
        sub_mesh = trimesh.Trimesh(vertices=group_vertices, faces=group_faces)
        
        # Generate an explicit standalone color material track
        mat_color = list(rgb_key) + [1.0] # Append full alpha
        sub_mesh.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=mat_color,
            metallicFactor=0.0,
            roughnessFactor=0.7
        )
        scene.add_geometry(sub_mesh)

    scene.export(output_path, file_type="glb")
    print(f"Success! Scalable mesh asset compiled at: {output_path}")

if __name__ == "__main__":
    test_mesh = trimesh.load("plush.glb")
    if isinstance(test_mesh, trimesh.Scene):
        test_mesh = test_mesh.to_geometry()

    # Process and build
    raw_cuboids = decompose_mesh_to_cuboids(test_mesh, max_cuboids=12000, color_threshold=0.08, initial_k=60)
    optimized_cuboids = merge_compatible_cuboids_fast(raw_cuboids, angle_tolerance=0.94)
    save_gltf_multi_material(optimized_cuboids, "flat_shaded_paimon.glb")