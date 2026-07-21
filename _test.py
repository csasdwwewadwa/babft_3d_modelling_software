import trimesh
import os

def trace_scene_materials(file_path="fern.glb"):
    scene = trimesh.load(file_path)

    if not isinstance(scene, trimesh.Scene):
        print("Asset is a single mesh.")
        return
        
    print("\n=== GLB MATERIAL ARCHITECTURE TRACE ===")
    nodes = list(scene.graph.nodes_geometry)
    
    for i, node in enumerate(nodes):
        transform, geom_name = scene.graph[node]
        if geom_name not in scene.geometry: continue
        
        mesh = scene.geometry[geom_name]
        print(f"\n[Node {i}]: '{node}' -> Geometry Key: '{geom_name}'")
        print(f" -> Vertices: {len(mesh.vertices)} | Faces: {len(mesh.faces)}")
        
        if not hasattr(mesh.visual, 'material'):
            print(" -> Visuals: No material data attached.")
            continue
            
        mat = mesh.visual.material
        print(f" -> Material Object: {type(mat).__name__}")
        
        # Track where the image payload is hiding
        img = None
        source_attr = None
        for attr in ['baseColorTexture', 'emissiveTexture', 'image', 'main_texture']:
            if hasattr(mat, attr) and getattr(mat, attr) is not None:
                img = getattr(mat, attr)
                source_attr = attr
                break
                
        if img is not None:
            if hasattr(img, 'image'):  # Handle trimesh texture wrapper
                img = img.image
            w, h = img.size
            print(f" -> ACTIVE TEXTURE FOUND via '{source_attr}': Dimension {w}x{h} px")
            
            # Save the image to disk so you can see exactly what it looks like
            filename = f"texture_node_{i}_{geom_name}.png".replace("/", "_").replace("\\", "_")
            img.save(filename)
            print(f"    [DUMPED ASSIGNMENT]: Saved texture map asset to '{filename}'")
        else:
            print(" -> ACTIVE TEXTURE FOUND: None (Uses flat color shading profiles)")
            if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
                print(f"    -> Contains direct vertex colors array: Shape {mesh.visual.vertex_colors.shape}")

if __name__ == "__main__":
    trace_scene_materials()