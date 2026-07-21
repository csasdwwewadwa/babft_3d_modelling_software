import numpy as np
import json

def rotation_matrix_to_euler(matrix):
    """Converts a 3x3 rotation matrix to Euler angles (radians)."""
    sy = np.sqrt(matrix[0,0] * matrix[0,0] +  matrix[1,0] * matrix[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(matrix[2,1], matrix[2,2])
        y = np.arctan2(-matrix[2,0], sy)
        z = np.arctan2(matrix[1,0], matrix[0,0])
    else:
        x = np.arctan2(-matrix[1,2], matrix[1,1])
        y = np.arctan2(-matrix[2,0], sy)
        z = 0
    return [float(x), float(y), float(z)]

def convert_npy_to_json(npy_path="cuboids_data.npy", json_path="cuboids.json"):
    cuboids = np.load(npy_path, allow_pickle=True)
    json_output = []

    for c in cuboids:
        entry = {
            "position": c['center'].tolist(),
            "rotation": rotation_matrix_to_euler(c['rotation']),
            "scale": c['extents'].tolist(),
            "color": (c['color'] * 255).astype(int).tolist()
        }
        json_output.append(entry)

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=4)
    
    print(f"Successfully converted {len(json_output)} cuboids to {json_path}")

# Run the converter
if __name__ == "__main__":
    convert_npy_to_json()