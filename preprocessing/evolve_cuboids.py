import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import cv2
import trimesh
import copy

# --- MATRICES ---
def perspective(fov, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0] = f / aspect
    m[1,1] = f
    m[2,2] = (far + near) / (near - far)
    m[2,3] = -1.0
    m[3,2] = (2.0 * far * near) / (near - far)
    return m

def lookat(eye, target, up):
    z = np.array(eye) - np.array(target)
    z /= np.linalg.norm(z)
    x = np.cross(np.array(up), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4, dtype=np.float32)
    m[0,0], m[1,0], m[2,0] = x
    m[0,1], m[1,1], m[2,1] = y
    m[0,2], m[1,2], m[2,2] = z
    m[3,0] = -np.dot(x, eye)
    m[3,1] = -np.dot(y, eye)
    m[3,2] = -np.dot(z, eye)
    return m

# --- BASE CUBE FOR DYNAMIC INDIVIDUALS ---
CUBE_DATA = np.array([
    -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,   0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
     0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
    -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,   0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
     0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
    -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
    -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
     0.5,  0.5,  0.5,  1.0,  0.0,  0.0,   0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
     0.5, -0.5, -0.5,  1.0,  0.0,  0.0,   0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
    -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,   0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
     0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
    -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,   0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
     0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
], dtype=np.float32)

CUBE_INDICES = np.array([
    0, 2, 1,     0, 3, 2,     4, 5, 6,     4, 6, 7,
    8, 10, 9,    8, 11, 10,   12, 13, 14,  12, 14, 15,
    16, 17, 18,  16, 18, 19,  20, 22, 21,  20, 23, 22
], dtype=np.uint32)

# --- SHADERS ---
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords; 

layout (location = 3) in mat4 iModel;
layout (location = 7) in vec3 iColor;

uniform mat4 view;
uniform mat4 projection;
uniform bool isInstancedMode; 
uniform mat4 staticModel;

out vec3 fragColor;
out vec2 fragTexCoords;
out vec3 fragNormal;
out vec3 fragWorldPos;

void main() {
    mat4 modelMat = isInstancedMode ? iModel : staticModel;
    vec4 worldPos = modelMat * vec4(aPos, 1.0);
    
    fragWorldPos = worldPos.xyz;
    gl_Position = projection * view * worldPos;
    
    fragTexCoords = aTexCoords;
    fragColor = isInstancedMode ? iColor : vec3(1.0); 
    fragNormal = normalize(mat3(transpose(inverse(modelMat))) * aNormal);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
in vec2 fragTexCoords;
in vec3 fragNormal;
in vec3 fragWorldPos;

uniform vec3 cameraPos; 
uniform bool renderMaskMode;
uniform bool isInstancedMode;
uniform sampler2D textureMap; 
uniform bool hasTexture;

out vec4 FragColor;

void main() {
    if (renderMaskMode) {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    } else {
        vec3 baseTexColor = vec3(1.0, 0.41, 0.70); // Fallback debug pink if texture completely drops
        if (isInstancedMode) {
            baseTexColor = fragColor;
        } else if (hasTexture) {
            baseTexColor = texture(textureMap, fragTexCoords).rgb;
        }
        
        vec3 lightColor = vec3(1.0, 1.0, 1.0);
        vec3 ambientColor = vec3(0.22, 0.22, 0.25); 
        
        vec3 norm = normalize(fragNormal);
        vec3 lightDir = normalize(cameraPos - fragWorldPos);
        
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor * 0.78;
        
        FragColor = vec4(baseTexColor * (ambientColor + diffuse), 1.0);
    }
}
"""

class UnifiedNativeRenderer:
    def __init__(self, res=(1024, 1024)):
        self.res = res
        if not glfw.init(): raise RuntimeError("Failed to init GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(res[0], res[1], "Unified Engine", None, None)
        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)
        
        self.shader = compileProgram(compileShader(VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        
        # 1. Setup Instanced Cuboid Buffers
        self.cubeVAO = glGenVertexArrays(1)
        self.cubeVBO, self.cubeEBO, self.InstanceVBO = glGenBuffers(3)
        glBindVertexArray(self.cubeVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.cubeVBO)
        glBufferData(GL_ARRAY_BUFFER, CUBE_DATA.nbytes, CUBE_DATA, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*4, ctypes.c_void_p(3*4))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cubeEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, CUBE_INDICES.nbytes, CUBE_INDICES, GL_STATIC_DRAW)
        
        # Setup Instance attributes slots
        glBindBuffer(GL_ARRAY_BUFFER, self.InstanceVBO)
        for i in range(4):
            glEnableVertexAttribArray(3+i)
            glVertexAttribPointer(3+i, 4, GL_FLOAT, GL_FALSE, 19*4, ctypes.c_void_p(i*16))
            glVertexAttribDivisor(3+i, 1)
        glEnableVertexAttribArray(7); glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, 19*4, ctypes.c_void_p(16*4))
        glVertexAttribDivisor(7, 1)

        # 2. Setup Asynchronous Extraction PBO
        self.PBO = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.PBO)
        glBufferData(GL_PIXEL_PACK_BUFFER, res[0]*res[1]*3, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        
        self.center_offset = np.zeros(3, dtype=np.float32)
        self.scale_factor = 1.0
        self.glbVAO = None
        self.glbCount = 0
        self.textureID = None

    def load_baseline_glb(self, path):
        """Extracts geometry, normalizes coordinates, and securely binds high-resolution textures."""
        mesh = trimesh.load(path)
        
        # Unroll geometry scene hierarchies cleanly
        if isinstance(mesh, trimesh.Scene):
            geo = mesh.to_geometry()
        else:
            geo = mesh

        verts = geo.vertices.copy()
        normals = geo.vertex_normals.copy()
        
        # Safely extract high-resolution image metrics from Trimesh material layers
        img_source = None
        if hasattr(geo, 'visual') and hasattr(geo.visual, 'material'):
            mat = geo.visual.material
            if hasattr(mat, 'image') and mat.image is not None:
                img_source = mat.image
            elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                img_source = mat.baseColorTexture

        if img_source is not None:
            img = np.array(img_source.convert('RGB'))
            img_data = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            self.textureID = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.textureID)
            
            # FIX 1: Enforce explicit texture sampling bounds to stop black edge artifacts
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_data.shape[1], img_data.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)

        # Extract UV coordinates safely
        if hasattr(geo, 'visual') and hasattr(geo.visual, 'uv'):
            uvs = geo.visual.uv.astype(np.float32).copy()
            # FIX 2: Invert the V axis. OpenGL texture tracking reads textures from bottom-left; GLTF reads from top-left.
            uvs[:, 1] = 1.0 - uvs[:, 1]
        else:
            uvs = np.zeros((len(verts), 2), dtype=np.float32)
        
        # Viewport alignment setup
        min_bounds = np.min(verts, axis=0)
        max_bounds = np.max(verts, axis=0)
        self.center_offset = (min_bounds + max_bounds) / 2.0
        self.scale_factor = 1.6 / max(np.max(max_bounds - min_bounds), 0.001)
        
        verts = (verts - self.center_offset) * self.scale_factor
        
        packed_data = np.hstack([verts, normals, uvs]).astype(np.float32)
        indices = geo.faces.flatten().astype(np.uint32)
        self.glbCount = len(indices)
        
        self.glbVAO = glGenVertexArrays(1)
        self.glbVBO, glbEBO = glGenBuffers(2)
        
        glBindVertexArray(self.glbVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.glbVBO)
        glBufferData(GL_ARRAY_BUFFER, packed_data.nbytes, packed_data, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*4, ctypes.c_void_p(3*4))
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*4, ctypes.c_void_p(6*4))
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glbEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

    def upload_cuboids(self, cuboids):
        num_instances = len(cuboids)
        if num_instances == 0: return 0
        
        buffer_dump = np.zeros(num_instances * 19, dtype=np.float32)
        for idx, c in enumerate(cuboids):
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = c['rotation'] * (c['extents'] * self.scale_factor)
            mat[:3, 3] = (c['center'] - self.center_offset) * self.scale_factor
            offset = idx * 19
            buffer_dump[offset:offset+16] = mat.flatten('F')
            buffer_dump[offset+16:offset+19] = np.array(c['color'], dtype=np.float32)[:3]
            
        glBindBuffer(GL_ARRAY_BUFFER, self.InstanceVBO)
        glBufferData(GL_ARRAY_BUFFER, buffer_dump.nbytes, buffer_dump, GL_DYNAMIC_DRAW)
        return num_instances

    def render_native_pass(self, eye, target=[0,0,0], up=[0,1,0], mask_mode=False, instanced_mode=True, count=0):
        glViewport(0, 0, self.res[0], self.res[1])
        glClearColor(0.0, 0.0, 0.0, 1.0) if mask_mode else glClearColor(0.12, 0.12, 0.14, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, perspective(45.0, self.res[0]/self.res[1], 0.05, 100.0))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, lookat(eye, target, up))
        glUniform3f(glGetUniformLocation(self.shader, "cameraPos"), eye[0], eye[1], eye[2])
        glUniform1i(glGetUniformLocation(self.shader, "renderMaskMode"), 1 if mask_mode else 0)
        glUniform1i(glGetUniformLocation(self.shader, "isInstancedMode"), 1 if instanced_mode else 0)
        
        if instanced_mode:
            glUniform1i(glGetUniformLocation(self.shader, "hasTexture"), 0)
            glBindVertexArray(self.cubeVAO)
            glDrawElementsInstanced(GL_TRIANGLES, len(CUBE_INDICES), GL_UNSIGNED_INT, None, count)
        else:
            glUniformMatrix4fv(glGetUniformLocation(self.shader, "staticModel"), 1, GL_FALSE, np.eye(4, dtype=np.float32))
            
            # FIX 3: Fully bind context textures safely while specifying the active uniform sampling slot explicitly
            if self.textureID is not None and not mask_mode:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self.textureID)
                glUniform1i(glGetUniformLocation(self.shader, "textureMap"), 0)
                glUniform1i(glGetUniformLocation(self.shader, "hasTexture"), 1)
            else:
                glUniform1i(glGetUniformLocation(self.shader, "hasTexture"), 0)
                
            glBindVertexArray(self.glbVAO)
            glDrawElements(GL_TRIANGLES, self.glbCount, GL_UNSIGNED_INT, None)
            glBindTexture(GL_TEXTURE_2D, 0)
            
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.PBO)
        glReadPixels(0, 0, self.res[0], self.res[1], GL_BGR, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        raw_data = ctypes.string_at(ptr, self.res[0] * self.res[1] * 3)
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        
        return cv2.flip(np.frombuffer(raw_data, dtype=np.uint8).reshape(self.res[1], self.res[0], 3), 0)

    def close(self): glfw.terminate()

# --- HIGH DENSITY UNIFIED MATRIX SAMPLING ---
def capture_matrix(renderer, num_angles=24, radius=2.5, instanced_mode=True, count=0):
    color_views, mask_views = [], []
    golden_ratio = (1 + 5 ** 0.5) / 2
    for i in range(num_angles):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1.0 - 2.0 * (i + 0.5) / num_angles)
        eye = [radius * np.sin(phi) * np.cos(theta), radius * np.sin(phi) * np.sin(theta), radius * np.cos(phi)]
        
        color_img = renderer.render_native_pass(eye, mask_mode=False, instanced_mode=instanced_mode, count=count)
        mask_img = renderer.render_native_pass(eye, mask_mode=True, instanced_mode=instanced_mode, count=count)
        
        color_views.append(color_img)
        mask_views.append(mask_img[:, :, 0] > 0)
    return color_views, mask_views

def save_dashboard_to_png(ref_color, current_color, filepath="evolution_dashboard.png"):
    sample_indices = [0, 6, 12, 18, 23]
    top_row = np.hstack([cv2.resize(ref_color[i], (256, 256)) for i in sample_indices])
    bottom_row = np.hstack([cv2.resize(current_color[i], (256, 256)) for i in sample_indices])
    dashboard = np.vstack([top_row, bottom_row])
    cv2.putText(dashboard, "TRUE GLB BASELINE (NATIVE PIPELINE)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, "EVOLVED MODEL CUBOIDS", (10, top_row.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(filepath, dashboard)

# --- RUN LOOP ENGINE ---
def run_evolution(glb_path="cirno.glb", cuboids_path="cuboids_data.npy", generations=2048):
    renderer = UnifiedNativeRenderer(res=(1024, 1024))
    
    print(f"Uploading True GLB geometry buffers: {glb_path}...")
    renderer.load_baseline_glb(glb_path)
    ref_color, ref_masks = capture_matrix(renderer, num_angles=24, instanced_mode=False)
    
    cuboids = list(np.load(cuboids_path, allow_pickle=True))
    num_instances = renderer.upload_cuboids(cuboids)
    current_color, current_masks = capture_matrix(renderer, num_angles=24, instanced_mode=True, count=num_instances)
    
    save_dashboard_to_png(ref_color, current_color)
    
    color_loss = sum(np.mean((r.astype(np.float32)-c.astype(np.float32))**2) for r, c in zip(ref_color, current_color))
    silhouette_loss = sum(np.sum(np.logical_xor(rm, cm)) for rm, cm in zip(ref_masks, current_masks))
    current_loss = (color_loss / 24.0) + (silhouette_loss / (24.0 * 1024 * 1024)) * 150.0
    
    print(f"Calibrated Baseline Entry Loss Score: {current_loss:.4f}")
    
    for gen in range(generations):
        mutated_cuboids = copy.deepcopy(cuboids)
        # num_mutations = max(1, int(len(cuboids) * 0.08))
        num_mutations = 1
        for idx in np.random.choice(len(cuboids), size=num_mutations, replace=False):
            c = mutated_cuboids[idx]
            mod_type = np.random.choice(['pos', 'scale', 'rot'])
            if mod_type == 'pos': c['center'] += np.random.normal(0, 0.05, size=3)
            elif mod_type == 'scale': c['extents'] = np.clip(c['extents'] * np.random.normal(1.0, 0.03, size=3), 0.001, None)
            elif mod_type == 'rot':
                angle = np.random.normal(0, 0.03)
                axis = np.random.normal(0, 1.0, size=3); axis /= np.linalg.norm(axis)
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                c['rotation'] = np.dot(np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K, K), c['rotation'])
        
        cand_count = renderer.upload_cuboids(mutated_cuboids)
        cand_color, cand_masks = capture_matrix(renderer, num_angles=24, instanced_mode=True, count=cand_count)
        
        c_loss = sum(np.mean((r.astype(np.float32)-c.astype(np.float32))**2) for r, c in zip(ref_color, cand_color))
        # Before calculating s_loss, blur the masks
        kernel = np.ones((5,5), np.uint8)
        # Then use a simplified Mean Squared Error on the masks instead of XOR
        s_loss = sum(np.mean((cv2.dilate(rm.astype(np.uint8), kernel, iterations=1).astype(float) - cv2.dilate(cm.astype(np.uint8), kernel, iterations=1).astype(float))**2) for rm, cm in zip(ref_masks, current_masks))
        # s_loss = sum(np.sum(np.logical_xor(rm, cm)) for rm, cm in zip(ref_masks, cand_masks))
        candidate_loss = (c_loss / 24.0) + (s_loss / (24.0 * 1024 * 1024)) * 150.0
        
        if candidate_loss < current_loss:
            current_loss, cuboids, current_color = candidate_loss, mutated_cuboids, cand_color
            np.save(cuboids_path, np.array(cuboids))
            save_dashboard_to_png(ref_color, current_color)
            
            print(f" -> Gen {gen+1:02d}: Accepted. Alignment Score: {current_loss:.4f}")
        else:
            save_dashboard_to_png(ref_color, cand_color)
            print(f" -> Gen {gen+1:02d}: Rejected.")
            
    renderer.close()

if __name__ == "__main__":
    run_evolution()