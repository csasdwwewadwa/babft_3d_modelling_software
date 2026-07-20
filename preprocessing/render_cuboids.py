import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

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

CUBE_DATA = np.array([
    # Back Face
    -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,   0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
     0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
    # Front Face
    -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,   0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
     0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
    # Left Face
    -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
    -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
    # Right Face
     0.5,  0.5,  0.5,  1.0,  0.0,  0.0,   0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
     0.5, -0.5, -0.5,  1.0,  0.0,  0.0,   0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
    # Bottom Face
    -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,   0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
     0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
    # Top Face
    -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,   0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
     0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
], dtype=np.float32)

CUBE_INDICES = np.array([
    0, 2, 1,     0, 3, 2,     4, 5, 6,     4, 6, 7,
    8, 10, 9,    8, 11, 10,   12, 13, 14,  12, 14, 15,
    16, 17, 18,  16, 18, 19,  20, 22, 21,  20, 23, 22
], dtype=np.uint32)

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in mat4 iModel;
layout (location = 6) in vec3 iColor;

uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;
out vec3 fragNormal;
out vec3 fragWorldPos;

void main() {
    vec4 worldPos = iModel * vec4(aPos, 1.0);
    fragWorldPos = worldPos.xyz;
    gl_Position = projection * view * worldPos;
    
    fragColor = iColor;
    fragNormal = normalize(mat3(transpose(inverse(iModel))) * aNormal);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
in vec3 fragNormal;
in vec3 fragWorldPos;

uniform vec3 cameraPos; 
out vec4 FragColor;

void main() {
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 ambientColor = vec3(0.25, 0.25, 0.28); 
    
    vec3 norm = normalize(fragNormal);
    
    // HEADLIGHT LOGIC: Light direction tracks camera location dynamically
    vec3 lightDir = normalize(cameraPos - fragWorldPos);
    
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * 0.75;
    
    vec3 baseColor = fragColor;

    FragColor = vec4(baseColor * (ambientColor + diffuse), 1.0);
}
"""

cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
cam_distance = 2.5
cam_yaw, cam_pitch = 0.0, 0.2
last_mouse_x, last_mouse_y = 0.0, 0.0
is_mmb_pressed = False

def mouse_button_callback(window, button, action, mods):
    global is_mmb_pressed, last_mouse_x, last_mouse_y
    if button == glfw.MOUSE_BUTTON_MIDDLE:
        if action == glfw.PRESS:
            is_mmb_pressed = True
            last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            is_mmb_pressed = False

def cursor_pos_callback(window, xpos, ypos):
    global cam_yaw, cam_pitch, cam_target, last_mouse_x, last_mouse_y
    if not is_mmb_pressed: return
    dx, dy = xpos - last_mouse_x, ypos - last_mouse_y
    last_mouse_x, last_mouse_y = xpos, ypos
    
    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS:
        sin_y, cos_y = np.sin(cam_yaw), np.cos(cam_yaw)
        right_x, right_z = cos_y, -sin_y
        pan_speed = 0.002 * cam_distance
        cam_target[0] += right_x * (-dx * pan_speed)
        cam_target[1] += dy * pan_speed
        cam_target[2] += right_z * (-dx * pan_speed)
    else:
        cam_yaw -= dx * 0.005
        cam_pitch = np.clip(cam_pitch + dy * 0.005, -1.4, 1.4)

def scroll_callback(window, xoffset, yoffset):
    global cam_distance
    cam_distance = max(0.1, cam_distance - yoffset * 0.2)

def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(1024, 768, "Type-Safe Instanced Mesh Viewer", None, None)
    if not window: return glfw.terminate()
    
    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    
    shader = compileProgram(compileShader(VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
    
    cuboids = np.load("cuboids_data.npy", allow_pickle=True)
    num_instances = len(cuboids)
    
    all_centers = np.array([c['center'] for c in cuboids])
    center_offset = (np.min(all_centers, axis=0) + np.max(all_centers, axis=0)) / 2.0
    scale_factor = 1.5 / max(np.max(np.max(all_centers, axis=0) - np.min(all_centers, axis=0)), 0.001)
    
    # CRITICAL FIX: Statically packing interleaved array allocations using explicit 32-bit types
    buffer_dump = np.zeros(num_instances * 19, dtype=np.float32)
    
    for idx, c in enumerate(cuboids):
        norm_center = (c['center'] - center_offset) * scale_factor
        norm_extents = c['extents'] * scale_factor
        
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = c['rotation'] * norm_extents
        mat[:3, 3] = norm_center
        
        offset = idx * 19
        # Flatten Column-Major Matrix to buffer array
        buffer_dump[offset:offset+16] = mat.flatten('F')
        # Cast and force explicit float32 type bounds on RGB variables
        buffer_dump[offset+16:offset+19] = np.array(c['color'], dtype=np.float32)[:3]
        
    stride = 19 * 4 
    
    VAO = glGenVertexArrays(1)
    VBO, EBO, InstanceVBO = glGenBuffers(3)
    
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, CUBE_DATA.nbytes, CUBE_DATA, GL_STATIC_DRAW)
    
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, CUBE_INDICES.nbytes, CUBE_INDICES, GL_STATIC_DRAW)
    
    glBindBuffer(GL_ARRAY_BUFFER, InstanceVBO)
    glBufferData(GL_ARRAY_BUFFER, buffer_dump.nbytes, buffer_dump, GL_STATIC_DRAW)
    
    for i in range(4):
        slot = 2 + i
        glEnableVertexAttribArray(slot)
        glVertexAttribPointer(slot, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(i * 16))
        glVertexAttribDivisor(slot, 1)
        
    glEnableVertexAttribArray(6)
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16 * 4))
    glVertexAttribDivisor(6, 1)
    
    glUseProgram(shader)
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    cam_pos_loc = glGetUniformLocation(shader, "cameraPos")
    
    while not glfw.window_should_close(window):
        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)
        
        glClearColor(0.12, 0.12, 0.14, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        cam_x = cam_target[0] + cam_distance * np.cos(cam_pitch) * np.sin(cam_yaw)
        cam_z = cam_target[2] + cam_distance * np.cos(cam_pitch) * np.cos(cam_yaw)
        cam_y = cam_target[1] + cam_distance * np.sin(cam_pitch)
        
        proj_mat = perspective(45.0, width / max(1.0, height), 0.05, 100.0)
        view_mat = lookat([cam_x, cam_y, cam_z], cam_target, [0, 1, 0])
        
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj_mat)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_mat)
        glUniform3f(cam_pos_loc, cam_x, cam_y, cam_z) 
        
        glBindVertexArray(VAO)
        glDrawElementsInstanced(GL_TRIANGLES, len(CUBE_INDICES), GL_UNSIGNED_INT, None, num_instances)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
    glfw.terminate()

if __name__ == "__main__":
    main()