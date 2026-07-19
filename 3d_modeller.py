import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import json
import os

# --- Constants ---
SCREEN_SIZE = (1280, 720)

class Cuboid:
    def __init__(self, x=0.0, y=0.0, z=0.0, color=(0.7, 0.7, 0.7)):
        self.position = np.array([x, y, z], dtype=float)
        self.scale = np.array([1.0, 1.0, 1.0], dtype=float)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=float) 
        self.color = color
        
        self._backup_pos = None
        self._backup_rot = None
        self._backup_scl = None

    def store_state(self):
        self._backup_pos = self.position.copy()
        self._backup_rot = self.rotation.copy()
        self._backup_scl = self.scale.copy()

    def restore_state(self):
        if self._backup_pos is not None:
            self.position = self._backup_pos.copy()
            self.rotation = self._backup_rot.copy()
            self.scale = self._backup_scl.copy()

    def to_dict(self):
        rgb_int = [int(c * 255) for c in self.color] if self.color else [180, 180, 180]
        return {
            "position": self.position.tolist(),
            "rotation": np.radians(self.rotation).tolist(),
            "scale": self.scale.tolist(),
            "color": rgb_int
        }

    @staticmethod
    def from_dict(d):
        c = Cuboid()
        c.position = np.array(d["position"], dtype=float)
        c.rotation = np.degrees(np.array(d["rotation"], dtype=float))
        c.scale = np.array(d["scale"], dtype=float)
        if d.get("color"):
            c.color = tuple([val / 255.0 for val in d["color"]])
        else:
            c.color = (0.7, 0.7, 0.7)
        return c

class Camera:
    def __init__(self):
        self.target = np.array([0.0, 0.0, 0.0], dtype=float)
        self.yaw = 45.0
        self.pitch = 30.0
        self.distance = 20.0
        self.ortho_mode = False 
        self.sensitivity = 0.2
        self.pan_speed = 0.02

    def get_pos(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        x = self.target[0] + self.distance * math.cos(rad_pitch) * math.sin(rad_yaw)
        y = self.target[1] + self.distance * math.sin(rad_pitch)
        z = self.target[2] + self.distance * math.cos(rad_pitch) * math.cos(rad_yaw)
        return x, y, z

    def update(self, dx, dy, mode='ORBIT'):
        if mode == 'ORBIT':
            self.yaw -= dx * self.sensitivity
            self.pitch += dy * self.sensitivity
            self.pitch = max(-89.9, min(89.9, self.pitch))
        elif mode == 'PAN':
            rad_yaw = math.radians(self.yaw)
            rx = math.cos(rad_yaw)
            rz = -math.sin(rad_yaw)
            speed = self.pan_speed * (self.distance / 10.0)
            if self.ortho_mode: speed *= 2
            
            pan_x = (rx * -dx) * speed
            pan_z = (rz * -dx) * speed
            pan_y = dy * speed
            self.target += np.array([pan_x, pan_y, pan_z])

    def set_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = SCREEN_SIZE[0] / SCREEN_SIZE[1]
        if self.ortho_mode:
            size = self.distance * 0.5 
            glOrtho(-size * aspect, size * aspect, -size, size, -500, 500)
        else:
            gluPerspective(60, aspect, 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)

    def apply(self):
        px, py, pz = self.get_pos()
        if abs(self.pitch) > 88:
            up = (0, 0, -1) if self.pitch > 0 else (0, 0, 1)
        else:
            up = (0, 1, 0)
        gluLookAt(px, py, pz, *self.target, *up)
        
    def get_right_vector(self):
        rad_yaw = math.radians(self.yaw)
        return np.array([math.cos(rad_yaw), 0, -math.sin(rad_yaw)])

    def get_up_vector(self):
        return np.array([0, 1, 0])

class Engine:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode(SCREEN_SIZE, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Cuboid Blender")
        
        self.camera = Camera()
        self.cuboids = []
        self.active_cuboid = None
        
        self.history = [] 
        self.history_index = -1 
        
        self.transform_mode = None 
        self.axis_constraint = None 
        self.accumulated_mouse = [0, 0]

        # Init Scene
        ground = Cuboid(0, -1, 0, (0.3, 0.3, 0.3))
        ground.scale = np.array([10.0, 0.2, 10.0])
        self.cuboids.append(ground)
        self.cuboids.append(Cuboid(0, 1, 0, (0.2, 0.6, 1.0)))
        
        self.save_history_state()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE); glCullFace(GL_BACK)
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_COLOR_MATERIAL); glEnable(GL_NORMALIZE)
        glLightfv(GL_LIGHT0, GL_POSITION, (5, 10, 10, 0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))

        # Fix picking alignment
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    # --- UNDO / REDO ---
    def save_history_state(self):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index+1]
        scene_dump = [c.to_dict() for c in self.cuboids]
        self.history.append(scene_dump)
        self.history_index += 1
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.load_from_history()
            print("Undo")

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.load_from_history()
            print("Redo")

    def load_from_history(self):
        data = self.history[self.history_index]
        self.cuboids = [Cuboid.from_dict(d) for d in data]
        self.active_cuboid = None

    # --- RENDER HELPERS ---
    def get_color_id(self, index):
        r = (index & 0x0000FF); g = (index & 0x00FF00) >> 8; b = (index & 0xFF0000) >> 16
        return (r/255.0, g/255.0, b/255.0)

    def draw_unit_cube(self, mode=GL_QUADS):
        vertices = [
            (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5),
            (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5)
        ]
        faces = [
            ((0, 0, -1), (0, 3, 2, 1)), ((0, 0, 1), (4, 5, 7, 6)),
            ((-1, 0, 0), (7, 2, 3, 6)), ((1, 0, 0), (0, 1, 5, 4)),
            ((0, 1, 0), (2, 7, 5, 1)), ((0, -1, 0), (0, 4, 6, 3))
        ]
        glBegin(mode)
        for normal, vertex_indices in faces:
            glNormal3fv(normal)
            for i in vertex_indices: glVertex3fv(vertices[i])
        glEnd()

    def draw_axis_line(self, center, rotation=None):
        if self.axis_constraint is None: return
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glLineWidth(2)
        glPushMatrix()
        glTranslatef(*center)
        if rotation is not None:
             glRotatef(rotation[2], 0, 0, 1)
             glRotatef(rotation[0], 1, 0, 0)
             glRotatef(rotation[1], 0, 1, 0)
        glBegin(GL_LINES)
        size = 1000.0
        if self.axis_constraint == 0:
            glColor3f(1, 0.2, 0.2); glVertex3f(-size, 0, 0); glVertex3f(size, 0, 0)
        elif self.axis_constraint == 1:
            glColor3f(0.2, 1, 0.2); glVertex3f(0, -size, 0); glVertex3f(0, size, 0)
        elif self.axis_constraint == 2:
            glColor3f(0.2, 0.2, 1); glVertex3f(0, 0, -size); glVertex3f(0, 0, size)
        glEnd()
        glPopMatrix()
        glLineWidth(1); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

    def draw_scene(self, picking=False):
        if not picking:
            glDisable(GL_LIGHTING); glLineWidth(1); glBegin(GL_LINES)
            glColor3f(0.25, 0.25, 0.25)
            for i in range(-20, 21):
                glVertex3f(i, 0, -20); glVertex3f(i, 0, 20)
                glVertex3f(-20, 0, i); glVertex3f(20, 0, i)
            glColor3f(1, 0, 0); glVertex3f(0,0,0); glVertex3f(1,0,0)
            glColor3f(0, 1, 0); glVertex3f(0,0,0); glVertex3f(0,1,0)
            glColor3f(0, 0, 1); glVertex3f(0,0,0); glVertex3f(0,0,1)
            glEnd(); glEnable(GL_LIGHTING)

        for i, cube in enumerate(self.cuboids):
            glPushMatrix()
            glTranslatef(*cube.position)
            glRotatef(cube.rotation[2], 0, 0, 1)
            glRotatef(cube.rotation[0], 1, 0, 0)
            glRotatef(cube.rotation[1], 0, 1, 0)
            glScalef(*cube.scale)
            if picking:
                glDisable(GL_LIGHTING); glColor3fv(self.get_color_id(i))
                self.draw_unit_cube()
                glEnable(GL_LIGHTING)
            else:
                if cube == self.active_cuboid:
                    glDisable(GL_LIGHTING); glColor3f(1.0, 0.6, 0.0) 
                    self.draw_unit_cube()
                    glColor3f(1, 1, 1); glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    glLineWidth(2); self.draw_unit_cube(); glLineWidth(1)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); glEnable(GL_LIGHTING)
                else:
                    glColor3fv(cube.color); self.draw_unit_cube()
            glPopMatrix()

        if not picking and self.transform_mode and self.active_cuboid:
            rot = self.active_cuboid.rotation if self.transform_mode in ['ROTATE', 'SCALE'] else None
            self.draw_axis_line(self.active_cuboid.position, rotation=rot)

    def pick_object(self, x, y):
        glClearColor(1, 1, 1, 1); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.set_projection(); glLoadIdentity(); self.camera.apply()
        self.draw_scene(picking=True)
        read_y = SCREEN_SIZE[1] - y - 1
        if read_y < 0 or read_y >= SCREEN_SIZE[1]: return None
        try:
            data = glReadPixels(x, read_y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
            if isinstance(data, bytes): r, g, b = data[0], data[1], data[2]
            else: flat = np.array(data).flatten(); r, g, b = flat[0], flat[1], flat[2]
            if r == 255 and g == 255 and b == 255: return None
            idx = r + (g << 8) + (b << 16)
            return self.cuboids[idx] if 0 <= idx < len(self.cuboids) else None
        except: return None

    def snap(self, value, step):
        return round(value / step) * step

    def update_title(self):
        if not self.transform_mode:
            pygame.display.set_caption("Cuboid Blender")
            return
        axis_str = "Global"
        if self.axis_constraint == 0: axis_str = "X-Axis"
        elif self.axis_constraint == 1: axis_str = "Y-Axis"
        elif self.axis_constraint == 2: axis_str = "Z-Axis"
        if self.transform_mode in ['ROTATE', 'SCALE'] and self.axis_constraint is not None:
            axis_str = f"Local {axis_str}"
        snap_str = " [SNAP]" if (pygame.key.get_mods() & KMOD_CTRL) else ""
        pygame.display.set_caption(f"Mode: {self.transform_mode} | Constraint: {axis_str}{snap_str} | [L-Click] Confirm")

    def handle_transform(self, dx, dy):
        if not self.active_cuboid: return
        
        self.accumulated_mouse[0] += dx
        self.accumulated_mouse[1] += dy

        total_dx, total_dy = self.accumulated_mouse
        
        snapping = (pygame.key.get_mods() & KMOD_CTRL)
        dist_factor = self.camera.distance / 15.0
        
        # Reset to backup state first
        self.active_cuboid.restore_state()
        
        # Determine SNAP step size
        snap_step = 0
        if self.transform_mode == 'GRAB': snap_step = 1.0
        elif self.transform_mode == 'ROTATE': snap_step = 45.0
        elif self.transform_mode == 'SCALE': snap_step = 0.5

        if self.transform_mode == 'GRAB':
            right = self.camera.get_right_vector()
            up = self.camera.get_up_vector()
            raw_move = (right * total_dx * 0.015 * dist_factor) + (up * -total_dy * 0.015 * dist_factor)
            
            if self.axis_constraint is not None:
                val = 0
                if self.axis_constraint == 0: val = raw_move[0] * 2 
                elif self.axis_constraint == 2: val = raw_move[2] * 2
                elif self.axis_constraint == 1: val = -total_dy * 0.02 * dist_factor 
                self.active_cuboid.position[self.axis_constraint] += val
            else:
                self.active_cuboid.position += raw_move
            
            # Snap Position (Only restricted axis if constrained)
            if snapping:
                for i in range(3):
                    if self.axis_constraint is None or self.axis_constraint == i:
                        self.active_cuboid.position[i] = self.snap(self.active_cuboid.position[i], snap_step)

        elif self.transform_mode == 'ROTATE':
            rot_delta = total_dx * 0.5
            
            if self.axis_constraint is not None:
                self.active_cuboid.rotation[self.axis_constraint] += rot_delta
            else:
                self.active_cuboid.rotation[1] += total_dx * 0.5
                self.active_cuboid.rotation[0] += total_dy * 0.5
            
            # Snap Rotation (Only restricted axis if constrained)
            if snapping:
                for i in range(3):
                    if self.axis_constraint is None or self.axis_constraint == i:
                        self.active_cuboid.rotation[i] = self.snap(self.active_cuboid.rotation[i], snap_step)

        elif self.transform_mode == 'SCALE':
            scale_delta = total_dx * 0.01
            
            if self.axis_constraint is not None:
                self.active_cuboid.scale[self.axis_constraint] += scale_delta
            else:
                self.active_cuboid.scale += scale_delta
            
            # Snap Scale (Only restricted axis if constrained)
            if snapping:
                for i in range(3):
                    if self.axis_constraint is None or self.axis_constraint == i:
                        self.active_cuboid.scale[i] = self.snap(self.active_cuboid.scale[i], snap_step)
            
            # Enforce minimum scale AFTER snapping to prevent 0 scale
            self.active_cuboid.scale = np.maximum(self.active_cuboid.scale, [0.01, 0.01, 0.01])

        self.update_title()

    def confirm_transform(self):
        self.save_history_state()
        self.transform_mode = None; self.update_title()

    def cancel_transform(self):
        if self.active_cuboid: self.active_cuboid.restore_state()
        self.transform_mode = None; self.update_title()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        last_mouse = pygame.mouse.get_pos()
        self.update_title()

        while running:
            mx, my = pygame.mouse.get_pos()
            dx, dy = mx - last_mouse[0], my - last_mouse[1]
            last_mouse = (mx, my)

            for event in pygame.event.get():
                if event.type == QUIT: running = False
                if event.type == KEYDOWN:
                    mods = pygame.key.get_mods()
                    if event.key == K_z and (mods & KMOD_CTRL):
                        if (mods & KMOD_SHIFT): self.redo()
                        else: self.undo()
                    if event.key == K_ESCAPE:
                        if self.transform_mode: self.cancel_transform()
                    
                    if self.active_cuboid and not self.transform_mode:
                        if event.key in [K_g, K_r, K_s]:
                            self.active_cuboid.store_state()
                            self.accumulated_mouse = [0, 0]; self.axis_constraint = None
                            if event.key == K_g: self.transform_mode = 'GRAB'
                            elif event.key == K_r: self.transform_mode = 'ROTATE'
                            elif event.key == K_s: self.transform_mode = 'SCALE'
                        elif event.key == K_x: 
                            self.save_history_state()
                            self.cuboids.remove(self.active_cuboid); self.active_cuboid = None
                    
                    if self.transform_mode:
                        if event.key == K_x: self.axis_constraint = 0
                        elif event.key == K_y: self.axis_constraint = 1
                        elif event.key == K_z: self.axis_constraint = 2
                        elif event.key == K_c: self.axis_constraint = None

                    if event.key == K_a and (mods & KMOD_SHIFT):
                        self.save_history_state()
                        pos = self.active_cuboid.position.copy() + [2,0,0] if self.active_cuboid else [0,2,0]
                        self.active_cuboid = Cuboid(*pos, color=tuple(np.random.rand(3)))
                        self.cuboids.append(self.active_cuboid)
                        self.transform_mode = 'GRAB'; self.active_cuboid.store_state(); self.accumulated_mouse = [0, 0]; self.axis_constraint = None
                    
                    if event.key == K_s and (mods & KMOD_CTRL): 
                        with open("scene.json", "w") as f: json.dump([c.to_dict() for c in self.cuboids], f, indent=4)
                        print("Saved.")
                    if event.key == K_l and (mods & KMOD_CTRL):
                        if os.path.exists("scene.json"):
                            self.save_history_state()
                            with open("scene.json", "r") as f: self.cuboids = [Cuboid.from_dict(d) for d in json.load(f)]
                            self.active_cuboid = None; print("Loaded.")

                    if event.key == K_KP1: self.camera.yaw=45; self.camera.pitch=10; self.camera.target=np.zeros(3)
                    if event.key == K_KP3: self.camera.yaw=-45; self.camera.pitch=10; self.camera.target=np.zeros(3)
                    if event.key == K_KP7: self.camera.yaw=0; self.camera.pitch=89.9; self.camera.target=np.zeros(3)
                    if event.key == K_KP5: self.camera.ortho_mode = not self.camera.ortho_mode

                if event.type == MOUSEBUTTONDOWN:
                    if self.transform_mode:
                        if event.button == 1: self.confirm_transform()
                        if event.button == 3: self.cancel_transform()
                    else:
                        if event.button == 1: self.active_cuboid = self.pick_object(mx, my)
                        if event.button == 4: self.camera.distance = max(1, self.camera.distance - 2)
                        if event.button == 5: self.camera.distance += 2

            if pygame.mouse.get_pressed()[1]: 
                self.camera.update(dx, dy, 'PAN' if (pygame.key.get_mods() & KMOD_SHIFT) else 'ORBIT')
            
            if self.transform_mode: self.handle_transform(dx, dy)

            glViewport(0, 0, SCREEN_SIZE[0], SCREEN_SIZE[1])
            glClearColor(0.15, 0.15, 0.15, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.camera.set_projection(); glLoadIdentity(); self.camera.apply()
            self.draw_scene(picking=False)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    Engine().run()