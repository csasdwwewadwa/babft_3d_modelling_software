import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys
import os

# --- CONFIGURATION ---
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimization
        self.iterations = 10000
        self.learning_rate = 0.015
        
        # Lifecycle (Progressive Growth)
        self.init_cuboids = 128      # Start with more
        self.max_cuboids = 2000      # Cap
        self.split_interval = 200    # How often to restructure
        
        # Loss Weights (TUNED for Tighter Fit)
        self.w_surface = 1.0         # Pull to surface
        self.w_normal = 0.5          # Align rotations (Increased)
        self.w_empty = 2.0           # CRITICAL: Push out of empty space (Increased 20x)
        self.w_size_reg = 0.02       # CRITICAL: Penalize volume (Increased 20x)
        
        # Temperature (SoftMin annealing)
        self.temp_start = 0.05       # Start sharper
        self.temp_end = 0.001
        
        # Sampling
        self.n_surf_points = 60000   # Mesh surface samples
        
        # Topology Thresholds
        self.thresh_big = 0.15       # If > 15% of unit cube, it's "Too Big" -> Kill it
        self.thresh_small = 0.005    # If < 0.5%, it's junk -> Kill it

# --- MATH UTILS ---
def quat_to_mat(quats):
    """Convert quaternions (N, 4) to rotation matrices (N, 3, 3)."""
    quats = quats / torch.norm(quats, dim=1, keepdim=True)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    rot = torch.empty((quats.shape[0], 3, 3), device=quats.device)
    rot[:, 0, 0] = 1 - 2 * (yy + zz)
    rot[:, 0, 1] = 2 * (xy - wz)
    rot[:, 0, 2] = 2 * (xz + wy)
    rot[:, 1, 0] = 2 * (xy + wz)
    rot[:, 1, 1] = 1 - 2 * (xx + zz)
    rot[:, 1, 2] = 2 * (yz - wx)
    rot[:, 2, 0] = 2 * (xz - wy)
    rot[:, 2, 1] = 2 * (yz + wx)
    rot[:, 2, 2] = 1 - 2 * (xx + yy)
    return rot

def transform_points_to_local(points, centers, rot_mats):
    """Vectorized World -> Local transformation."""
    # points: (P, 3), centers: (C, 3), rot_mats: (C, 3, 3)
    diff = points.unsqueeze(1) - centers.unsqueeze(0) # (P, C, 3)
    rot_mats_T = rot_mats.transpose(1, 2)
    p_local = torch.einsum('pci,cji->pcj', diff, rot_mats_T)
    return p_local

def ud_box(p_local, extents):
    """Unsigned Distance to Box in local coords."""
    if extents.dim() == 2:
        extents = extents.unsqueeze(0)
    d = torch.abs(p_local) - extents
    outside_dist = torch.norm(torch.clamp(d, min=0.0), dim=2)
    inside_dist = torch.min(torch.max(d, dim=2)[0], torch.zeros_like(d[:,:,0]))
    return outside_dist + torch.abs(inside_dist)

# --- MODEL ---
class CuboidModel(nn.Module):
    def __init__(self, cfg, mesh_bounds):
        super().__init__()
        self.cfg = cfg
        
        self.n_cuboids = cfg.init_cuboids
        
        # Initialize randomly within bounds
        center_init = torch.rand(self.n_cuboids, 3, device=cfg.device)
        center_init = center_init * (mesh_bounds[1] - mesh_bounds[0]) + mesh_bounds[0]
        self.centers = nn.Parameter(center_init)
        
        # Rotations: Randomly perturbed identity
        quat_init = torch.zeros(self.n_cuboids, 4, device=cfg.device)
        quat_init[:, 0] = 1.0 
        quat_init += torch.randn_like(quat_init) * 0.1
        self.quats = nn.Parameter(quat_init)
        
        # Extents: Initialize SMALL to prevent giant blobs
        # 1/50th of the mesh size
        scale_init = (mesh_bounds[1] - mesh_bounds[0]) / 50.0
        self.log_extents = nn.Parameter(torch.log(torch.ones(self.n_cuboids, 3, device=cfg.device) * scale_init))

    def get_params(self):
        extents = torch.exp(self.log_extents)
        rot_mats = quat_to_mat(self.quats)
        return self.centers, rot_mats, extents

    def add_cuboids(self, new_centers, new_scale):
        with torch.no_grad():
            self.centers = nn.Parameter(torch.cat([self.centers, new_centers], dim=0))
            
            new_quats = torch.zeros(len(new_centers), 4, device=self.cfg.device)
            new_quats[:, 0] = 1.0
            self.quats = nn.Parameter(torch.cat([self.quats, new_quats], dim=0))
            
            new_logs = torch.log(torch.ones(len(new_centers), 3, device=self.cfg.device) * new_scale)
            self.log_extents = nn.Parameter(torch.cat([self.log_extents, new_logs], dim=0))
            self.n_cuboids = self.centers.shape[0]

    def remove_cuboids(self, mask_keep):
        if mask_keep.sum() == 0: return 
        with torch.no_grad():
            self.centers = nn.Parameter(self.centers[mask_keep])
            self.quats = nn.Parameter(self.quats[mask_keep])
            self.log_extents = nn.Parameter(self.log_extents[mask_keep])
            self.n_cuboids = self.centers.shape[0]

# --- PIPELINE ---
class ProgressiveMesher:
    def __init__(self, mesh_path):
        self.cfg = Config()
        
        print(f"Loading: {mesh_path}")
        self.mesh = trimesh.load(mesh_path, force='mesh')
        
        # Normalize
        self.mesh_center = self.mesh.bounding_box.centroid
        self.mesh_scale = np.max(self.mesh.bounding_box.extents)
        self.mesh.apply_translation(-self.mesh_center)
        self.mesh.apply_scale(1.0 / self.mesh_scale)
        
        # Surface Sampling (Target)
        print(f"Sampling {self.cfg.n_surf_points} surface points...")
        pts, face_indices = trimesh.sample.sample_surface(self.mesh, self.cfg.n_surf_points)
        self.surf_points = torch.tensor(pts, dtype=torch.float32, device=self.cfg.device)
        self.surf_normals = torch.tensor(self.mesh.face_normals[face_indices], dtype=torch.float32, device=self.cfg.device)
        
        # UDF Proxy (For Empty Space Loss)
        print("Building Distance Field Proxy (KDTree)...")
        # We sample vertices + some edge midpoints for robustness
        # Actually just vertices is usually enough if mesh is dense, otherwise subdivision helps.
        # For this script, we assume vertices are enough proxy for 'inside/outside' checking on surface.
        self.kdtree = cKDTree(self.mesh.vertices)
        
        # Init Model
        bounds = (torch.tensor([-0.5]*3, device=self.cfg.device), 
                  torch.tensor([0.5]*3, device=self.cfg.device))
        self.model = CuboidModel(self.cfg, bounds)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

    def get_current_temperature(self, step):
        alpha = min(1.0, step / self.cfg.iterations)
        return self.cfg.temp_start * (1 - alpha) + self.cfg.temp_end * alpha

    def compute_empty_space_loss(self, centers, rot_mats, extents):
        """
        Heuristic: Sample points on cuboid faces. If they are far from mesh surface, 
        the cuboid is bloating into empty space.
        """
        # Increase subset size for better stability
        n_sample = min(self.model.n_cuboids, 400)
        indices = torch.randperm(self.model.n_cuboids)[:n_sample]
        
        c_sub = centers[indices]
        r_sub = rot_mats[indices]
        e_sub = extents[indices]
        
        # Sample K points per cuboid face
        K = 40
        raw = torch.rand(n_sample, K, 3, device=self.cfg.device) * 2.0 - 1.0
        # Project to surface of unit cube
        max_val, _ = torch.max(torch.abs(raw), dim=2, keepdim=True)
        raw = raw / max_val 
        
        # Transform to world
        scaled = raw * e_sub.unsqueeze(1)
        rotated = torch.einsum('sji,ski->skj', r_sub, scaled)
        p_world = c_sub.unsqueeze(1) + rotated
        
        # Query CPU KDTree
        p_cpu = p_world.reshape(-1, 3).detach().cpu().numpy()
        dists, _ = self.kdtree.query(p_cpu, k=1) 
        dists_torch = torch.tensor(dists, device=self.cfg.device, dtype=torch.float32)
        
        # Aggressive penalty for anything > 2% off surface
        loss = torch.mean(torch.relu(dists_torch - 0.02))
        return loss

    def manage_topology(self, dists_matrix, extents, step):
        """
        Lifecycle management: Kill big/useless cuboids, spawn small ones at errors.
        """
        if self.model.n_cuboids >= self.cfg.max_cuboids:
            return

        with torch.no_grad():
            # Metrics
            min_vals, min_idx = torch.min(dists_matrix, dim=1)
            counts = torch.bincount(min_idx, minlength=self.model.n_cuboids)
            max_dim = torch.max(extents, dim=1)[0]
            
            # --- 1. Identify "Bad" Cuboids ---
            
            # A. Too Small (Degenerate)
            is_too_small = max_dim < self.cfg.thresh_small
            
            # B. Unused (Occluded or redundant)
            # Threshold: If it covers less than 1/10th of expected average points
            expected_avg = self.cfg.n_surf_points / self.model.n_cuboids
            is_unused = counts < (expected_avg * 0.1)
            
            # C. Too BIG (The "Giant Box" fix)
            # If a cuboid is > 15% of the total scene size, it's too abstract. Kill it.
            is_too_big = max_dim > self.cfg.thresh_big
            
            # Kill mask: Small OR Unused OR Big
            # We kill Big ones so they get replaced by clusters of Small ones
            kill_mask = ~(is_too_small | is_unused | is_too_big)
            
            # Safety: Always keep at least 10 cuboids
            if kill_mask.sum() < 10: 
                kill_mask[:] = True 
                
            # --- 2. Identify Spawn Locations ---
            
            # A. High Error Regions
            error_threshold = torch.mean(min_vals) * 1.5
            high_error_mask = min_vals > error_threshold
            
            # B. Regions formerly owned by "Too Big" cuboids
            # If we are killing a big cuboid, we must spawn new ones in its territory
            big_cuboid_indices = torch.nonzero(is_too_big).squeeze(1)
            if len(big_cuboid_indices) > 0:
                points_in_big_zones = torch.isin(min_idx, big_cuboid_indices)
                # Combine masks
                spawn_candidate_mask = high_error_mask | points_in_big_zones
            else:
                spawn_candidate_mask = high_error_mask
                
            candidate_points = self.surf_points[spawn_candidate_mask]
            
            # Cap spawn rate
            n_slots = self.cfg.max_cuboids - kill_mask.sum()
            n_spawn = min(40, n_slots)
            
            if len(candidate_points) > n_spawn and n_spawn > 0:
                perm = torch.randperm(len(candidate_points))[:n_spawn]
                new_centers = candidate_points[perm]
                
                # Apply Removal
                self.model.remove_cuboids(kill_mask)
                
                # Apply Addition (Spawn SMALL)
                self.model.add_cuboids(new_centers, new_scale=0.03)
            else:
                # Just removal
                self.model.remove_cuboids(kill_mask)

    def train(self):
        pbar = tqdm(range(self.cfg.iterations))
        
        for step in pbar:
            self.optimizer.zero_grad()
            
            centers, rot_mats, extents = self.model.get_params()
            temp = self.get_current_temperature(step)
            
            # 1. Distances
            p_local = transform_points_to_local(self.surf_points, centers, rot_mats)
            dists = ud_box(p_local, extents) # (P, C)
            
            # 2. Surface Loss (SoftMin)
            # Stability: Subtract min for numerical safety in exp
            min_dists, min_indices = torch.min(dists, dim=1) 
            # Standard SoftMin
            weights = torch.softmax(-dists / temp, dim=1)
            loss_surf = torch.sum(weights * dists, dim=1).mean()
            
            # 3. Normal Alignment
            # We want cuboid normals to align with mesh normals
            closest_rot = rot_mats[min_indices]
            n_mesh = self.surf_normals
            # Rotate mesh normal to local space
            n_local = torch.einsum('pij,pi->pj', closest_rot.transpose(1,2), n_mesh)
            # Maximize the dominant component (L-infinity norm approx)
            loss_norm = 1.0 - torch.mean(torch.max(torch.abs(n_local), dim=1)[0])
            
            # 4. Empty Space
            loss_empty = self.compute_empty_space_loss(centers, rot_mats, extents)
            
            # 5. Size Reg (Force small cuboids)
            loss_size = torch.mean(torch.sum(extents**2, dim=1))
            
            # Total
            total_loss = (self.cfg.w_surface * loss_surf +
                          self.cfg.w_normal * loss_norm + 
                          self.cfg.w_empty * loss_empty + 
                          self.cfg.w_size_reg * loss_size)
            
            total_loss.backward()
            self.optimizer.step()
            
            # Topology Management
            if step > 0 and step % self.cfg.split_interval == 0:
                self.manage_topology(dists, extents, step)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

            # Logging
            if step % 10 == 0:
                pbar.set_description(f"Cubs: {self.model.n_cuboids} | Surf: {loss_surf.item():.4f} | Emp: {loss_empty.item():.4f}")

    def export_obj(self, filename="output.obj"):
        print("Exporting...")
        centers, rot_mats, extents = self.model.get_params()
        centers = centers.detach().cpu().numpy()
        rot_mats = rot_mats.detach().cpu().numpy()
        extents = extents.detach().cpu().numpy()
        
        final_mesh = trimesh.Trimesh()
        
        for i in range(len(centers)):
            box = trimesh.creation.box(extents=extents[i]*2.0)
            matrix = np.eye(4)
            matrix[:3, :3] = rot_mats[i]
            matrix[:3, 3] = centers[i]
            box.apply_transform(matrix)
            final_mesh += box
            
        final_mesh.apply_scale(self.mesh_scale)
        final_mesh.apply_translation(self.mesh_center)
        final_mesh.export(filename)
        print(f"Saved to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cuboids.py <mesh.obj>")
        print("Generating dummy torus...")
        mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.5)
        mesh.export("dummy.obj")
        mesh_path = "dummy.obj"
    else:
        mesh_path = sys.argv[1]

    mesher = ProgressiveMesher(mesh_path)
    mesher.train()
    mesher.export_obj("result_cuboids.obj")