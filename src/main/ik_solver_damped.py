#!/usr/bin/env python3
"""
IK Jacobian Solver with Lambda Damping
Uses validated DH parameters from TF Matrix.py
"""

import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================
# DH Parameters (from TF Matrix.py - VERIFIED)
# ==========================================
DH_PARAMS = {
    1: {'alpha': -np.pi/2, 'a': 0.0400, 'd': 0.3300, 'offset': 0},
    2: {'alpha': 0.0, 'a': 0.3450, 'd': 0.0, 'offset': -np.pi/2},
    3: {'alpha': -np.pi/2, 'a': 0.0400, 'd': 0.0, 'offset': 0},
    4: {'alpha': np.pi/2, 'a': 0.0, 'd': 0.3400, 'offset': 0},
    5: {'alpha': -np.pi/2, 'a': 0.0, 'd': 0.0, 'offset': 0},
    6: {'alpha': 0.0, 'a': 0.0, 'd': 0.2413, 'offset': 0},
}

BASE_OFFSET = np.array([0.0187, 0, 0])

# ==========================================
# Joint Limits & Velocity Limits
# ==========================================
QD_MAX = 3.5  # max joint velocity (rad/s) -- prevents singularity explosions

# Joint limits (rad) - Yaskawa GP8 (from CoppeliaSim model properties)
Q_MIN = np.array([-2.97, -1.75, -3.14, -3.49, -2.09, -6.28])
Q_MAX = np.array([ 2.97,  2.62,  1.22,  3.49,  2.09,  6.28])


def dh_matrix(alpha, a, d, theta):
    """Standard Denavit-Hartenberg transformation"""
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)
    
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])


def compute_fk_dh(angles):
    """Compute FK using DH parameters (like TF Matrix.py)"""
    # Base transformation
    T_Base = np.array([
        [1, 0, 0, 0.0187],
        [0, 1, 0, 0.0000],
        [0, 0, 1, 0.0000],
        [0, 0, 0, 1.0000]
    ])
    
    # DH transformations
    T01 = dh_matrix(-np.pi/2, 0.040, 0.330,  angles[0])
    T12 = dh_matrix(0,        0.345, 0,      angles[1] - np.pi/2)  # KEY: q2 offset!
    T23 = dh_matrix(-np.pi/2, 0.040, 0,      angles[2])
    T34 = dh_matrix(np.pi/2,  0,     0.340,  angles[3])
    T45 = dh_matrix(-np.pi/2, 0,     0,      angles[4])
    T56 = dh_matrix(0,        0,     0.2413, angles[5])
    
    # Chain multiplication
    T = T_Base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56
    return T


def extract_position(T):
    """Extract XYZ position from transformation matrix"""
    return T[:3, 3]


def calculate_jacobian(angles, step_size=1e-6):
    """Compute geometric Jacobian using finite differences"""
    J = np.zeros((3, 6))
    
    base_fk = compute_fk_dh(angles)
    base_pos = extract_position(base_fk)
    
    for i in range(6):
        angles_perturb = angles.copy()
        angles_perturb[i] += step_size
        
        perturb_fk = compute_fk_dh(angles_perturb)
        perturb_pos = extract_position(perturb_fk)
        
        J[:, i] = (perturb_pos - base_pos) / step_size
    
    return J


class IKSolverDamped:
    def __init__(self, client):
        self.client = client
        self.sim = client.require('sim')
        
        self.robot_base_obj = self.sim.getObject('/yaskawa')
        
        try:
            self.joint_handles = [self.sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
        except:
            self.joint_handles = [self.sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]
        
        try:
            self.ee_handle = self.sim.getObject('/yaskawa/gripperEF')
        except:
            self.ee_handle = self.sim.getObjectHandle('/yaskawa/gripperEF')
        
        # Get robot base frame for coordinate transformation
        self.robot_base_pos = np.array(self.sim.getObjectPosition(self.robot_base_obj, self.sim.handle_world))
        self.robot_base_ori = np.array(self.sim.getObjectOrientation(self.robot_base_obj, self.sim.handle_world))
    
    def get_ee_position_world(self):
        """Get EE position in world frame"""
        m = self.sim.getObjectMatrix(self.ee_handle, self.sim.handle_world)
        return np.array([m[3], m[7], m[11]])
    
    def world_to_base_frame(self, pos_world):
        """Transform position from world frame to robot base frame using transformation matrix"""
        # Get robot base transformation matrix (world → base)
        m_base = self.sim.getObjectMatrix(self.robot_base_obj, self.sim.handle_world)
        
        # Create 4x4 transformation matrix
        T_base = np.array([
            [m_base[0], m_base[1], m_base[2],  m_base[3]],
            [m_base[4], m_base[5], m_base[6],  m_base[7]],
            [m_base[8], m_base[9], m_base[10], m_base[11]],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Get inverse (world ← base)
        T_inv = np.linalg.inv(T_base)
        
        # Transform world point to base frame
        p_world_h = np.array([pos_world[0], pos_world[1], pos_world[2], 1.0])
        p_base_h = T_inv @ p_world_h
        
        return p_base_h[:3]
    
    def get_joint_angles_sim(self):
        """Get current joint angles from simulation"""
        return np.array([self.sim.getJointPosition(h) for h in self.joint_handles])
    
    def set_joint_angles(self, angles):
        """Set joint angles using direct position control"""
        for i, h in enumerate(self.joint_handles):
            self.sim.setJointPosition(h, angles[i])
    
    def solve_ik(self, target_pos, max_iterations=500, tolerance=0.005, 
                 lambda_damp=0.01, step_size=0.5):
        """
        Solve IK using Jacobian with lambda damping
        
        target_pos: target EE position in ROBOT BASE FRAME [x, y, z]
        max_iterations: max solver iterations
        tolerance: convergence tolerance (m)
        lambda_damp: damping factor (higher = smoother but slower)
        step_size: velocity scaling factor
        
        Features:
        - Velocity limiting with QD_MAX to prevent singularity explosions
        - Joint angle clamping within hardware limits (Q_MIN, Q_MAX)
        - Damped pseudo-inverse for numerical stability
        
        Returns: (success, converged_angles)
        """
        
        print(f"Target (BASE FRAME): {target_pos}")
        print(f"Joint limits: Q_MIN={Q_MIN}, Q_MAX={Q_MAX}")
        print(f"Tolerance: {tolerance*1000:.2f} mm")
        print(f"Lambda damping: {lambda_damp}")
        print("-" * 80)
        
        # Get initial angles
        initial_angles = self.get_joint_angles_sim()
        current_angles = initial_angles.copy()
        converged_angles = current_angles.copy()
        best_error = float('inf')
        
        for iteration in range(max_iterations):
            # Compute current FK position
            T_ee = compute_fk_dh(current_angles)
            ee_pos = extract_position(T_ee)
            
            # Position error
            error = target_pos - ee_pos
            error_norm = np.linalg.norm(error)
            
            # Track best solution
            if error_norm < best_error:
                best_error = error_norm
                converged_angles = current_angles.copy()
            
            # Print progress
            if iteration % 50 == 0 or iteration < 10 or error_norm < tolerance * 2:
                print(f"Iter {iteration:3d}: error={error_norm*1000:7.3f}mm  "
                      f"q=[{current_angles[0]:6.2f}, {current_angles[1]:6.2f}, {current_angles[2]:6.2f}]")
            
            # Check convergence
            if error_norm < tolerance:
                print(f"\n✓ CONVERGED in {iteration+1} iterations")
                print(f"  Final error: {error_norm*1000:.3f} mm")
                
                # Apply final angles to simulation
                self.set_joint_angles(current_angles)
                for _ in range(20):
                    self.sim.step()
                
                return True, current_angles
            
            # Calculate Jacobian
            J = calculate_jacobian(current_angles)
            
            # Damped pseudo-inverse: J_pseudo = J^T @ inv(J @ J^T + lambda * I)
            try:
                J_T = J.T
                J_J_T = J @ J_T
                J_damped = J_J_T + lambda_damp * np.eye(3)
                J_inv = np.linalg.inv(J_damped)
                
                # Check condition number to detect near-singularities
                cond = np.linalg.cond(J_J_T)
                if cond > 100:
                    print(f"  [!] Warning: Near singularity detected (condition={cond:.1f}) at iteration {iteration}")
                
                J_pseudo = J_T @ J_inv
            except np.linalg.LinAlgError:
                print(f"[!] Singular matrix at iteration {iteration} - returning best solution found")
                self.set_joint_angles(converged_angles)
                for _ in range(50):
                    self.sim.step()
                return False, converged_angles
            
            # Calculate joint update
            dq = J_pseudo @ error * step_size
            
            # Velocity limiting - prevent singularity explosions
            max_dq = np.max(np.abs(dq))
            if max_dq > QD_MAX:
                dq = dq * (QD_MAX / max_dq)
            
            # Update angles
            current_angles = current_angles + dq
            
            # Clamp angles to joint limits
            current_angles = np.clip(current_angles, Q_MIN, Q_MAX)
            
            # Apply to simulation
            self.set_joint_angles(current_angles)
            for _ in range(10):
                self.sim.step()
                time.sleep(0.001)
        
        print(f"\n✗ Did not converge after {max_iterations} iterations")
        print(f"  Best error achieved: {best_error*1000:.3f} mm")
        
        # Apply best solution
        self.set_joint_angles(converged_angles)
        for _ in range(50):
            self.sim.step()
        
        return False, converged_angles


def main():
    print("="*80)
    print("IK JACOBIAN SOLVER - DAMPED VERSION WITH SINGULARITY PROTECTION")
    print("="*80)
    
    print("\n[SAFETY PARAMETERS]")
    print(f"  Max joint velocity (QD_MAX): {QD_MAX} rad/s")
    print(f"  Joint angle limits (Q_MIN):  {Q_MIN}")
    print(f"  Joint angle limits (Q_MAX):  {Q_MAX}")
    
    # Load trajectory data
    print("\n[LOADING TRAJECTORY DATA]")
    try:
        data = np.load('trajectory_cup_data.npz')
        cup_px = data['px']
        cup_py = data['py']
        cup_pz = data['pz']
        grab_step = int(data['grab_step'])
        
        grab_pos = np.array([cup_px[grab_step], cup_py[grab_step], cup_pz[grab_step]])
        
        print(f"  ✓ Loaded {len(cup_px)} trajectory points")
        print(f"  → Grab step: {grab_step}")
        print(f"  → Grab position: {grab_pos}")
    except FileNotFoundError:
        print("[✗] trajectory_cup_data.npz not found")
        print("    Using default test position")
        grab_pos = np.array([0.5, -0.5, 0.4])
    
    # Connect to sim
    try:
        client = RemoteAPIClient()
        sim = client.require('sim')
        sim.startSimulation()
        print("\n[✓] Connected to CoppeliaSim")
    except Exception as e:
        print(f"[✗] Connection failed: {e}")
        return
    
    time.sleep(0.5)
    
    solver = IKSolverDamped(client)
    
    print("\n[INITIAL STATE]")
    initial_angles = solver.get_joint_angles_sim()
    initial_ee = solver.get_ee_position_world()
    print(f"  EE Position: {initial_ee}")
    print(f"  Joint angles: {initial_angles}")
    
    print("\n" + "="*80)
    print("[SOLVING IK TO GRAB POSITION]")
    print("="*80 + "\n")
    
    # Transform grab position from world to robot base frame
    print("[COORDINATE FRAME TRANSFORMATION]")
    print(f"  Grab position (WORLD): {grab_pos}")
    print(f"  Robot base pos:        {solver.robot_base_pos}")
    print(f"  Robot base ori (deg):  {np.degrees(solver.robot_base_ori)}")
    
    grab_pos_base = solver.world_to_base_frame(grab_pos)
    print(f"  Grab position (BASE):  {grab_pos_base}")
    print()
    
    # Solve IK with lambda damping (using base frame position)
    success, ik_angles = solver.solve_ik(grab_pos_base, max_iterations=500, tolerance=0.005, 
                                         lambda_damp=0.01, step_size=0.5)
    
    print("\n[FINAL STATE]")
    
    # Apply IK angles and wait for convergence
    solver.set_joint_angles(ik_angles)
    for _ in range(100):
        solver.sim.step()
    
    final_ee_world = solver.get_ee_position_world()
    final_ee_base = solver.world_to_base_frame(final_ee_world)
    final_fk_base = extract_position(compute_fk_dh(ik_angles))
    
    print(f"  IK joint angles: {ik_angles}")
    print(f"\n  [World Frame] (for reference)")
    print(f"    EE Position (Sim):     {final_ee_world}")
    print(f"    Target Position (Sim): {grab_pos}")
    print(f"\n  [Robot Base Frame] (where IK computation happens)")
    print(f"    EE Position (Sim):     {final_ee_base}")
    print(f"    EE Position (FK):      {final_fk_base}")
    print(f"    Target Position (IK):  {grab_pos_base}")
    print(f"\n  [ERRORS]")
    print(f"    Error (FK-Target in BASE):    {np.linalg.norm(final_fk_base - grab_pos_base)*1000:.3f} mm")
    print(f"    Error (Sim-FK in BASE):       {np.linalg.norm(final_ee_base - final_fk_base)*1000:.3f} mm")
    print(f"    Error (Sim-Target in WORLDy): {np.linalg.norm(final_ee_world - grab_pos)*1000:.3f} mm")
    
    sim.stopSimulation()
    print("\n[✓] Simulation stopped")


if __name__ == '__main__':
    main()
