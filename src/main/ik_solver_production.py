#!/usr/bin/env python3
"""
Damped Jacobian IK Solver - Production Version
Validated with Yaskawa GP8 robot in CoppeliaSim
CRITICAL FIX: J3 and J5 have opposite rotation directions in CoppeliaSim
"""

import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# DH Parameters (Yaskawa GP8)
DH_PARAMS = {
    1: {'alpha': -np.pi/2, 'a': 0.0400, 'd': 0.3300, 'offset': 0},
    2: {'alpha': 0.0, 'a': 0.3450, 'd': 0.0, 'offset': -np.pi/2},
    3: {'alpha': -np.pi/2, 'a': 0.0400, 'd': 0.0, 'offset': 0},
    4: {'alpha': np.pi/2, 'a': 0.0, 'd': 0.3400, 'offset': 0},
    5: {'alpha': -np.pi/2, 'a': 0.0, 'd': 0.0, 'offset': 0},
    6: {'alpha': 0.0, 'a': 0.0, 'd': 0.2413, 'offset': 0},
}

BASE_OFFSET = np.array([0.0187, 0, 0])

# Joint limits (rad)
Q_MIN = np.array([-2.97, -1.75, -3.14, -3.49, -2.09, -6.28])
Q_MAX = np.array([ 2.97,  2.62,  1.22,  3.49,  2.09,  6.28])

# IK Parameters
LAMBDA_DAMP = 0.01        # Damping factor
QD_MAX = 3.5              # Max angular velocity (rad/s)
MAX_ITERATIONS = 200
CONVERGENCE_THRESHOLD = 0.005  # 5mm

def dh_matrix(alpha, a, d, theta):
    """DH transformation matrix"""
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
    """Forward kinematics using DH convention
    
    Args:
        angles: Joint angles in radians (relative to DH theta offsets)
    
    Returns:
        4x4 transformation matrix T06 (World -> EE)
    
    NOTE: This function works in standard DH space.
          When reading/writing J3 and J5 from/to CoppeliaSim,
          negate them (opposite sign convention in sim).
    """
    T_Base = np.array([[1, 0, 0, 0.0187], [0, 1, 0, 0], 
                       [0, 0, 1, 0], [0, 0, 0, 1]])
    
    T01 = dh_matrix(-np.pi/2, 0.040, 0.330,  angles[0])
    T12 = dh_matrix(0,        0.345, 0,      angles[1] - np.pi/2)
    T23 = dh_matrix(-np.pi/2, 0.040, 0,      angles[2])
    T34 = dh_matrix(np.pi/2,  0,     0.340,  angles[3])
    T45 = dh_matrix(-np.pi/2, 0,     0,      angles[4])
    T56 = dh_matrix(0,        0,     0.2413, angles[5])
    
    return T_Base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56

def damped_ik(target_pos, current_angles, joint_handles, sim, ee_handle, 
              max_iterations=MAX_ITERATIONS, verbose=True):
    """Damped Jacobian IK using pseudo-inverse
    
    Args:
        target_pos: Target EE position [x, y, z] (World frame)
        current_angles: Initial joint angles
        joint_handles: List of joint object handles
        sim: CoppeliaSim API object
        ee_handle: End effector handle
        max_iterations: Max iterations (default 200)
        verbose: Print progress (default True)
    
    Returns:
        (converged, best_angles, best_error)
    """
    current_angles = current_angles.copy()
    best_angles = current_angles.copy()
    best_error = float('inf')
    
    for iteration in range(max_iterations):
        # Compute FK at current position
        T_ee = compute_fk_dh(current_angles)
        ee_pos = T_ee[:3, 3]
        
        # Compute error
        error = target_pos - ee_pos
        error_norm = np.linalg.norm(error)
        
        if error_norm < best_error:
            best_error = error_norm
            best_angles = current_angles.copy()
        
        if verbose and (iteration % 50 == 0 or iteration < 5 or error_norm < 0.01):
            print(f"Iter {iteration:3d}: error={error_norm*1000:7.3f}mm  "
                  f"q=[{current_angles[0]:6.2f}, {current_angles[1]:6.2f}, "
                  f"{current_angles[2]:6.2f}]")
        
        if error_norm < CONVERGENCE_THRESHOLD:
            if verbose:
                print(f"\n✓ CONVERGED in {iteration+1} iterations\n")
            return (True, best_angles, best_error)
        
        # Compute Jacobian via finite differences
        step = 1e-6
        J = np.zeros((3, 6))
        for i in range(6):
            angles_pert = current_angles.copy()
            angles_pert[i] += step
            T_pert = compute_fk_dh(angles_pert)
            dp = (T_pert[:3, 3] - ee_pos) / step
            J[:, i] = dp
        
        # Damped pseudo-inverse
        try:
            J_pseudo = J.T @ np.linalg.inv(J @ J.T + LAMBDA_DAMP * np.eye(3))
        except np.linalg.LinAlgError:
            if verbose:
                print(f"[!] Singular Jacobian at iteration {iteration}")
            return (False, best_angles, best_error)
        
        # Update angles
        dq = J_pseudo @ error * 0.5
        
        # Limit velocity
        max_dq = np.max(np.abs(dq))
        if max_dq > QD_MAX:
            dq = dq * (QD_MAX / max_dq)
        
        current_angles = current_angles + dq
        current_angles = np.clip(current_angles, Q_MIN, Q_MAX)
        
        # Apply to simulation (negate J3 and J5 per CoppeliaSim convention)
        angles_sim = current_angles.copy()
        angles_sim[2] *= -1  # J3 opposite direction
        angles_sim[4] *= -1  # J5 opposite direction
        for i, h in enumerate(joint_handles):
            sim.setJointPosition(h, angles_sim[i])
        
        # Step simulation
        for _ in range(5):
            sim.step()
    
    if verbose:
        print(f"\n[!] Did not converge after {max_iterations} iterations")
    return (False, best_angles, best_error)

def solve_ik(target_pos, joint_handles, sim, ee_handle):
    """Solve IK and apply to robot"""
    # Get current position (negate J3/J5 when reading from sim)
    angles_sim = np.array([sim.getJointPosition(h) for h in joint_handles])
    angles_dh = angles_sim.copy()
    angles_dh[2] *= -1  # J3
    angles_dh[4] *= -1  # J5
    
    # Solve IK
    converged, best_angles, error = damped_ik(target_pos, angles_dh, 
                                              joint_handles, sim, ee_handle)
    
    if converged:
        # Apply best solution (negate J3/J5 for sim)
        best_angles_sim = best_angles.copy()
        best_angles_sim[2] *= -1
        best_angles_sim[4] *= -1
        for i, h in enumerate(joint_handles):
            sim.setJointPosition(h, best_angles_sim[i])
        
        for _ in range(50):
            sim.step()
        
        # Verify final position
        m_ee = sim.getObjectMatrix(ee_handle, sim.handle_world)
        final_ee = np.array([m_ee[3], m_ee[7], m_ee[11]])
        final_error = np.linalg.norm(final_ee - target_pos)
        
        print(f"\n[RESULT]")
        print(f"Target:      {target_pos}")
        print(f"EE (Sim):    {final_ee}")
        print(f"Error:       {final_error*1000:.1f} mm")
        print(f"Success:     {'✓ YES' if final_error < 0.01 else '✗ NO'}")
        
        return (converged, best_angles, final_error)
    else:
        print(f"\n[FAILED] IK did not converge (best error: {error*1000:.1f}mm)")
        return (False, best_angles, error)

if __name__ == "__main__":
    # Test with grab point from trajectory
    try:
        client = RemoteAPIClient()
        sim = client.require('sim')
        sim.startSimulation()
        
        joint_handles = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
        ee_handle = sim.getObject('/yaskawa/gripperEF')
        
        # Load grab point
        data = np.load('trajectory_cup_data.npz')
        grab_step = int(data['grab_step'])
        grab_pos = np.array([data['px'][grab_step], data['py'][grab_step], 
                            data['pz'][grab_step]])
        
        print(f"Target grab position: {grab_pos}")
        solve_ik(grab_pos, joint_handles, sim, ee_handle)
        
        sim.stopSimulation()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
