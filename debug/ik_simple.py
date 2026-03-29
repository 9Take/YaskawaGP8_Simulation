#!/usr/bin/env python3
"""
Simple IK Solver - World Frame (No Frame Transform)
Just like TF Matrix.py but with IK
"""

import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# DH Parameters
DH_PARAMS = {
    1: {'alpha': -np.pi/2, 'a': 0.0400, 'd': 0.3300, 'offset': 0},
    2: {'alpha': 0.0, 'a': 0.3450, 'd': 0.0, 'offset': -np.pi/2},
    3: {'alpha': -np.pi/2, 'a': 0.0400, 'd': 0.0, 'offset': 0},
    4: {'alpha': np.pi/2, 'a': 0.0, 'd': 0.3400, 'offset': 0},
    5: {'alpha': -np.pi/2, 'a': 0.0, 'd': 0.0, 'offset': 0},
    6: {'alpha': 0.0, 'a': 0.0, 'd': 0.2413, 'offset': 0},
}

BASE_OFFSET = np.array([0.0187, 0, 0])

# Joint limits
Q_MIN = np.array([-2.97, -1.75, -3.14, -3.49, -2.09, -6.28])
Q_MAX = np.array([ 2.97,  2.62,  1.22,  3.49,  2.09,  6.28])


def dh_matrix(alpha, a, d, theta):
    """DH transformation"""
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
    """FK using DH (standard)"""
    T_Base = np.array([[1, 0, 0, 0.0187], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T01 = dh_matrix(-np.pi/2, 0.040, 0.330,  angles[0])
    T12 = dh_matrix(0,        0.345, 0,      angles[1] - np.pi/2)
    T23 = dh_matrix(-np.pi/2, 0.040, 0,      angles[2])
    T34 = dh_matrix(np.pi/2,  0,     0.340,  angles[3])
    T45 = dh_matrix(-np.pi/2, 0,     0,      angles[4])
    T56 = dh_matrix(0,        0,     0.2413, angles[5])
    return T_Base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56


print("="*80)
print("SIMPLE IK TEST - WORLD FRAME")
print("="*80)

try:
    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.startSimulation()
    print("\n[✓] Connected\n")
    time.sleep(0.5)
except Exception as e:
    print(f"[✗] Error: {e}")
    exit()

# Get handles
try:
    joint_handles = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
except:
    joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]

try:
    ee_handle = sim.getObject('/yaskawa/gripperEF')
except:
    ee_handle = sim.getObjectHandle('/yaskawa/gripperEF')

# Reset to HOME
print("[RESET TO HOME]")
for h in joint_handles:
    sim.setJointPosition(h, 0.0)

for _ in range(50):
    sim.step()

angles_home = np.array([sim.getJointPosition(h) for h in joint_handles])
# Negate J3 and J5 (opposite directions in CoppeliaSim)
angles_home[2] *= -1  # J3
angles_home[4] *= -1  # J5
print(f"Joint angles: {angles_home}")

# Get home EE position
m_ee = sim.getObjectMatrix(ee_handle, sim.handle_world)
ee_home = np.array([m_ee[3], m_ee[7], m_ee[11]])
print(f"EE position (home): {ee_home}\n")

# Load grab position (WORLD FRAME)
try:
    data = np.load('trajectory_cup_data.npz')
    grab_step = int(data['grab_step'])
    grab_pos = np.array([data['px'][grab_step], data['py'][grab_step], data['pz'][grab_step]])
    print(f"[GRAB POINT]")
    print(f"  Step: {grab_step}")
    print(f"  Position (WORLD): {grab_pos}")
    print(f"  Distance from home: {np.linalg.norm(grab_pos - ee_home)*1000:.1f} mm\n")
except FileNotFoundError:
    grab_pos = np.array([0.5, -0.5, 0.4])
    print(f"Using default grab: {grab_pos}\n")

# Simple IK test
print("[IK SOLVER - WORLD FRAME]")
print(f"Target: {grab_pos}")
print("-" * 80)

current_angles = angles_home.copy()
best_error = float('inf')
best_angles = current_angles.copy()

for iteration in range(200):
    # Compute FK
    T_ee = compute_fk_dh(current_angles)
    ee_pos = T_ee[:3, 3]
    
    # Error
    error = grab_pos - ee_pos
    error_norm = np.linalg.norm(error)
    
    if error_norm < best_error:
        best_error = error_norm
        best_angles = current_angles.copy()
    
    if iteration % 50 == 0 or iteration < 5 or error_norm < 0.01:
        print(f"Iter {iteration:3d}: error={error_norm*1000:7.3f}mm  q=[{current_angles[0]:6.2f}, {current_angles[1]:6.2f}, {current_angles[2]:6.2f}]")
    
    if error_norm < 0.005:
        print(f"\n✓ CONVERGED in {iteration+1} iterations")
        break
    
    # Jacobian (finite diff)
    step = 1e-6
    J = np.zeros((3, 6))
    for i in range(6):
        a_pert = current_angles.copy()
        a_pert[i] += step
        T_pert = compute_fk_dh(a_pert)
        dp = (T_pert[:3, 3] - ee_pos) / step
        J[:, i] = dp
    
    # Damped pseudo-inverse
    lambda_damp = 0.01
    try:
        J_pseudo = J.T @ np.linalg.inv(J @ J.T + lambda_damp * np.eye(3))
    except:
        print(f"[!] Singular - using best")
        break
    
    # Update
    dq = J_pseudo @ error * 0.5
    
    # Limit velocity
    max_dq = np.max(np.abs(dq))
    if max_dq > 3.5:
        dq = dq * (3.5 / max_dq)
    
    current_angles = current_angles + dq
    current_angles = np.clip(current_angles, Q_MIN, Q_MAX)
    
    # Apply to sim (negate J3 and J5 per model config)
    angles_sim = current_angles.copy()
    angles_sim[2] *= -1  # J3
    angles_sim[4] *= -1  # J5
    for i, h in enumerate(joint_handles):
        sim.setJointPosition(h, angles_sim[i])
    
    for _ in range(5):
        sim.step()

print(f"\nBest error: {best_error*1000:.3f} mm")
print(f"Best angles: {best_angles}\n")

# Apply best and check (negate J3 and J5 for sim)
best_angles_sim = best_angles.copy()
best_angles_sim[2] *= -1  # J3
best_angles_sim[4] *= -1  # J5
print("[FINAL CHECK]")
for i, h in enumerate(joint_handles):
    sim.setJointPosition(h, best_angles_sim[i])

for _ in range(50):
    sim.step()

m_ee = sim.getObjectMatrix(ee_handle, sim.handle_world)
final_ee = np.array([m_ee[3], m_ee[7], m_ee[11]])
final_fk = compute_fk_dh(best_angles)[:3, 3]

print(f"EE (Sim):    {final_ee}")
print(f"EE (FK):     {final_fk}")
print(f"Target:      {grab_pos}")
print(f"\nError (FK-Target):    {np.linalg.norm(final_fk - grab_pos)*1000:.3f} mm")
print(f"Error (Sim-Target):   {np.linalg.norm(final_ee - grab_pos)*1000:.3f} mm")
print(f"Error (Sim-FK):       {np.linalg.norm(final_ee - final_fk)*1000:.3f} mm")

sim.stopSimulation()
print("\n[✓] Done")
