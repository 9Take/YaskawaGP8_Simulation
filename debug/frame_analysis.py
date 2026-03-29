#!/usr/bin/env python3
"""
Debug: Coordinate Frame Analysis
Compares EE position readings from different frame references
"""

import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


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
    """Compute FK using DH parameters"""
    # Base transformation
    T_Base = np.array([
        [1, 0, 0, 0.0187],
        [0, 1, 0, 0.0000],
        [0, 0, 1, 0.0000],
        [0, 0, 0, 1.0000]
    ])
    
    # DH transformations
    T01 = dh_matrix(-np.pi/2, 0.040, 0.330,  angles[0])
    T12 = dh_matrix(0,        0.345, 0,      angles[1] - np.pi/2)
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


def main():
    print("="*80)
    print("COORDINATE FRAME DEBUG")
    print("="*80)
    
    try:
        client = RemoteAPIClient()
        sim = client.require('sim')
        sim.startSimulation()
        print("\n[✓] Connected to CoppeliaSim")
    except Exception as e:
        print(f"[✗] Connection failed: {e}")
        return
    
    time.sleep(0.5)
    
    # Get handles
    try:
        robot_base_handle = sim.getObject('/yaskawa')
    except:
        robot_base_handle = sim.getObjectHandle('/yaskawa')
    
    try:
        ee_handle = sim.getObject('/yaskawa/gripperEF')
    except:
        ee_handle = sim.getObjectHandle('/yaskawa/gripperEF')
    
    try:
        joint_handles = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
    except:
        joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]
    
    # Get joint angles
    angles = np.array([sim.getJointPosition(h) for h in joint_handles])
    
    print("\n[1. HOME CONFIGURATION]")
    print(f"    Joint angles: {angles}")
    
    # ==================================================
    # Test different EE position reading methods
    # ==================================================
    
    print("\n[2. EE POSITION - DIFFERENT FRAME REFERENCES]")
    print("-" * 80)
    
    # Method 1: getObjectMatrix with world frame
    m_world = sim.getObjectMatrix(ee_handle, sim.handle_world)
    T_world = np.array([
        [m_world[0], m_world[1], m_world[2],  m_world[3]],
        [m_world[4], m_world[5], m_world[6],  m_world[7]],
        [m_world[8], m_world[9], m_world[10], m_world[11]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    pos_world = T_world[:3, 3]
    print(f"  Method A - getObjectMatrix(ee, WORLD):")
    print(f"    Position: {pos_world}")
    print(f"    Rotation:\n{T_world[:3, :3]}\n")
    
    # Method 2: getObjectMatrix with robot base frame
    m_base = sim.getObjectMatrix(ee_handle, robot_base_handle)
    T_base = np.array([
        [m_base[0], m_base[1], m_base[2],  m_base[3]],
        [m_base[4], m_base[5], m_base[6],  m_base[7]],
        [m_base[8], m_base[9], m_base[10], m_base[11]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    pos_base = T_base[:3, 3]
    print(f"  Method B - getObjectMatrix(ee, ROBOT_BASE):")
    print(f"    Position: {pos_base}")
    print(f"    Rotation:\n{T_base[:3, :3]}\n")
    
    # Method 3: getObjectPosition with world frame
    pos_world_alt = np.array(sim.getObjectPosition(ee_handle, sim.handle_world))
    print(f"  Method C - getObjectPosition(ee, WORLD):")
    print(f"    Position: {pos_world_alt}\n")
    
    # Method 4: getObjectPosition with robot base frame
    pos_robot_alt = np.array(sim.getObjectPosition(ee_handle, robot_base_handle))
    print(f"  Method D - getObjectPosition(ee, ROBOT_BASE):")
    print(f"    Position: {pos_robot_alt}\n")
    
    # FK computation
    T_fk = compute_fk_dh(angles)
    pos_fk = extract_position(T_fk)
    print(f"  FK Computation (DH model):")
    print(f"    Position: {pos_fk}")
    print(f"    Rotation:\n{T_fk[:3, :3]}\n")
    
    # ==================================================
    # Compare all methods
    # ==================================================
    
    print("[3. COMPARISON & ERRORS]")
    print("-" * 80)
    
    methods = {
        'A (getMatrix WORLD)': pos_world,
        'B (getMatrix BASE)': pos_base,
        'C (getPos WORLD)': pos_world_alt,
        'D (getPos BASE)': pos_robot_alt,
        'FK (DH model)': pos_fk,
    }
    
    for name1, pos1 in methods.items():
        for name2, pos2 in methods.items():
            if name1 >= name2:
                continue
            error = np.linalg.norm(pos1 - pos2) * 1000
            print(f"  {name1:20s} vs {name2:20s}: {error:7.3f} mm")
    
    print("\n[4. ROBOT BASE FRAME DETAILS]")
    print("-" * 80)
    robot_base_pos = np.array(sim.getObjectPosition(robot_base_handle, sim.handle_world))
    robot_base_ori = np.array(sim.getObjectOrientation(robot_base_handle, sim.handle_world))
    print(f"  Robot base position (world): {robot_base_pos}")
    print(f"  Robot base orientation (rad): {robot_base_ori}")
    print(f"  Robot base orientation (deg): {np.degrees(robot_base_ori)}")
    
    # ==================================================
    # Test with IK solution angles
    # ==================================================
    
    print("\n[5. TEST WITH IK SOLUTION ANGLES]")
    print("-" * 80)
    
    # Load IK angles from trajectory
    try:
        data = np.load('../trajectory_cup_data.npz')
        grab_step = int(data['grab_step'])
        grab_pos = np.array([data['px'][grab_step], data['py'][grab_step], data['pz'][grab_step]])
        print(f"  Target grab position: {grab_pos}\n")
    except:
        grab_pos = np.array([0.5, -0.5, 0.4])
        print(f"  Using default grab position: {grab_pos}\n")
    
    # IK solution angles (from previous run)
    ik_angles = np.array([-0.7920073, 0.31202176, 0.15516769, -0.02662137, 0.02491295, 0.0])
    
    # Set angles to sim
    for i, h in enumerate(joint_handles):
        sim.setJointPosition(h, ik_angles[i])
    
    # Wait for sim to update
    for _ in range(100):
        sim.step()
    
    print("  IK angles applied to simulation")
    
    # Read all methods again
    m_world = sim.getObjectMatrix(ee_handle, sim.handle_world)
    pos_world_ik = np.array([m_world[3], m_world[7], m_world[11]])
    
    m_base = sim.getObjectMatrix(ee_handle, robot_base_handle)
    pos_base_ik = np.array([m_base[3], m_base[7], m_base[11]])
    
    pos_world_alt_ik = np.array(sim.getObjectPosition(ee_handle, sim.handle_world))
    pos_robot_alt_ik = np.array(sim.getObjectPosition(ee_handle, robot_base_handle))
    
    T_fk_ik = compute_fk_dh(ik_angles)
    pos_fk_ik = extract_position(T_fk_ik)
    
    print(f"  Method A (getMatrix WORLD): {pos_world_ik}")
    print(f"  Method B (getMatrix BASE):  {pos_base_ik}")
    print(f"  Method C (getPos WORLD):    {pos_world_alt_ik}")
    print(f"  Method D (getPos BASE):     {pos_robot_alt_ik}")
    print(f"  FK (DH model):              {pos_fk_ik}")
    print(f"  Target position:            {grab_pos}\n")
    
    print("[6. ERROR TO TARGET]")
    print("-" * 80)
    print(f"  Method A error: {np.linalg.norm(pos_world_ik - grab_pos)*1000:7.3f} mm")
    print(f"  Method B error: {np.linalg.norm(pos_base_ik - grab_pos)*1000:7.3f} mm")
    print(f"  Method C error: {np.linalg.norm(pos_world_alt_ik - grab_pos)*1000:7.3f} mm")
    print(f"  Method D error: {np.linalg.norm(pos_robot_alt_ik - grab_pos)*1000:7.3f} mm")
    print(f"  FK error:       {np.linalg.norm(pos_fk_ik - grab_pos)*1000:7.3f} mm")
    
    sim.stopSimulation()
    print("\n[✓] Test complete")


if __name__ == '__main__':
    main()
