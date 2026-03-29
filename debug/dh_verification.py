"""Test individual joint rotations to identify DH parameter issues"""
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

def dh_matrix(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def compute_fk_dh(angles):
    T_Base = np.array([[1, 0, 0, 0.0187], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T01 = dh_matrix(-np.pi/2, 0.040, 0.330,  angles[0])
    T12 = dh_matrix(0,        0.345, 0,      angles[1] - np.pi/2)
    T23 = dh_matrix(-np.pi/2, 0.040, 0,      angles[2])
    T34 = dh_matrix(np.pi/2,  0,     0.340,  angles[3])
    T45 = dh_matrix(-np.pi/2, 0,     0,      angles[4])
    T56 = dh_matrix(0,        0,     0.2413, angles[5])
    return T_Base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56

client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()

try:
    joint_handles = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
    ee_handle = sim.getObject('/yaskawa/gripperEF')
except:
    joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]
    ee_handle = sim.getObjectHandle('/yaskawa/gripperEF')

print("=" * 80)
print("DH PARAMETER VERIFICATION - SINGLE JOINT TESTS")
print("=" * 80)

# Test each joint individually
for joint_idx in range(6):
    print(f"\n[JOINT {joint_idx + 1} - Individual Rotation Test]")
    
    # Reset to home
    for j in joint_handles:
        sim.setJointPosition(j, 0.0)
    time.sleep(0.3)
    
    # Get FK at home for this joint
    home_angles = np.array([sim.getJointPosition(j) for j in joint_handles])
    
    # Move only this joint
    test_angles = home_angles.copy()
    test_angles[joint_idx] = 0.3  # 30 degrees
    
    for j, angle in zip(joint_handles, test_angles):
        sim.setJointPosition(j, angle)
    time.sleep(0.3)
    
    # Compare FK vs Sim
    angles_read = np.array([sim.getJointPosition(j) for j in joint_handles])
    m_pos = sim.getObjectMatrix(ee_handle, sim.handle_world)
    sim_pos = np.array([m_pos[3], m_pos[7], m_pos[11]])
    fk_pos = compute_fk_dh(angles_read)[:3, 3]
    
    error = np.linalg.norm(sim_pos - fk_pos)
    print(f"  Angle: J{joint_idx+1} = 0.3 rad")
    print(f"  FK:  {fk_pos}")
    print(f"  Sim: {sim_pos}")
    print(f"  Error: {error*1000:.1f} mm")
    
    if error > 0.020:  # > 20mm error
        print(f"  ⚠️ LARGE ERROR - Potential DH issue")

print("\n" + "=" * 80)
print("[Analysis]")
print("If errors are large for one joint but small for others,")
print("that joint's DH parameters or direction may be wrong.")
print("=" * 80)

sim.stopSimulation()
