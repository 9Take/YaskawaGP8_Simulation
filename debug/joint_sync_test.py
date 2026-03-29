"""Test if joint angles set are actually being applied in simulation"""
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

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
    """FK using DH"""
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

# Get robot parts
try:
    joint_handles = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
    ee_handle = sim.getObject('/yaskawa/gripperEF')
except:
    joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]
    ee_handle = sim.getObjectHandle('/yaskawa/gripperEF')

print("=" * 80)
print("JOINT SYNCHRONIZATION TEST")
print("=" * 80)

# Reset to home
print("\n[RESET TO HOME]")
for j in joints:
    sim.setJointPosition(j, 0.0)
time.sleep(0.5)

# Read home position
home_angles = np.array([sim.getJointPosition(j) for j in joints])
T_home_sim = sim.getObjectMatrix(ef, robot_base)
T_home_fk = compute_fk_dh(home_angles)

print(f"Joint angles: {home_angles}")
print(f"EE pos (sim):  {T_home_sim[:3,3]}")
print(f"EE pos (FK):   {T_home_fk[:3,3]}")
print(f"Difference:    {T_home_sim[:3,3] - T_home_fk[:3,3]}")

# Now set to a specific pose
print("\n[TEST POSE 1: SMALL MOVEMENT]")
test_angles_1 = np.array([-0.2, 0.1, 0.05, 0.0, 0.0, 0.0])
print(f"Setting angles: {test_angles_1}")
for j, angle in zip(joints, test_angles_1):
    sim.setJointPosition(j, angle)
time.sleep(0.5)

# Read what was actually set
read_angles_1 = np.array([sim.getJointPosition(j) for j in joints])
T_pose1_sim = sim.getObjectMatrix(ef, robot_base)
T_pose1_fk = compute_fk_dh(read_angles_1)

print(f"Angles (set):  {test_angles_1}")
print(f"Angles (read): {read_angles_1}")
print(f"Match: {np.allclose(test_angles_1, read_angles_1)}")

print(f"\nEE position:")
print(f"  FK:          {T_pose1_fk[:3,3]}")
print(f"  Sim:         {T_pose1_sim[:3,3]}")
print(f"  Error (Sim - FK): {T_pose1_sim[:3,3] - T_pose1_fk[:3,3]}")

# Try the actual IK solution
print("\n[TEST POSE 2: IK SOLUTION]")
test_angles_2 = np.array([-0.7920073, 0.31202176, 0.15516769, -0.02662137, 0.02491295, 0.0])
print(f"Setting angles: {test_angles_2}")
for j, angle in zip(joints, test_angles_2):
    sim.setJointPosition(j, angle)
time.sleep(0.5)

read_angles_2 = np.array([sim.getJointPosition(j) for j in joints])
T_pose2_sim = sim.getObjectMatrix(ef, robot_base)
T_pose2_fk = compute_fk_dh(read_angles_2)

print(f"Angles (set):  {test_angles_2}")
print(f"Angles (read): {read_angles_2}")
print(f"Match: {np.allclose(test_angles_2, read_angles_2)}")

print(f"\nEE position:")
print(f"  FK:          {T_pose2_fk[:3,3]}")
print(f"  Sim:         {T_pose2_sim[:3,3]}")
print(f"  Error (Sim - FK): {T_pose2_sim[:3,3] - T_pose2_fk[:3,3]}")

# ANALYSIS
print("\n[POTENTIAL ISSUES]")
print(f"1. Joint angle mismatch?")
print(f"   Pose1: {not np.allclose(test_angles_1, read_angles_1)}")
print(f"   Pose2: {not np.allclose(test_angles_2, read_angles_2)}")

print(f"\n2. FK computation vs Sim match at each pose?")
pose1_error = np.linalg.norm(T_pose1_sim[:3,3] - T_pose1_fk[:3,3])
pose2_error = np.linalg.norm(T_pose2_sim[:3,3] - T_pose2_fk[:3,3])
print(f"   Pose1 error: {pose1_error*1000:.1f} mm (match: {pose1_error < 0.01})")
print(f"   Pose2 error: {pose2_error*1000:.1f} mm (match: {pose2_error < 0.01})")

print(f"\n3. Is there a systematic offset?")
print(f"   Home: Sim-FK = {np.linalg.norm(T_home_sim[:3,3] - T_home_fk[:3,3])*1000:.1f} mm")
print(f"   Pose1: Sim-FK = {pose1_error*1000:.1f} mm")
print(f"   Pose2: Sim-FK = {pose2_error*1000:.1f} mm")

print("\n[✓] Done")
sim.stopSimulation()
