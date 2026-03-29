"""Test J3 and J5 with opposite directions"""
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

def compute_fk_dh(angles, j3_neg=False, j5_neg=False):
    """FK with optional J3/J5 negation"""
    a = angles.copy()
    if j3_neg: a[2] *= -1
    if j5_neg: a[4] *= -1
    
    T_Base = np.array([[1, 0, 0, 0.0187], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T01 = dh_matrix(-np.pi/2, 0.040, 0.330,  a[0])
    T12 = dh_matrix(0,        0.345, 0,      a[1] - np.pi/2)
    T23 = dh_matrix(-np.pi/2, 0.040, 0,      a[2])
    T34 = dh_matrix(np.pi/2,  0,     0.340,  a[3])
    T45 = dh_matrix(-np.pi/2, 0,     0,      a[4])
    T56 = dh_matrix(0,        0,     0.2413, a[5])
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
print("TESTING J3 AND J5 ROTATION DIRECTION")
print("=" * 80)

# Test J3 with +0.3
print("\n[J3 Test: +0.3 rad]")
for j in joint_handles:
    sim.setJointPosition(j, 0.0)
time.sleep(0.3)

angles = np.array([0, 0, 0.3, 0, 0, 0])
for j, a in zip(joint_handles, angles):
    sim.setJointPosition(j, a)
time.sleep(0.3)

angles_read = np.array([sim.getJointPosition(j) for j in joint_handles])
m_pos = sim.getObjectMatrix(ee_handle, sim.handle_world)
sim_pos = np.array([m_pos[3], m_pos[7], m_pos[11]])
fk_pos = compute_fk_dh(angles_read, j3_neg=False)[:3, 3]
fk_neg = compute_fk_dh(angles_read, j3_neg=True)[:3, 3]

print(f"FK (normal):  {fk_pos}")
print(f"FK (negated): {fk_neg}")
print(f"Sim:          {sim_pos}")
print(f"Error (normal):  {np.linalg.norm(sim_pos - fk_pos)*1000:.1f} mm")
print(f"Error (negated): {np.linalg.norm(sim_pos - fk_neg)*1000:.1f} mm")

# Test J5 with +0.3
print("\n[J5 Test: +0.3 rad]")
for j in joint_handles:
    sim.setJointPosition(j, 0.0)
time.sleep(0.3)

angles = np.array([0, 0, 0, 0, 0.3, 0])
for j, a in zip(joint_handles, angles):
    sim.setJointPosition(j, a)
time.sleep(0.3)

angles_read = np.array([sim.getJointPosition(j) for j in joint_handles])
m_pos = sim.getObjectMatrix(ee_handle, sim.handle_world)
sim_pos = np.array([m_pos[3], m_pos[7], m_pos[11]])
fk_pos = compute_fk_dh(angles_read, j5_neg=False)[:3, 3]
fk_neg = compute_fk_dh(angles_read, j5_neg=True)[:3, 3]

print(f"FK (normal):  {fk_pos}")
print(f"FK (negated): {fk_neg}")
print(f"Sim:          {sim_pos}")
print(f"Error (normal):  {np.linalg.norm(sim_pos - fk_pos)*1000:.1f} mm")
print(f"Error (negated): {np.linalg.norm(sim_pos - fk_neg)*1000:.1f} mm")

# Test J3 with -0.3
print("\n[J3 Test: -0.3 rad]")
for j in joint_handles:
    sim.setJointPosition(j, 0.0)
time.sleep(0.3)

angles = np.array([0, 0, -0.3, 0, 0, 0])
for j, a in zip(joint_handles, angles):
    sim.setJointPosition(j, a)
time.sleep(0.3)

angles_read = np.array([sim.getJointPosition(j) for j in joint_handles])
m_pos = sim.getObjectMatrix(ee_handle, sim.handle_world)
sim_pos = np.array([m_pos[3], m_pos[7], m_pos[11]])
fk_pos = compute_fk_dh(angles_read, j3_neg=False)[:3, 3]
fk_neg = compute_fk_dh(angles_read, j3_neg=True)[:3, 3]

print(f"FK (normal):  {fk_pos}")
print(f"FK (negated): {fk_neg}")
print(f"Sim:          {sim_pos}")
print(f"Error (normal):  {np.linalg.norm(sim_pos - fk_pos)*1000:.1f} mm")
print(f"Error (negated): {np.linalg.norm(sim_pos - fk_neg)*1000:.1f} mm")

sim.stopSimulation()
