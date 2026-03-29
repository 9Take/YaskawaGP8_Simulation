#!/usr/bin/env python3
"""Quick validation: Test IK solver on multiple target positions"""
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import sys
sys.path.insert(0, '/home/porsche/KMUTNB/Year3/Term2/Final_Robot/src/main')
from ik_solver_production import solve_ik, compute_fk_dh

client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()

try:
    joint_handles = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
    ee_handle = sim.getObject('/yaskawa/gripperEF')
except:
    joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]
    ee_handle = sim.getObjectHandle('/yaskawa/gripperEF')

# Reset to home
for h in joint_handles:
    sim.setJointPosition(h, 0.0)
for _ in range(50):
    sim.step()

# Get home position
m = sim.getObjectMatrix(ee_handle, sim.handle_world)
home_pos = np.array([m[3], m[7], m[11]])

print("=" * 70)
print("IK SOLVER VALIDATION - MULTIPLE TARGET POSITIONS")
print("=" * 70)

targets = [
    ("Home + X", home_pos + np.array([0.1, 0, 0])),
    ("Home + Y", home_pos + np.array([0, 0.1, 0])),
    ("Home + Z", home_pos + np.array([0, 0, -0.1])),
    ("Diagonal", home_pos + np.array([0.1, -0.1, -0.1])),
    ("Grab from trajectory", np.array([0.498, -0.488, 0.426])),
]

results = []
for name, target in targets:
    print(f"\n[TEST: {name}]")
    print(f"Target: {target}")
    
    converged, angles, error = solve_ik(target, joint_handles, sim, ee_handle)
    results.append((name, converged, error))
    print(f"Result: {'✓ CONVERGED' if converged else '✗ FAILED'} (error: {error*1000:.1f}mm)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for name, converged, error in results:
    status = "✓" if converged and error < 0.01 else "✗"
    print(f"{status} {name:25s} {error*1000:6.1f}mm")

sim.stopSimulation()
