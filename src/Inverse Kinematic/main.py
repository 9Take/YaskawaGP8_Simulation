import numpy as np
import math as m
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --- Initialization ---
client = RemoteAPIClient()
sim = client.require('sim')
sim.setStepping(True)

# --- DH Parameters (Your specific GP8 values) ---
d1, a1 = 0.33, 0.01867
a2, a3, a4 = 0.04, 0.345, 0.04
d5 = -0.34  # Lower arm length (negative based on your DH convention)

def DH_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st, 0, a],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa, ca, d*ca],
        [0, 0, 0, 1]
    ])

def get_all_transforms(q_deg):
    """Returns all intermediate T matrices to extract z-vectors for Jw"""
    q = np.radians(q_deg)
    # T6_7 is your tool/gripper offset
    T6_7 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, -0.241], [0, 0, 0, 1]])

    T01 = DH_matrix(q[0],           d1, a1, 0)
    T12 = DH_matrix(q[1] - np.pi/2, 0,  a2, -np.pi/2)
    T23 = DH_matrix(q[2],           0,  a3, np.pi)
    T34 = DH_matrix(q[3] + np.pi,   d5, a4, -np.pi/2)
    T45 = DH_matrix(q[4],           0,  0,  -np.pi/2)
    T56 = DH_matrix(q[5],           0,  0,  np.pi/2)

    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56
    T07 = T06 @ T6_7 # TCP Frame
    
    
    return [np.eye(4), T01, T02, T03, T04, T05, T06], T07

def calculate_hybrid_jacobian(q_deg):
    """Jv: Numerical Derivative | Jw: Explicit z-vectors"""
    J = np.zeros((6, 6))
    eps = 1e-6
    T_list, T_curr = get_all_transforms(q_deg)
    p_curr = T_curr[:3, 3]

    for i in range(6):
        # --- Jv: Direct Numerical Derivative ---
        q_eps = np.array(q_deg, dtype=float)
        q_eps[i] += eps
        _, T_eps = get_all_transforms(q_eps)
        J[:3, i] = (T_eps[:3, 3] - p_curr) / np.radians(eps)

        # --- Jw: Explicit Form (z_{i-1} vectors) ---
        # z-vector is the 3rd column of the previous rotation matrix
        J[3:, i] = T_list[i][:3, 2] 
        
    return J

def cubic_point(x0, xf, T, t):
    """Calculates position and velocity for a single dimension"""
    tau = t / T
    x = x0 + (xf - x0) * (3*tau**2 - 2*tau**3)
    v = (xf - x0) * (6*tau/T - 6*tau**2/T)
    return x, v

# --- Main Execution ---
def main():
    motor = [sim.getObject(f'/joint{i+1}') for i in range(6)]
    dt = sim.getSimulationTimeStep()
    sim.startSimulation()

    # 1. Trajectory Targets (Home -> Pre-pick)
    # Start (Current Home)
    # q_current = np.array([0, 0, 0, 0, 0, 0])  # Replace with actual home joint angles if different
    q_current = np.array([sim.getJointPosition(motor[j]) for j in range(6)])
    _, T_start = get_all_transforms(q_current)
    p0 = T_start[:3, 3] # [x, y, z]
    
    # Target (Example Pre-pick from your midpoint data)
    pf = np.array([0.2296, -0.5286, 0.3818]) 
    T_duration = 5.0
    total_steps = int(T_duration / dt)

    print(f"Moving from {p0} to {pf} using Inverse Jacobian...")

    for i in range(total_steps):
        t = i * dt
        
        # 2. Get Target Velocity (v) from Cubic Path
        # For simplicity, we keep orientation constant (w = 0) or use Euler diff
        vx, v_val_x = cubic_point(p0[0], pf[0], T_duration, t)
        vy, v_val_y = cubic_point(p0[1], pf[1], T_duration, t)
        vz, v_val_z = cubic_point(p0[2], pf[2], T_duration, t)
        
        # Combined Task Velocity Vector (Twist)
        x_dot = np.array([v_val_x, v_val_y, v_val_z, 0, 0, 0])

        # 3. Inverse Jacobian Control
        J = calculate_hybrid_jacobian(q_current)
        # Use Pseudo-Inverse for stability near singularities
        J_inv = np.linalg.pinv(J)
        
        # Calculate Joint Velocities (rad/s)
        q_dot = J_inv @ x_dot
        
        # 4. Integrate to find new Joint Positions
        q_current += np.degrees(q_dot) * dt
        
        # 5. Update Simulator
        for j in range(6):
            sim.setJointPosition(motor[j], np.radians(q_current[j]))
        
        sim.step()

    print("Target Reached.")
    sim.stopSimulation()

if __name__ == "__main__":
    main()