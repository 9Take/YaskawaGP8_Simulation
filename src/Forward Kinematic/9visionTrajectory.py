import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- ฟังก์ชันสร้างเส้นทาง S-Curve ---
def generate_cartesian_trajectory(p_start, p_end, steps=50):
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    trajectory_points = []
    
    for step in range(1, steps + 1):
        t = step / steps
        s = (1.0 - math.cos(math.pi * t)) / 2.0  # S-Curve Equation
        p_current = p_start + (p_end - p_start) * s
        trajectory_points.append(p_current.tolist())
        
    return trajectory_points

# --- 1. กำหนดพิกัด Waypoints (ดึงมาจาก 1-1 และ 2-1) ---
P_PICK = [0.4072, 0.1600, -0.4000]  # จุดหยิบ
P_PLACE = [0.3897, 0.0623, -0.5609] # จุดวาง
P_HOVER_PICK = [P_PICK[0], P_PICK[1], P_PICK[2] + 0.15]   # ยกขึ้น 15 ซม.
P_HOVER_PLACE = [P_PLACE[0], P_PLACE[1], P_PLACE[2] + 0.15] # ยกขึ้น 15 ซม.

# --- 2. สร้างเส้นทาง 3 ช่วง ---
path_up = generate_cartesian_trajectory(P_PICK, P_HOVER_PICK, steps=30)
path_cross = generate_cartesian_trajectory(P_HOVER_PICK, P_HOVER_PLACE, steps=60)
path_down = generate_cartesian_trajectory(P_HOVER_PLACE, P_PLACE, steps=30)

# รวมเป็นเส้นทางเดียว
full_trajectory = np.array(path_up + path_cross + path_down)

# --- 3. พล็อตลงกราฟ 3 มิติ (Plot 3D) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# วาดเส้นทางการเคลื่อนที่
ax.plot(full_trajectory[:,0], full_trajectory[:,1], full_trajectory[:,2], 
        label='S-Curve Trajectory Path', color='blue', linewidth=2)

# จุดไข่ปลา (Waypoints)
ax.scatter(*P_PICK, color='green', s=100, label='Pick (Start)')
ax.scatter(*P_HOVER_PICK, color='cyan', s=50, label='Hover Pick')
ax.scatter(*P_HOVER_PLACE, color='magenta', s=50, label='Hover Place')
ax.scatter(*P_PLACE, color='red', s=100, label='Place (End)')

# ตกแต่งกราฟ
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('Robot End-Effector 3D Cartesian Trajectory')
ax.legend()

# โชว์กราฟ
plt.show()