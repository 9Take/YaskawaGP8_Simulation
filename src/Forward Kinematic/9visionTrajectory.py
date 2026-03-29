import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. กำหนดจุดสำคัญ (Waypoints) จากพิกัดที่คุณหามาได้
# =================================================================
# จุดเริ่มต้น (Home)
home = np.array([0.4500, 0.0000, 0.4000])

# จุดหยิบแก้ว (Pick)
pick = np.array([0.4072, 0.1600, -0.4000])
hover_pick = pick + np.array([0.0, 0.0, 0.15]) # จุดโฉบเหนือแก้ว (ยกสูงขึ้น 15 cm)

# จุดวางแก้ว (Place)
place = np.array([0.3897, 0.0623, -0.4000])
hover_place = place + np.array([0.0, 0.0, 0.15]) # จุดโฉบเหนือแท่น (ยกสูงขึ้น 15 cm)

# =================================================================
# 2. ฟังก์ชัน S-Curve (ทำให้การเคลื่อนที่สมูท เริ่มช้า-กลางเร็ว-จบช้า)
# =================================================================
def scurve(t, duration):
    # คืนค่าตัวคูณตั้งแต่ 0.0 ถึง 1.0 แบบโค้ง S-Curve
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t / duration, 0.0, 1.0)))

# =================================================================
# 3. สร้างแผนการเดินทาง (Phase Planning)
# =================================================================
# รูปแบบ: (จุดเริ่มต้น, จุดหมาย, เวลาที่ใช้เดินทางวินาที)
phases = [
    (home, hover_pick, 2.0),         # Phase 1: จาก Home ไปโฉบเหนือแก้ว
    (hover_pick, pick, 1.0),         # Phase 2: ก้มลงไปหยิบแก้ว
    (pick, hover_pick, 1.0),         # Phase 3: ยกแก้วขึ้น
    (hover_pick, hover_place, 2.0),  # Phase 4: สวิงข้ามไปเหนือแท่นวาง (TRANSPORT)
    (hover_place, place, 1.0),       # Phase 5: ก้มลงไปวาง
    (place, hover_place, 1.0),       # Phase 6: ถอยแขนขึ้น
    (hover_place, home, 2.0)         # Phase 7: กลับ Home
]

dt = 0.05 # คำนวณทุกๆ 0.05 วินาที
time_log = []
traj_x, traj_y, traj_z = [], [], []

current_time = 0.0

# คำนวณพิกัดย่อยๆ ในแต่ละ Phase
for start_pt, end_pt, duration in phases:
    steps = int(duration / dt)
    for i in range(steps):
        t = i * dt
        s = scurve(t, duration) # ใช้ S-Curve คุมความเร็ว
        
        # คำนวณพิกัดปัจจุบัน: P = P_start + s * (P_end - P_start)
        p = start_pt + s * (end_pt - start_pt)
        
        traj_x.append(p[0])
        traj_y.append(p[1])
        traj_z.append(p[2])
        time_log.append(current_time + t)
        
    current_time += duration

# =================================================================
# 4. วาดกราฟ (Plotting)
# =================================================================
fig = plt.figure(figsize=(14, 6))

# กราฟที่ 1: การเดินทางของแกน X, Y, Z เทียบกับเวลา
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(time_log, traj_x, label='X Position (m)', color='red', linewidth=2)
ax1.plot(time_log, traj_y, label='Y Position (m)', color='green', linewidth=2)
ax1.plot(time_log, traj_z, label='Z Position (m)', color='blue', linewidth=2)
ax1.set_title('Trajectory Profile (Position vs Time)')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Position (meters)')
ax1.grid(True)
ax1.legend()

# ใส่เส้นประแบ่ง Phase ให้ดูง่ายๆ
phase_times = [0]
for _, _, duration in phases:
    phase_times.append(phase_times[-1] + duration)
    ax1.axvline(x=phase_times[-1], color='gray', linestyle='--', alpha=0.5)

# กราฟที่ 2: เส้นทาง 3 มิติ (3D Path)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(traj_x, traj_y, traj_z, label='End-Effector Path', color='purple', linewidth=2)
ax2.scatter(home[0], home[1], home[2], color='black', s=50, label='Home')
ax2.scatter(pick[0], pick[1], pick[2], color='blue', s=50, label='Pick (Cup)')
ax2.scatter(place[0], place[1], place[2], color='red', s=50, label='Place (Zone)')
ax2.set_title('3D Robot Trajectory')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Z (m)')
ax2.legend()

plt.tight_layout()
plt.show()
# เพิ่มโค้ดนี้ไว้บรรทัดล่างสุดของไฟล์ plot_trajectory.py
np.savez('my_planned_path.npz', x=traj_x, y=traj_y, z=traj_z, time=time_log)
print("💾 เซฟไฟล์เส้นทาง 'my_planned_path.npz' สำเร็จแล้ว!")