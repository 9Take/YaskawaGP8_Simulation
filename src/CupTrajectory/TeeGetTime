import time
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================
# 1. เชื่อมต่อกับ CoppeliaSim
# ==========================================
client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()  # เริ่ม Simulation
# ระบุชื่อ Object ของแก้วน้ำ (แก้ '/Cup' ให้ตรงกับชื่อในโปรแกรมของคุณ)
cup_name = '/Cup' 
try:
    cup_handle = sim.getObject(cup_name)
except Exception as e:
    print(f"หา Object ชื่อ {cup_name} ไม่พบ กรุณาตรวจสอบชื่อใน Scene")
    exit()

# เตรียมลิสต์เก็บข้อมูล
time_data = []
x_data, y_data, z_data = [], [], []
vx_data, vy_data, vz_data = [], [], []

print("กำลังบันทึกข้อมูลตำแหน่ง และ ความเร็ว ของแก้วน้ำ...")
record_duration = 20  # บันทึกข้อมูลเป็นเวลา 10 วินาที
start_time = sim.getSimulationTime()

# ==========================================
# 2. ลูปเก็บข้อมูลจาก Simulation
# ==========================================
while True:
    t = sim.getSimulationTime()
    if t - start_time > record_duration:
        break
        
    # ดึงค่าตำแหน่ง (x, y, z) อ้างอิงจาก World Frame
    pos = sim.getObjectPosition(cup_handle, sim.handle_world)
    
    # ดึงค่าความเร็ว (ฟังก์ชันนี้จะคืนค่า [Linear Velocity], [Angular Velocity])
    linear_vel, angular_vel = sim.getObjectVelocity(cup_handle)
    
    time_data.append(t)
    x_data.append(pos[0])
    y_data.append(pos[1])
    z_data.append(pos[2])
    
    vx_data.append(linear_vel[0])
    vy_data.append(linear_vel[1])
    vz_data.append(linear_vel[2])
    
    time.sleep(0.05) # หน่วงเวลาเล็กน้อย

# แปลงเป็น numpy array
t_arr = np.array(time_data)
x_arr, y_arr, z_arr = np.array(x_data), np.array(y_data), np.array(z_data)
vx_arr, vy_arr, vz_arr = np.array(vx_data), np.array(vy_data), np.array(vz_data)

# ==========================================
# 3. พล็อตกราฟตำแหน่ง และ ความเร็ว (Position & Velocity vs Time)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# กราฟตำแหน่ง (Position vs Time)
ax1.plot(t_arr, x_arr, label='X Position', color='r', linewidth=2)
ax1.plot(t_arr, y_arr, label='Y Position', color='g', linewidth=2)
ax1.plot(t_arr, z_arr, label='Z Position', color='b', linewidth=2)
ax1.set_title('Cup Linear Position (x, y, z) vs Time')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)')
ax1.legend()
ax1.grid(True)

# กราฟความเร็ว (Velocity vs Time)
ax2.plot(t_arr, vx_arr, label='Vx Velocity', color='r', linewidth=2)
ax2.plot(t_arr, vy_arr, label='Vy Velocity', color='g', linewidth=2)
ax2.plot(t_arr, vz_arr, label='Vz Velocity', color='b', linewidth=2)
ax2.set_title('Cup Linear Velocity (Vx, Vy, Vz) vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# คำนวณหาค่าเฉลี่ยความเร็ว เพื่อเอาไปใช้ตั้งสมการ Trajectory ของแขนกล
vx_mean = np.mean(vx_arr)
vy_mean = np.mean(vy_arr)
vz_mean = np.mean(vz_arr)

print("\n=== ข้อมูลสำหรับไปทำ Trajectory Generation ===")
print(f"ความเร็วเฉลี่ยแกน X (Vx) : {vx_mean:.4f} m/s")
print(f"ความเร็วเฉลี่ยแกน Y (Vy) : {vy_mean:.4f} m/s")
print(f"ความเร็วเฉลี่ยแกน Z (Vz) : {vz_mean:.4f} m/s")
print("==============================================")