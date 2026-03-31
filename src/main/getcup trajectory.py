from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib.pyplot as plt
import math

client = RemoteAPIClient()
sim = client.require('sim')

timestep = 0.5  # 50 ms
step = 400

place_pos_conv = np.array([0.455, 0.500, 0.400]) # ตำแหน่งที่เราจะวางแก้ว (อาจต้องปรับตามฉากของคุณ)
place_ori_conv = np.array([0.0, 0.0, 0.0]) # สมมติว่าเราไม่สนใจการวางแก้วในมุมเฉพาะ (หรือปรับตามความต้องการ)
# ============================================================
# Object names and handles
# ============================================================
CUP_NAME = '/conveyorSystem/Cup'
EE_NAME  = '/yaskawa/MicoHand'
ROBOT_NAME  = '/yaskawa'
CONV_NAME = '/conveyor'
def get_handle(name):
    """ดึง handle — ถ้าไม่เจอพิมพ์ชื่อที่ผิด แต่ไม่หยุดโปรแกรม"""
    try:
        h = sim.getObject(name)
        print(f"  [OK] {name}")
        return h
    except:
        print(f"  [WARN] หา '{name}' ไม่พบ — ตรวจชื่อใน CoppeliaSim")
        return None

# try:
#     conv_h    = sim.getObject('/conveyor')
#     place_pos = np.array(sim.getObjectPosition(conv_h, sim.handle_world))
#     place_ori = np.array(sim.getObjectOrientation(conv_h, sim.handle_world))
#     print(f"  Place conv: pos={place_pos.round(4)}")
#     print(f"              ori(deg)={np.degrees(place_ori).round(1)}")
# except Exception:
#     place_pos = np.array([0.455, 0.500, 0.400]) #
#     place_ori = np.array([0.0, 0.0, 0.0])
#     print(f"  Place conv: (default) {place_pos}")

cup_handle = get_handle(CUP_NAME)
ee_handle = get_handle(EE_NAME)
robot_handle = get_handle(ROBOT_NAME)
conv_handle = get_handle(CONV_NAME)


sim.setStepping(True)  # เริ่มจำลองเพื่อให้ได้ตำแหน่งเริ่มต้นของวัตถุ
sim.startSimulation()

ee_pos_home  = np.array(sim.getObjectPosition(ee_handle, sim.handle_world))
cup_pos = np.array(sim.getObjectPosition(cup_handle, sim.handle_world))
robot_base = np.array(sim.getObjectPosition(robot_handle, sim.handle_world))
place_pos = np.array(sim.getObjectPosition(conv_handle, sim.handle_world)) if conv_handle else place_pos_conv
place_ori = np.array(sim.getObjectOrientation(conv_handle, sim.handle_world)) if conv_handle else place_ori_conv
print(f"  EF home:    {ee_pos_home.round(4)}")
print(f"  Robot base: {robot_base.round(4)}")
print(f"  Cup start:  {cup_pos.round(4)}")
print(f"  Place conv: {place_pos.round(4)}")
print(f"  Place ori:  {np.degrees(place_ori).round(1)}")

# Position
cup_px, cup_py, cup_pz = [], [], []
# Velocity
cup_vx, cup_vy, cup_vz = [], [], []
# Orientation (Euler angles)
cup_ox, cup_oy, cup_oz = [], [], []

for i in range(step):
    pos = sim.getObjectPosition(cup_handle, sim.handle_world)
    ori = sim.getObjectOrientation(cup_handle, sim.handle_world)
    vel, _= sim.getObjectVelocity(cup_handle, sim.handle_world)

    cup_px.append(pos[0]); cup_py.append(pos[1]); cup_pz.append(pos[2])
    cup_vx.append(vel[0]); cup_vy.append(vel[1]); cup_vz.append(vel[2])
    cup_ox.append(ori[0]); cup_oy.append(ori[1]); cup_oz.append(ori[2])

    sim.step()

sim.setStepping(False)  # หยุดจำลองเพื่อไม่ให้ข้อมูลเปลี่ยนแปลงขณะวิเคราะห์
sim.stopSimulation()

cup_px = np.array(cup_px); cup_py = np.array(cup_py); cup_pz = np.array(cup_pz);
cup_vx = np.array(cup_vx); cup_vy = np.array(cup_vy); cup_vz = np.array(cup_vz);
cup_ox = np.array(cup_ox); cup_oy = np.array(cup_oy); cup_oz = np.array(cup_oz);
cup_speed = np.sqrt(cup_vx**2 + cup_vy**2 + cup_vz**2)

print(f"  Max speed: {cup_speed.max():.4f} m/s")
print(f"  Collected {step} steps  Final position: {np.array([cup_px[-1], cup_py[-1], cup_pz[-1]]).round(4)}")

# ============================================================
# PHASE 2: Workspace Analysis & Optimal Grab Selection
# ============================================================q

# 1. คำนวณระยะห่าง 2D (X-Y) จากฐานหุ่นยนต์ไปยังแก้ว
dist_from_base = np.hypot(cup_px - robot_base[0], cup_py - robot_base[1])

# 2. วิเคราะห์หาช่วงเวลาที่แก้วเข้ามาใน Workspace
MAX_REACH = 0.70  # สมมติระยะเอื้อมสูงสุดของ GP8 + MicoHand (หน่วย: เมตร)
MIN_REACH = 0.20  # ระยะใกล้สุดเพื่อป้องกันหุ่นยนต์หนีบตัวเอง

# หา index (step) ทั้งหมดที่แก้วอยู่ในระยะ
reachable_idx = np.where((dist_from_base <= MAX_REACH) & (dist_from_base >= MIN_REACH))[0]

if len(reachable_idx) > 0:
    # คอนเซปต์ใหม่: แนะนำจุดกึ่งกลางของช่วงที่เอื้อมถึง เพื่อให้แขนมีพื้นที่ขยับสบายที่สุด
    recommended_idx = reachable_idx[len(reachable_idx) // 2]
    
    print(f"  [INFO] Cup is in reach from step {reachable_idx[0]} to {reachable_idx[-1]}")
    print(f"  [RECOMMEND] Best step to grab: {recommended_idx}")
    print(f"  [DETAILS] Distance: {dist_from_base[recommended_idx]:.3f} m, Time: {recommended_idx * timestep:.2f} s")
else:
    print("  [WARNING] The cup NEVER enters the reachable workspace!")
    recommended_idx = np.argmin(dist_from_base) # ถ้าไม่เจอเลย เอาจุดที่ใกล้สุดแทน

# ============================================================
# 3. Plot Workspace & Trajectory
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Yaskawa GP8 Grab Point Analysis', fontsize=14, fontweight='bold')

# กราฟ 1: Top-down Workspace (ดูจากมุมบน)
ax1.plot(cup_px, cup_py, 'gray', linestyle='--', label='Cup Path')
ax1.scatter(cup_px, cup_py, c=dist_from_base, cmap='coolwarm', s=10)
ax1.plot(robot_base[0], robot_base[1], 'k^', markersize=12, label='Robot Base')
ax1.plot(cup_px[recommended_idx], cup_py[recommended_idx], 'r*', markersize=15, label='Recommended Grab')

# วาดวงกลมแสดงขอบเขตระยะเอื้อม (Workspace)
circle_max = plt.Circle((robot_base[0], robot_base[1]), MAX_REACH, color='g', fill=False, linestyle='--', label=f'Max Reach ({MAX_REACH}m)')
circle_min = plt.Circle((robot_base[0], robot_base[1]), MIN_REACH, color='r', fill=False, linestyle='--', label=f'Min Reach ({MIN_REACH}m)')
ax1.add_patch(circle_max)
ax1.add_patch(circle_min)

ax1.set_aspect('equal')
ax1.set_title('Top-Down View: Conveyor & Workspace')
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
ax1.legend()

# กราฟ 2: Distance over Time (ดูจังหวะเวลา)
time_array = np.arange(step) * timestep
ax2.plot(time_array, dist_from_base, 'b-', label='Distance to Base')
ax2.axhline(MAX_REACH, color='g', linestyle='--', label='Max Reach Limit')
ax2.axhline(MIN_REACH, color='r', linestyle='--', label='Min Reach Limit')
ax2.axvline(recommended_idx * timestep, color='purple', linestyle=':', linewidth=2, label='Recommended Time')

ax2.set_title('Cup Distance from Robot over Time')
ax2.set_xlabel('Time (seconds)'); ax2.set_ylabel('Distance (m)')
ax2.legend()

plt.tight_layout()
plt.show() # โปรแกรมจะหยุดตรงนี้จนกว่าคุณจะปิดหน้าต่างกราฟ

# ============================================================
# 4. Confirm & Save Data
# ============================================================
# ให้ผู้ใช้ยืนยันจุดที่จะหยิบ (กด Enter เพื่อใช้ค่าแนะนำ หรือพิมพ์เลขใหม่)
user_input = input(f"\n  >>> Enter GRAB_STEP (Press ENTER to use recommended {recommended_idx}): ").strip()
final_grab_step = int(user_input) if user_input.isdigit() else recommended_idx

print(f"  [LOCKED] Proceeding with GRAB_STEP: {final_grab_step}")

# Print cup coordinates at the selected grab step
grab_time = final_grab_step * timestep
grab_pos = np.array([cup_px[final_grab_step], cup_py[final_grab_step], cup_pz[final_grab_step]])
grab_ori = np.array([cup_ox[final_grab_step], cup_oy[final_grab_step], cup_oz[final_grab_step]])
grab_vel = np.array([cup_vx[final_grab_step], cup_vy[final_grab_step], cup_vz[final_grab_step]])

print(f"\n  === Cup Coordinates at GRAB_STEP {final_grab_step} (Time: {grab_time:.2f}s) ===")
print(f"  Position (X, Y, Z):     {grab_pos.round(4)}")
print(f"  Orientation (rad):      {grab_ori.round(4)}")
print(f"  Orientation (degrees):  {np.degrees(grab_ori).round(2)}")
print(f"  Velocity (Vx, Vy, Vz):  {grab_vel.round(4)}")
print(f"  Speed:                  {np.linalg.norm(grab_vel):.4f} m/s")
print()

# Save ข้อมูลลงไฟล์ .npz เพื่อส่งต่อให้ไฟล์อื่น
filename = 'trajectory_cup_data.npz'
np.savez(filename,
         px=cup_px, py=cup_py, pz=cup_pz,
         ox=cup_ox, oy=cup_oy, oz=cup_oz,
         vx=cup_vx, vy=cup_vy, vz=cup_vz,
         robot_base=robot_base,
         ee_home=ee_pos_home,
         timestep=timestep,
         total_steps=step,
         grab_step=final_grab_step,
         place_pos=place_pos,
         place_ori=place_ori)
            
    
print(f"  [SUCCESS] Data saved to {filename}. Ready for Kinematics calculation!")

