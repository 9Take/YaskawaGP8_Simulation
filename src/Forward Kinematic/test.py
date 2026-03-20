import math as m
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==============================================================================
# 1. SETUP & INITIALIZE
# ==============================================================================
print("กำลังเชื่อมต่อกับ CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')
sim.setStepping(True) 

JOINT_NAMES = ['/yaskawa/joint1', '/yaskawa/joint2', '/yaskawa/joint3',
               '/yaskawa/joint4', '/yaskawa/joint5', '/yaskawa/joint6']
joints = [sim.getObject(name) for name in JOINT_NAMES]

mico_motor1 = sim.getObject('/yaskawa/MicoHand/fingers12_motor1')
mico_motor2 = sim.getObject('/yaskawa/MicoHand/fingers12_motor2')

# เตรียมเก็บข้อมูลเพื่อพล็อตกราฟ Vgripper (Rubric 4.2)
time_log = []
vgripper_log = []
current_vgripper = 0.0

def step_sim():
    """ฟังก์ชันก้าวเวลาและเก็บข้อมูลกริปเปอร์"""
    time_log.append(sim.getSimulationTime())
    vgripper_log.append(current_vgripper)
    sim.step()

# ==============================================================================
# 2. INVERSE KINEMATICS ENGINE (Rubric 3)
# ==============================================================================
L1, L2, L3, L4, L5, L6, L7 = 330, 345, 40, 40, 340, 80, 161.33

def std_dh(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def get_jacobian_and_pos(q):
    pi_half = np.pi / 2
    T0 = np.eye(4)
    T1 = T0 @ std_dh(q[0], L1, L3,  pi_half)     
    T2 = T1 @ std_dh(q[1],  0, L2, 0)          
    T3 = T2 @ std_dh(q[2],  0, L4, pi_half)    
    T4 = T3 @ std_dh(q[3], L5,  0, -pi_half)   
    T5 = T4 @ std_dh(q[4],  0,  0, pi_half)    
    T6 = T5 @ std_dh(q[5], L6+L7, 0, 0) 
    P_curr = T6[:3, 3] 
    J = np.zeros((6, 6))
    Transforms = [T0, T1, T2, T3, T4, T5]
    for i in range(6):
        Z_i = Transforms[i][:3, 2]
        P_i = Transforms[i][:3, 3]
        J[:3, i] = np.cross(Z_i, P_curr - P_i) 
        J[3:, i] = Z_i                         
    return J, P_curr

def calculate_ik_angles(target_xyz, initial_angles_deg):
    """🔥 IK ฉบับ 'กันตาย': ล็อกข้อมือชี้ลงพื้น 100% ป้องกันแขนควงสว่าน"""
    q_curr = np.radians(initial_angles_deg)
    target = np.array(target_xyz)
    
    for _ in range(500): 
        # 💡 ยาแรง: ล็อกข้อมือทุกรอบการคำนวณ!
        # Joint 4 (หมุนข้อมือ) = 0
        # Joint 5 (พับข้อมือ) = -90 (ชี้ลงพื้นเป๊ะๆ)
        # Joint 6 (ปลายกริปเปอร์) = 0
        q_curr[3] = 0.0
        q_curr[4] = np.radians(-90.0)
        q_curr[5] = 0.0
        
        J, P_curr = get_jacobian_and_pos(q_curr)
        error = target - P_curr
        
        if np.linalg.norm(error) < 1.0: 
            break # ถ้าแม่นยำระดับ 1 mm แล้วให้หยุด
            
        # เราจะใช้ Jacobian แค่ 3 ข้อต่อแรก (J1, J2, J3) มาขยับ XYZ
        V_cmd = error * 5.0
        J_pos = J[:3, :3] 
        
        lam = 0.2
        J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + (lam**2) * np.eye(3))
        q_dot = J_dls @ V_cmd
        
        # ขยับแค่ 3 ข้อต่อหลัก
        q_curr[0] += q_dot[0] * 0.05
        q_curr[1] += q_dot[1] * 0.05
        q_curr[2] += q_dot[2] * 0.05
        
    return np.degrees(q_curr).tolist()

# ==============================================================================
# 3. LFPB TRAJECTORY ENGINE (Rubric 2)
# ==============================================================================
def lfpb_step(t, q0, qf, tf, tb):
    if tf == 0: return qf
    accel = (qf - q0) / (tb * (tf - tb))
    if 0 <= t <= tb:
        return q0 + 0.5 * accel * (t**2)
    elif tb < t <= (tf - tb):
        return q0 + accel * tb * (t - tb/2)
    elif (tf - tb) < t <= tf:
        return qf - 0.5 * accel * (tf - t)**2
    return qf

def move_lfpb_sync(target_angles, duration):
    global current_angles
    start_angles = current_angles.copy()
    tb = duration * 0.2  
    start_sim_time = sim.getSimulationTime()
    
    while True:
        elapsed = sim.getSimulationTime() - start_sim_time
        if elapsed >= duration: break
        for i in range(6):
            q_next = lfpb_step(elapsed, start_angles[i], target_angles[i], duration, tb)
            sim.setJointPosition(joints[i], m.radians(q_next))
            current_angles[i] = q_next
        step_sim() # ใช้ฟังก์ชันเก็บข้อมูลกราฟ
        
    current_angles = np.array(target_angles)
    for i in range(6): sim.setJointPosition(joints[i], m.radians(current_angles[i]))

# ==============================================================================
# 4. DEFINE TARGETS & CALCULATE IK (หยิบจากด้านบนแบบถูกต้อง 100%)
# ==============================================================================
# 📍 พิกัดจุดศูนย์กลางแก้วน้ำที่วินาที 7.5
CUP_X = 272.25   
CUP_Y = -561.54  
CUP_Z = -156.57  

# 📏 ชดเชยความยาวกริปเปอร์ในแนวตั้ง (Z-Axis Offset)
# ถ้ากริปเปอร์ยังจิ้มลึกไป ให้เพิ่มเลขนี้ ถ้าหนีบลอยไป ให้ลดเลขนี้ครับ
GRIPPER_LENGTH = 160.0 

# 🎯 เป้าหมายข้อมือ: ให้อยู่จุดเดียวกับแก้ว แต่ "สูงกว่า" เท่ากับความยาวกริปเปอร์!
PICK_X = CUP_X
PICK_Y = CUP_Y
PICK_Z = CUP_Z + GRIPPER_LENGTH  
HOVER_Z = PICK_Z + 150.0 # ลอยรอเหนือแก้ว 15 ซม.

# 📍 เป้าหมายฝั่งวาง (Place) --- สายพานเส้นตรง
PLACE_CUP_X = -300.0  
PLACE_CUP_Y = -400.0  
PLACE_CUP_Z = 0.0     

# ต้องบวกความสูงกริปเปอร์ตอนวางด้วยเช่นกัน
PLACE_X = PLACE_CUP_X
PLACE_Y = PLACE_CUP_Y
PLACE_Z = PLACE_CUP_Z + GRIPPER_LENGTH

print(f"🧠 เป้าหมายข้อมือฝั่ง Pick: X={PICK_X:.1f}, Y={PICK_Y:.1f}, Z={PICK_Z:.1f}")

# 1. ท่าเริ่มต้นและท่าเตรียมพร้อม
home_angles = [0.0, 20.0, 20.0, 0.0, -90.0, 0.0] 
ready_angles = [0.0, 20.0, 20.0, 0.0, -90.0, 0.0]

# คำนวณมุมที่แก้วน้ำอยู่เทียบกับฐาน
angle_to_pick = m.degrees(m.atan2(PICK_Y, PICK_X))   # จะได้ประมาณ -64 องศา
angle_to_place = m.degrees(m.atan2(PLACE_Y, PLACE_X)) # จะได้ประมาณ -126 องศา

# ให้คำใบ้ฝั่ง Pick: หันหน้าตรงไปที่แก้ว, ศอกยกขึ้น 40, ข้อมือชี้ลงพื้น -90
guess_pick_angles = [angle_to_pick, 40.0, -40.0, 0.0, -90.0, 0.0] 
pick_approach_angles = calculate_ik_angles([PICK_X, PICK_Y, HOVER_Z], guess_pick_angles)
pick_angles          = calculate_ik_angles([PICK_X, PICK_Y, PICK_Z], pick_approach_angles)

# ให้คำใบ้ฝั่ง Place: หันหน้าไปที่สายพานตรง
guess_place_angles = [angle_to_place, 40.0, -40.0, 0.0, -90.0, 0.0]
place_approach_angles = calculate_ik_angles([PLACE_X, PLACE_Y, HOVER_Z], guess_place_angles)
place_angles          = calculate_ik_angles([PLACE_X, PLACE_Y, PLACE_Z], place_approach_angles)

print("✅ คำนวณเสร็จสิ้น! ล็อกพิกัดไม่ให้แขนสวิงมั่วแล้ว")

# ==============================================================================
# 5. MAIN SEQUENCE (RUN SIMULATION)
# ==============================================================================
try:
    sim.startSimulation()
    print("🚀 เริ่มลุย Pick & Place!")

    # เซ็ตค่าเริ่มต้นให้ตรงกับ Home
    current_angles = np.array(home_angles)
    for i in range(6): sim.setJointPosition(joints[i], m.radians(home_angles[i]))
    for _ in range(10): step_sim()

    # เปิดกริปเปอร์รอ
    current_vgripper = 0.15 
    sim.setJointTargetVelocity(mico_motor1, current_vgripper)
    sim.setJointTargetVelocity(mico_motor2, current_vgripper)

    # 🌟 สเตปที่เพิ่มมา: วอร์มแขนเข้าท่า Ready ก่อน (ใช้เวลา 2 วินาที)
    print("📍 ขยับเข้าท่าเตรียมพร้อม (หลบ Singularity)...")
    move_lfpb_sync(ready_angles, 2.0)

    # 1. ไปรอเหนือแก้ว (ใช้เวลา 2 วิ)
    print("📍 เคลื่อนที่ไปดักรอเหนือแก้ว...")
    move_lfpb_sync(pick_approach_angles, 2.0)
    
    # 2. ดักรอเวลา! - รอจนถึงวินาทีที่ 6.5 (หักเวลาที่เดินทางมาแล้วออก)
    print("⏳ รอให้แก้วน้ำวิ่งเข้ามาในระยะเป้าหมาย (วินาทีที่ 7.5)...")
    while sim.getSimulationTime() < 6.5:
        step_sim()
        
    # 3. โฉบลงไปหยิบ (ใช้เวลา 1 วินาที -> จะไปถึงแก้วตอน 7.5 พอดีเป๊ะ!)
    print("📍 สับมือลงไปหยิบ!")
    move_lfpb_sync(pick_angles, 1.0)
    
    # ... (หลังจากนี้ก็เป็นโค้ดหนีบแก้ว ยกแก้ว ไปวาง เหมือนเดิมเลยครับ) ...
    
    # 4. หนีบแก้ว
    print("🔒 หนีบกริปเปอร์...")
    current_vgripper = -0.15 # เปลี่ยนความเร็วเป็นติดลบ (ปิดมือ)
    sim.setJointTargetVelocity(mico_motor1, current_vgripper)
    sim.setJointTargetVelocity(mico_motor2, current_vgripper)
    for _ in range(20): step_sim() # รอ 1 วิ ให้หนีบแน่น
    
    # 5. ยกแก้วขึ้น
    move_lfpb_sync(pick_approach_angles, 1.5)
    
    # 6. ย้ายไปฝั่งวาง
    print("📍 ย้ายแก้วไปที่สายพานเป้าหมาย...")
    move_lfpb_sync(place_approach_angles, 3.0)
    
    # 7. วางแก้วลง
    move_lfpb_sync(place_angles, 1.5)
    
    # 8. ปล่อยกริปเปอร์
    print("🔓 ปล่อยแก้ว...")
    current_vgripper = 0.15 # เปลี่ยนความเร็วเป็นบวก (เปิดมือ)
    sim.setJointTargetVelocity(mico_motor1, current_vgripper)
    sim.setJointTargetVelocity(mico_motor2, current_vgripper)
    for _ in range(20): step_sim() 
    
    # 9. ยกแขนขึ้น
    move_lfpb_sync(place_approach_angles, 1.5)
    
    # 10. กลับ Home
    move_lfpb_sync(home_angles, 3.0)
    
    print("🎉 ภารกิจสำเร็จ! เตรียมดูกราฟ...")

finally:
    sim.setStepping(False)
    sim.stopSimulation()

# ==============================================================================
# 6. PLOT GRIPPER VELOCITY (Rubric 4.2)
# ==============================================================================
plt.figure(figsize=(8, 4))
plt.plot(time_log, vgripper_log, 'g-', linewidth=2)
plt.title("Gripper Velocity vs Time (Rubric 4.2)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.fill_between(time_log, vgripper_log, 0, alpha=0.3, color='green')
plt.show()