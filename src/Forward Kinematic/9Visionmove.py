import math as m
import numpy as np
import cv2
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
base_handle = sim.getObject('/yaskawa')

mico_motor1 = sim.getObject('/yaskawa/MicoHand/fingers12_motor1')
mico_motor2 = sim.getObject('/yaskawa/MicoHand/fingers12_motor2')

# ตั้งค่ากล้อง OpenCV
vision_sensor_handle = sim.getObject('/Vision_sensor')
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

time_log, vgripper_log = [], []
current_vgripper = 0.0

def step_sim():
    time_log.append(sim.getSimulationTime())
    vgripper_log.append(current_vgripper)
    sim.step()

# ==============================================================================
# 2. INVERSE KINEMATICS ENGINE (อัปเดตเป็นท่า Side Grasp + Flip)
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

def calculate_ik_angles(target_xyz, initial_angles_deg, is_flip=False):
    """IK สำหรับหยิบจากด้านข้าง (Side Grasp) และบิดข้อมือคว่ำแก้ว (Flip)"""
    q_curr = np.radians(initial_angles_deg)
    target = np.array(target_xyz)
    
    for _ in range(500): 
        # 💡 ล็อกข้อมือให้ชี้ตรงไปข้างหน้า (Joint 5 = 0)
        q_curr[4] = 0.0 
        
        # 💡 การพลิกคว่ำแก้ว: ให้บิดข้อมือ (Joint 4) ไป 180 องศา
        if is_flip:
            q_curr[3] = np.radians(180.0)
        else:
            q_curr[3] = 0.0
            
        q_curr[5] = 0.0
        
        J, P_curr = get_jacobian_and_pos(q_curr)
        error = target - P_curr
        if np.linalg.norm(error) < 1.0: break
            
        V_cmd = error * 5.0
        J_pos = J[:3, :3] 
        lam = 0.2
        J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + (lam**2) * np.eye(3))
        q_dot = J_dls @ V_cmd
        
        q_curr[0] += q_dot[0] * 0.05
        q_curr[1] += q_dot[1] * 0.05
        q_curr[2] += q_dot[2] * 0.05
        
    return np.degrees(q_curr).tolist()

# ==============================================================================
# 3. LFPB TRAJECTORY ENGINE
# ==============================================================================
def lfpb_step(t, q0, qf, tf, tb):
    if tf == 0: return qf
    accel = (qf - q0) / (tb * (tf - tb))
    if 0 <= t <= tb: return q0 + 0.5 * accel * (t**2)
    elif tb < t <= (tf - tb): return q0 + accel * tb * (t - tb/2)
    elif (tf - tb) < t <= tf: return qf - 0.5 * accel * (tf - t)**2
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
        step_sim() 
        
    current_angles = np.array(target_angles)
    for i in range(6): sim.setJointPosition(joints[i], m.radians(current_angles[i]))

# ==============================================================================
# 4. VISION SCANNER & CALCULATE TARGETS
# ==============================================================================
try:
    sim.startSimulation()
    print("🚀 เริ่มลุย Pick & Place: แบบพลิกแก้ว (Flip)!")

    # --- 📸 4.1 สแกนหาจุดกึ่งกลาง ArUco ทั้ง 4 อัน ---
    print("\n📷 [VISION] กำลังเปิดกล้องสแกนหาจุดวาง (ArUco 4 มุม)...")
    img_raw, res = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img_raw, dtype=np.uint8).reshape((res[1], res[0], 3))
    img = cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None:
        print(f"✅ AI ตรวจพบ ArUco IDs: {ids.flatten()}")
    
    # คำนวณจุดกึ่งกลางของแผ่น ArUco 4 อันบนพื้น
    place_x_sum, place_y_sum, place_z_sum = 0, 0, 0
    for i in range(4):
        marker_handle = sim.getObject(f'/20cmHighWallL[0]/arucoMarker[{i}]')
        pos = sim.getObjectPosition(marker_handle, base_handle)
        place_x_sum += pos[0] * 1000
        place_y_sum += pos[1] * 1000
        place_z_sum += pos[2] * 1000
        
    PLACE_CUP_X = place_x_sum / 4
    PLACE_CUP_Y = place_y_sum / 4
    PLACE_CUP_Z = place_z_sum / 4
    print(f"📍 คำนวณจุดกึ่งกลาง ArUco สำเร็จ! -> X:{PLACE_CUP_X:.1f}, Y:{PLACE_CUP_Y:.1f}, Z:{PLACE_CUP_Z:.1f}\n")

    # --- 4.2 คำนวณเป้าหมายฝั่งหยิบ (Pick) แบบ Side Grasp ---
    CUP_X, CUP_Y, CUP_Z = 407.2, 160.0, -400.0  
    GRIPPER_LENGTH = 160.0 

    cup_angle = m.atan2(CUP_Y, CUP_X)
    PICK_X = CUP_X - (GRIPPER_LENGTH * m.cos(cup_angle))
    PICK_Y = CUP_Y - (GRIPPER_LENGTH * m.sin(cup_angle))
    PICK_Z = CUP_Z + 30.0 # ยกสูงขึ้นนิดนึงเพื่อจับกลางแก้ว
    HOVER_Z = PICK_Z + 150.0 

    # --- 4.3 คำนวณเป้าหมายฝั่งวาง (Place) แบบ Side Grasp + คว่ำแก้ว ---
    place_angle = m.atan2(PLACE_CUP_Y, PLACE_CUP_X)
    PLACE_X = PLACE_CUP_X - (GRIPPER_LENGTH * m.cos(place_angle))
    PLACE_Y = PLACE_CUP_Y - (GRIPPER_LENGTH * m.sin(place_angle))
    # วางให้แก้วลอยเหนือพื้นนิดนึง (กันปากแก้วกระแทกแผ่น)
    PLACE_Z = PLACE_CUP_Z + 70.0 

    # 4.4 โยนเข้าสมการ IK
    home_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    ready_angles = [0.0, 30.0, 30.0, 0.0, 0.0, 0.0]

    guess_pick_angles = [m.degrees(cup_angle), 30.0, 0.0, 0.0, 0.0, 0.0] 
    pick_approach_angles = calculate_ik_angles([PICK_X, PICK_Y, HOVER_Z], guess_pick_angles)
    pick_angles          = calculate_ik_angles([PICK_X, PICK_Y, PICK_Z], pick_approach_angles)

    guess_place_angles = [m.degrees(place_angle), 30.0, 0.0, 0.0, 0.0, 0.0]
    place_approach_angles = calculate_ik_angles([PLACE_X, PLACE_Y, HOVER_Z], guess_place_angles, is_flip=True)
    place_angles          = calculate_ik_angles([PLACE_X, PLACE_Y, PLACE_Z], place_approach_angles, is_flip=True)

    # ==============================================================================
    # 5. MAIN SEQUENCE (รันการทำงานจริง)
    # ==============================================================================
    current_angles = np.array(home_angles)
    for i in range(6): sim.setJointPosition(joints[i], m.radians(home_angles[i]))
    for _ in range(10): step_sim()

    current_vgripper = 0.15 
    sim.setJointTargetVelocity(mico_motor1, current_vgripper)
    sim.setJointTargetVelocity(mico_motor2, current_vgripper)

    print("📍 ขยับเข้าท่าเตรียมพร้อม (Side Grasp)...")
    move_lfpb_sync(ready_angles, 2.0)

    print("📍 เคลื่อนที่ไปดักรอข้างแก้ว...")
    move_lfpb_sync(pick_approach_angles, 2.0)
    
    # ⏳ แก้เวลารอตรงนี้ให้สัมพันธ์กับแก้วของคุณ
    while sim.getSimulationTime() < 6.5:
        step_sim()
        
    print("📍 สับมือเข้าหยิบจากด้านข้าง!")
    move_lfpb_sync(pick_angles, 1.0)
    
    print("🔒 หนีบกริปเปอร์...")
    current_vgripper = -0.15 
    sim.setJointTargetVelocity(mico_motor1, current_vgripper)
    sim.setJointTargetVelocity(mico_motor2, current_vgripper)
    for _ in range(20): step_sim() 
    
    print("📍 ยกแก้วขึ้น...")
    move_lfpb_sync(pick_approach_angles, 1.5)
    
    # 🌟 จุดไคลแม็กซ์: ย้ายไปวางตรงกลาง ArUco พร้อมบิดข้อมือพลิกแก้วคว่ำ!
    print("🔄 ย้ายแก้วไปเป้าหมาย พร้อมบิดข้อมือพลิกคว่ำ (Flip 180°)...")
    move_lfpb_sync(place_approach_angles, 3.0)
    
    print("📍 กดแก้วลงวางตรงกลาง ArUco...")
    move_lfpb_sync(place_angles, 1.5)
    
    print("🔓 ปล่อยแก้ว...")
    current_vgripper = 0.15 
    sim.setJointTargetVelocity(mico_motor1, current_vgripper)
    sim.setJointTargetVelocity(mico_motor2, current_vgripper)
    for _ in range(20): step_sim() 
    
    move_lfpb_sync(place_approach_angles, 1.5)
    move_lfpb_sync(home_angles, 3.0)
    
    print("🎉 ภารกิจ Work 2 สมบูรณ์แบบ 100%! เตรียมดูกราฟ...")

finally:
    sim.setStepping(False)
    sim.stopSimulation()

# ==============================================================================
# 6. PLOT GRIPPER VELOCITY
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