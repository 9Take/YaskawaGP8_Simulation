import numpy as np

# =========================================================
# 📝 1. เอาค่า X, Y, Z จากตาราง Input table 1 มาใส่ตรงนี้ครับ!
# (หน่วยเป็นเมตร ตามที่กล้องสแกนได้เลยครับ)
# =========================================================
TARGET_X = 0.4072  # <--- เปลี่ยนเลขตรงนี้
TARGET_Y = 0.1600  # <--- เปลี่ยนเลขตรงนี้
TARGET_Z = -0.4000  # <--- เปลี่ยนเลขตรงนี้ (แก้วน้ำบนแท่น)

# ความยาวกริปเปอร์ (ชดเชยเพื่อให้จับพอดี ไม่กระแทก)
GRIPPER_OFFSET = 0.16  

# คำนวณจุดที่ข้อมือ (Joint 6) ต้องอยู่เวลาหนีบแก้ว
angle_to_target = np.arctan2(TARGET_Y, TARGET_X)
WRIST_X = TARGET_X - (GRIPPER_OFFSET * np.cos(angle_to_target))
WRIST_Y = TARGET_Y - (GRIPPER_OFFSET * np.sin(angle_to_target))
WRIST_Z = TARGET_Z 

target_pos = np.array([WRIST_X, WRIST_Y, WRIST_Z])

# =========================================================
# 🧠 2. สมการ Inverse Kinematics (Jacobian)
# =========================================================
L1, L2, L3, L4, L5, L6, L7 = 0.330, 0.345, 0.040, 0.040, 0.340, 0.080, 0.16133

def std_dh(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def get_jacob_pos(q):
    pi_h = np.pi / 2
    T0 = np.eye(4)
    T1 = T0 @ std_dh(q[0], L1, L3,  pi_h)     
    T2 = T1 @ std_dh(q[1],  0, L2, 0)          
    T3 = T2 @ std_dh(q[2],  0, L4, pi_h)    
    T4 = T3 @ std_dh(q[3], L5,  0, -pi_h)   
    T5 = T4 @ std_dh(q[4],  0,  0, pi_h)    
    T6 = T5 @ std_dh(q[5], L6+L7, 0, 0) 
    P = T6[:3, 3] 
    J = np.zeros((3, 6))
    Transforms = [T0, T1, T2, T3, T4, T5]
    for i in range(6):
        J[:, i] = np.cross(Transforms[i][:3, 2], P - Transforms[i][:3, 3])
    return J, P

# เดาค่าเริ่มต้น (หันหน้าไปหาเป้าหมาย, ยกศอกนิดๆ, ข้อมือชี้ตรง)
q_cur = np.array([angle_to_target, np.radians(30), 0.0, 0.0, 0.0, 0.0])

print("\n⚙️ กำลังคำนวณ Inverse Kinematics...")
for _ in range(500):
    q_cur[4] = 0.0 # ล็อกข้อ 5 ชี้ตรง (Side Grasp)
    q_cur[5] = 0.0 # ล็อกข้อ 6
    
    J, P_cur = get_jacob_pos(q_cur)
    error = target_pos - P_cur
    
    if np.linalg.norm(error) < 0.001: # แม่นยำระดับ 1 มิลลิเมตร
        break
        
    J_pos = J[:3, :3] # ใช้แค่ 3 แกนแรกมาขยับแขน
    J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + (0.01**2) * np.eye(3))
    q_dot = J_dls @ (error * 2.0)
    
    q_cur[0:3] += q_dot * 0.1

# =========================================================
# 🎯 3. ผลลัพธ์สำหรับ Output Table 1
# =========================================================
j_deg = np.degrees(q_cur)
print("\n🎯 === ข้อมูลสำหรับ Output table 1 (หน้า 5) ===")
print(f"j1 : {j_deg[0]:+7.2f} องศา")
print(f"j2 : {j_deg[1]:+7.2f} องศา")
print(f"j3 : {j_deg[2]:+7.2f} องศา")
print(f"j4 : {j_deg[3]:+7.2f} องศา")
print(f"j5 : {j_deg[4]:+7.2f} องศา")
print(f"j6 : {j_deg[5]:+7.2f} องศา")
print("===========================================")