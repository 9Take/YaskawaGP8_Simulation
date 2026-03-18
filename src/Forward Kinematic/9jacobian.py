import numpy as np

# ==========================================
# 1. พารามิเตอร์หุ่นยนต์ และฟังก์ชัน DH
# ==========================================
L1, L2, L3, L4, L5 = 330, 345, 40, 40, 340
L6, L7 = 80, 161.33

def std_dh(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def get_jacobian_and_pos(q):
    t1, t2, t3, t4, t5, t6 = q
    pi_half = np.pi / 2
    
    T0 = np.eye(4)
    T1 = T0 @ std_dh(t1, L1, L3,  pi_half)     
    T2 = T1 @ std_dh(t2,  0, L2, 0)          
    T3 = T2 @ std_dh(t3,  0, L4, pi_half)    
    T4 = T3 @ std_dh(t4, L5,  0, -pi_half)   
    T5 = T4 @ std_dh(t5,  0,  0, pi_half)    
    T6 = T5 @ std_dh(t6, L6+L7, 0, 0) 
    
    P_curr = T6[:3, 3] 
    
    J = np.zeros((6, 6))
    Transforms = [T0, T1, T2, T3, T4, T5]
    for i in range(6):
        Z_i = Transforms[i][:3, 2]
        P_i = Transforms[i][:3, 3]
        J[:3, i] = np.cross(Z_i, P_curr - P_i) 
        J[3:, i] = Z_i                         
    return J, P_curr

# ==========================================
# 2. จำลอง Trajectory (Wait and Pick) - แก้ไขแขนคด
# ==========================================
step = 380
dt = 0.05
t_vals = [i * dt for i in range(step)]

# 🎯 พิกัดเป้าหมายของคุณ (อัปเดตแล้ว)
px_target = 353.76
py_target = -515.39
pz_pick = -639.88
pz_hover = pz_pick + 150.0  # ลอยรอเหนือแก้ว 150 mm

print("กำลังคำนวณหามุมข้อต่อ j1 ถึง j6... (อาจใช้เวลาไม่กี่วินาที)")

# 💡 แก้ไข 1: ตั้งท่าเริ่มต้นใหม่ ให้แขนยื่นไปข้างหน้าและมือจับชี้ลงพื้น
q_curr = np.array([0.0, np.radians(45), np.radians(-30), 0.0, np.radians(-90), 0.0])

# 💡 แก้ไข 2: ให้โปรแกรมคำนวณพิกัด p0 (Home) จาก q_curr อัตโนมัติ! แขนจะได้ไม่กระตุก
_, P_start = get_jacobian_and_pos(q_curr)
p0x, p0y, p0z = P_start[0], P_start[1], P_start[2]

# สร้างจุด Waypoints
p0 = np.array([p0x, p0y, p0z])                          # จุด Home (คำนวณอัตโนมัติ)
p1 = np.array([px_target, py_target, pz_hover])         # จุดไปรอเหนือแก้ว
p2 = np.array([px_target, py_target, pz_pick])          # โฉบลงมาหยิบแก้ว
p3 = np.array([px_target, py_target, pz_hover])         # ยกแก้วขึ้นตรงๆ
p4 = np.array([p0x, p0y, p0z])                          # กลับ Home

j1_list, j2_list, j3_list, j4_list, j5_list, j6_list = [], [], [], [], [], []

px, py, pz = p0[0], p0[1], p0[2]

for t in t_vals:
    # (ส่วนโค้ดคำนวณ Jacobian ด้างล่างใช้เหมือนเดิมได้เลยครับ)
    if t >= 12.10:
        vx, vy, vz = (p4 - p3) / 5.0    
    elif t >= 10.10:
        vx, vy, vz = (p3 - p2) / 2.0    
    elif t >= 8.10:
        vx, vy, vz = (p2 - p1) / 2.0    
    else:
        vx, vy, vz = (p1 - p0) / 8.10   
        
    px += vx * dt; py += vy * dt; pz += vz * dt
    P_target = np.array([px, py, pz])
    V_target = np.array([vx, vy, vz])

    J, P_curr = get_jacobian_and_pos(q_curr)
    
    # ดึงให้ตรงพิกัดเป้าหมาย (ลดความดุดันลงนิดนึงเพื่อไม่ให้กระชาก)
    Kp = 3.0 
    V_error = Kp * (P_target - P_curr)
    
    # ยังคงคุมให้ข้อมือพยายามชี้ลงล่าง
    V_cmd = np.concatenate([(V_target + V_error), [0.0, 0.0, 0.0]])
    
    # 💡 พระเอกขี่ม้าขาว: ใช้ Damped Least Squares (DLS) แทนการทำ Pseudo-inverse ปกติ
    # ช่วยป้องกันสมการระเบิดเวลาแขนยืดตึงสุด หรือเจอจุด Singularity
    lam = 0.1  # ค่าความยืดหยุ่น ยิ่งมากแขนยิ่งสมูทแต่จะตามเป้าหมายคลาดเคลื่อนนิดนึง
    J_dls = J.T @ np.linalg.inv(J @ J.T + (lam**2) * np.eye(6))
    
    # ใช้ J_dls แทน J_pinv
    q_dot = J_dls @ V_cmd
    
    q_curr = q_curr + (q_dot * dt)
    
    j1_list.append(round(q_curr[0], 4))
    j2_list.append(round(q_curr[1], 4))
    j3_list.append(round(q_curr[2], 4))
    j4_list.append(round(q_curr[3], 4))
    j5_list.append(round(q_curr[4], 4))
    j6_list.append(round(q_curr[5], 4))

print("\n=== คัดลอก List ด้านล่างไปใช้ในโค้ดขยับแขนได้เลย! ===")
print(f"j1 = {j1_list}\n")
print(f"j2 = {j2_list}\n")
print(f"j3 = {j3_list}\n")
print(f"j4 = {j4_list}\n")
print(f"j5 = {j5_list}\n")
print(f"j6 = {j6_list}\n")