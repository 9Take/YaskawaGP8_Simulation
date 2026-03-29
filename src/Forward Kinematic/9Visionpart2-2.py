import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# =========================================================
# 🔌 1. เชื่อมต่อและดึงชิ้นส่วนจาก CoppeliaSim ก่อน!
# =========================================================
print("🔌 เชื่อมต่อกับ CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

# ดึง Handle ของหุ่นยนต์มาเก็บไว้ในตัวแปร
base_handle = sim.getObject('/yaskawa')
ef_handle = sim.getObject('/yaskawa/gripperEF')
joints = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]

# =========================================================
# 🎯 2. ตั้งค่าเป้าหมาย (พิกัด Pick)
# =========================================================
TARGET_X = 0.3897 
TARGET_Y = 0.0623 
TARGET_Z = -0.5609 

GRIPPER_OFFSET = 0.16  
angle_to_target = np.arctan2(TARGET_Y, TARGET_X)
target_pos = np.array([
    TARGET_X - (GRIPPER_OFFSET * np.cos(angle_to_target)),
    TARGET_Y - (GRIPPER_OFFSET * np.sin(angle_to_target)),
    TARGET_Z
])

# =========================================================
# 🧠 3. ฟังก์ชันสร้าง Jacobian แบบ "แกล้งขยับแล้ววัดผล" (Foolproof)
# =========================================================
def get_jacob_from_sim(q_angles):
    # ยัดมุมปัจจุบันเข้าไปก่อน เพื่อหาจุดตั้งต้น
    for i in range(6): 
        sim.setJointPosition(joints[i], float(q_angles[i]))
    
    P_cur = np.array(sim.getObjectPosition(ef_handle, base_handle))
    
    J = np.zeros((3, 6))
    DELTA = 0.001  # ขยับหลอกๆ ทีละ 1 มิลลิเรเดียน
    
    for i in range(6):
        # 1. แกล้งดันข้อต่อที่ i ไปข้างหน้า แล้วจดพิกัดไว้
        sim.setJointPosition(joints[i], float(q_angles[i] + DELTA))
        P_plus = np.array(sim.getObjectPosition(ef_handle, base_handle))
        
        # 2. แกล้งดึงข้อต่อที่ i ถอยหลัง แล้วจดพิกัดไว้
        sim.setJointPosition(joints[i], float(q_angles[i] - DELTA))
        P_minus = np.array(sim.getObjectPosition(ef_handle, base_handle))
        
        # 3. คืนค่าข้อต่อกลับที่เดิม!
        sim.setJointPosition(joints[i], float(q_angles[i]))
        
        # 4. คำนวณความชัน (Jacobian) = ระยะทางที่เปลี่ยนไป / มุมที่เปลี่ยนไป
        J[:, i] = (P_plus - P_minus) / (2.0 * DELTA)
        
    return J, P_cur

# =========================================================
# ⚙️ 4. เริ่มลูปคำนวณ IK
# =========================================================
q_cur = np.array([angle_to_target, np.radians(30), 0.0, 0.0, 0.0, 0.0])

print("\n⚙️ กำลังคำนวณ IK ด้วยพลัง Transformation Matrix จาก Sim...")

for step in range(500):
    # ล็อกข้อมือให้ตั้งตรง (ท่า Pick)
    q_cur[4] = 0.0 
    q_cur[5] = 0.0 
    
    J, P_cur = get_jacob_from_sim(q_cur)
    error = target_pos - P_cur
    
    # ถ้าพิกัดห่างจากเป้าหมายน้อยกว่า 1 มิลลิเมตร ถือว่าสำเร็จ
    if np.linalg.norm(error) < 0.001: 
        print(f"✅ สำเร็จที่ลูป {step}")
        break
        
    # ดึง Jacobian มาแก้สมการ 3x3
    J_pos = J[:3, :3] 
    J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + (0.01**2) * np.eye(3))
    q_dot = J_dls @ (error * 2.0)
    q_cur[:3] += q_dot * 0.1

# =========================================================
# 📊 5. แสดงผลลัพธ์
# =========================================================
j_deg = np.degrees(q_cur)
print("\n🎯 === ผลลัพธ์มุมข้อต่อ (j1-j6) ===")
print(f"j1 : {j_deg[0]:+7.2f} องศา")
print(f"j2 : {j_deg[1]:+7.2f} องศา")
print(f"j3 : {j_deg[2]:+7.2f} องศา")
print(f"j4 : {j_deg[3]:+7.2f} องศา")
print(f"j5 : {j_deg[4]:+7.2f} องศา")
print(f"j6 : {j_deg[5]:+7.2f} องศา")
print("==================================")