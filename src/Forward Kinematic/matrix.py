from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

print("📡 Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

# ==========================================
# 1. ดึง Handles
# ==========================================
j_handles = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
ef_handle = sim.getObject('/yaskawa/gripperEF')

# ดึงมุมปัจจุบันจากหุ่น (Radian)
q = [sim.getJointPosition(h) for h in j_handles]

# ==========================================
# 2. ปริ้นท์ DH Table แบบ Live ลง Terminal
# ==========================================
print("\n" + "="*75)
print("📊 1. DH Table (Yaskawa GP8) - มุม q คือค่าที่อ่านได้แบบ Real-time")
print("="*75)
print("| Link |  alpha (rad) |   a (m)  |   d (m)  |   theta (rad)        |")
print("-" * 75)
print(f"|  1   |    -1.5708   |  0.0400  |  0.3300  | q1 = {q[0]:>13.4f}  |")
print(f"|  2   |     0.0000   |  0.3450  |  0.0000  | q2 = {(q[1] - np.pi/2):>13.4f}  |") 
print(f"|  3   |    -1.5708   |  0.0400  |  0.0000  | q3 = {q[2]:>13.4f}  |")
print(f"|  4   |     1.5708   |  0.0000  |  0.3400  | q4 = {q[3]:>13.4f}  |")
print(f"|  5   |    -1.5708   |  0.0000  |  0.0000  | q5 = {q[4]:>13.4f}  |")
print(f"|  6   |     0.0000   |  0.0000  |  0.2413  | q6 = {q[5]:>13.4f}  |")
print("="*75)

# ==========================================
# 3. คำนวณ Transformation Matrix ย่อย (DH)
# ==========================================
def dh_matrix(alpha, a, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

T_Base = np.array([
    [1, 0, 0, 0.0187],
    [0, 1, 0, 0.0000],
    [0, 0, 1, 0.0000],
    [0, 0, 0, 1.0000]
])

T01 = dh_matrix(-np.pi/2, 0.040, 0.330,  q[0])
T12 = dh_matrix(0,        0.345, 0,      q[1] - np.pi/2)
T23 = dh_matrix(-np.pi/2, 0.040, 0,      q[2])
T34 = dh_matrix(np.pi/2,  0,     0.340,  q[3])
T45 = dh_matrix(-np.pi/2, 0,     0,      q[4])
T56 = dh_matrix(0,        0,     0.2413, q[5])

# คูณรวดเดียวหา T06
T06_DH = T_Base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56

# ==========================================
# 4. ฟังก์ชันทำความสะอาด Matrix (ลบ -0.0) และปริ้นท์ทีละข้อต่อ
# ==========================================
np.set_printoptions(precision=4, suppress=True)

def clean_mat(mat):
    m = mat.copy() # copy ไว้ จะได้ไม่กระทบผลลัพธ์หลัก
    m[np.abs(m) < 1e-5] = 0.0
    return m

print("\n🤖 2. Transformation Matrices ย่อย (T01 ถึง T56):")
print("-" * 50)
matrices = [("T_Base (Offset ฐาน)", T_Base), ("T01", T01), ("T12", T12), 
            ("T23", T23), ("T34", T34), ("T45", T45), ("T56", T56)]

for name, mat in matrices:
    print(f"[{name}] =")
    print(clean_mat(mat))
    print("-" * 30)

# ==========================================
# 5. Get TF Matrix จาก CoppeliaSim (World -> End-Effector)
# ==========================================
m = sim.getObjectMatrix(ef_handle, sim.handle_world)
T06_Sim = np.array([
    [m[0], m[1], m[2],  m[3]],
    [m[4], m[5], m[6],  m[7]],
    [m[8], m[9], m[10], m[11]],
    [0.0,  0.0,  0.0,   1.0]
])

# ==========================================
# 6. ปริ้นท์เทียบ T06 (DH vs Sim)
# ==========================================
print("\n🧮 3. TF Matrix รวม (T06_DH) จากทฤษฎี:")
print("-" * 50)
print(clean_mat(T06_DH))

print("\n🎯 4. TF Matrix รวม (T06_Sim) จาก CoppeliaSim:")
print("-" * 50)
print(clean_mat(T06_Sim))

# เทียบ Error ของ Position
pos_dh = T06_DH[:3, 3]
pos_sim = T06_Sim[:3, 3]
pos_err = np.linalg.norm(pos_dh - pos_sim) * 1000

print("\n🔍 5. สรุปผล Position (End-Effector):")
print("-" * 50)
print(f"📍 DH  X,Y,Z : [{pos_dh[0]:>7.4f}, {pos_dh[1]:>7.4f}, {pos_dh[2]:>7.4f}]")
print(f"📍 Sim X,Y,Z : [{pos_sim[0]:>7.4f}, {pos_sim[1]:>7.4f}, {pos_sim[2]:>7.4f}]")
print(f"⚠️ Position Error = {pos_err:.4f} มิลลิเมตร")
print("="*75)