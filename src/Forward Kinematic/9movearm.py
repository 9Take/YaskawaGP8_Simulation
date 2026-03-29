import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print("🔌 เชื่อมต่อกับ CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

# เชื่อมต่อชิ้นส่วนต่างๆ
joints = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
mico_motor1 = sim.getObject('/yaskawa/MicoHand/fingers12_motor1')
mico_motor2 = sim.getObject('/yaskawa/MicoHand/fingers12_motor2')

# 💡 เช็ค Path แก้วให้ตรงกับ Scene ของคุณ
cup_handle = sim.getObject('/20cmHighWallL[1]/Cup') 
ef_handle = sim.getObject('/yaskawa/gripperEF')

# =================================================================
# 🎯 ชุดตัวเลขมุมที่คุณคำนวณมาได้
# =================================================================
home_angles  = [0.00,   0.00,   0.00,   0.00,  -90.0, 0.00] # ท่าพัก

# ท่าหยิบแก้ว (เป๊ะตามที่คุณหามา)
pick_angles  = [14.90, -26.80, -59.48,   0.00,  0.00, 0.00] 

# ท่าวางคว่ำ (💡 ผมแก้ j4 ให้เป็น 180.00 ให้แล้วครับ แก้วจะได้พลิกคว่ำ!)
place_angles = [ 3.92,   5.43, -34.46, 180.00,  0.00, 0.00] 

# =================================================================
# 🚁 จุดแวะพัก (Hover) 
# =================================================================
# ก๊อปปี้ตัวเลขจากด้านบนมาเลย แต่ปรับ j2 (ตัวที่ 2) ให้แขนยกสูงขึ้น
hover_pick   = [14.90, -10.00, -59.48,   0.00,  0.00, 0.00] # ปรับ j2 จาก -26.80 มาเป็น -10.00
hover_place  = [ 3.92, -20.00, -34.46, 180.00,  0.00, 0.00] # ปรับ j2 จาก 5.43 มาเป็น -20.00 (เพื่อดึงแขนขึ้น)

# =================================================================
# ⚙️ ฟังก์ชันควบคุมหุ่นยนต์
# =================================================================
def move_arm(target_deg, steps=40, delay=0.01):
    """ฟังก์ชันขยับข้อต่อแบบสมูทๆ ค่อยๆ เลื่อนทีละเฟรม"""
    current_rad = [sim.getJointPosition(j) for j in joints]
    target_rad = [math.radians(deg) for deg in target_deg]
    
    for step in range(1, steps + 1):
        for i in range(6):
            new_pos = current_rad[i] + (target_rad[i] - current_rad[i]) * (step / steps)
            sim.setJointPosition(joints[i], new_pos)
        sim.step()
        time.sleep(delay)

def set_gripper(velocity):
    """ฟังก์ชันอ้า/หุบ กริปเปอร์"""
    sim.setJointTargetVelocity(mico_motor1, velocity)
    sim.setJointTargetVelocity(mico_motor2, velocity)
    for _ in range(20): sim.step()

# =================================================================
# 🎬 แอคชัน! (เริ่มถ่ายทำ)
# =================================================================
sim.setStepping(True)
sim.startSimulation()

try:
    print("1️⃣ เริ่มต้น: ยกแขนเตรียมพร้อม")
    move_arm(home_angles)
    set_gripper(0.15) # อ้ามือรอ
    time.sleep(0.5)
    
    print("2️⃣ เอื้อมไปหยิบแก้ว (Pick)")
    move_arm(hover_pick)  # โฉบไปรอเหนือแก้ว
    move_arm(pick_angles) # กดแขนลงไปตำแหน่ง X, Y, Z ของคุณ
    
    print("3️⃣ หนีบแก้ว!")
    set_gripper(-0.15)
    # 💡 ทริควิศวกร: แปะแก้วติดมือแบบล็อคตายตัว ป้องกันร่วงตอนตีลังกา
    sim.setObjectInt32Param(cup_handle, sim.shapeintparam_static, 1)
    sim.setObjectParent(cup_handle, ef_handle, True)
    
    print("4️⃣ ยกแก้วขึ้น")
    move_arm(hover_pick)
    
    print("5️⃣ สวิงข้ามไปฝั่งแท่นวาง พร้อมตีลังกาข้อมือ (FLIP!)")
    move_arm(hover_place, steps=60) # เพิ่ม step ให้สวิงช้าลงภาพจะได้สวยๆ
    move_arm(place_angles) # กดแขนลงวางคว่ำ
    
    print("6️⃣ ปล่อยแก้วลงบนแท่น (Place)")
    sim.setObjectParent(cup_handle, -1, True) # ปลดล็อคแก้วออกจากมือ
    sim.setObjectInt32Param(cup_handle, sim.shapeintparam_static, 0) # เปิดฟิสิกส์ให้แก้วอีกครั้ง
    set_gripper(0.15) # อ้ามือออก
    
    print("7️⃣ ดึงแขนกลับท่า Home")
    move_arm(hover_place)
    move_arm(home_angles)
    
    print("🎉 คัท! การแสดงจบสมบูรณ์แบบ ได้วิดีโอส่งอาจารย์แล้วครับ!")
    
finally:
    sim.stopSimulation()
    sim.setStepping(False)
    print("🔚 ปิดการเชื่อมต่อ")