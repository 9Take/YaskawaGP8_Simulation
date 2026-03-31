"""
Yaskawa GP8 Pick & Place Controller
Core Principles:
1. Real-time Numerical Jacobian Inverse Kinematics
2. Damped Least Squares (DLS) for singularity avoidance
3. Decoupled Joint Control (Independent J5/J6 override)
4. Dynamic Parent-Child Attachment for stable grasping
"""

import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# =========================================================
# ⚙️ 1. INITIALIZATION & SETUP
# =========================================================
print("🔌 Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

# กำหนดพิกัดเป้าหมาย (Delivery Point)
DELIVERY_XYZ = [0.6109, 0.0623, 0.5392]

try:
    # ดึง Handles ของชิ้นส่วนหลัก
    robot = sim.getObject('/yaskawa')
    tip = sim.getObject('/yaskawa/gripperEF')
    joint_h = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
    
    # ดึง Handles ของ Gripper (รองรับชื่อต่างกันใน Scene)
    try:
        j0 = sim.getObject('/yaskawa/fingers12_motor1')
        j1 = sim.getObject('/yaskawa/fingers12_motor2')
    except:
        j0 = sim.getObject('/yaskawa/MicoHand/fingers12_motor1')
        j1 = sim.getObject('/yaskawa/MicoHand/fingers12_motor2')
        
    # ดึง Handle ของแก้วน้ำ (รองรับกรณีมี Index)
    try:
        cup_h = sim.getObject('/20cmHighWallL[1]/Cup')
    except:
        cup_h = sim.getObject('/20cmHighWallL/Cup')
        
except Exception as e:
    print(f"❌ Object check failed: {e}. Please check your scene hierarchy.")
    exit()

# =========================================================
# 🛠️ 2. HELPER FUNCTIONS
# =========================================================
def print_joint_angles(label="State"):
    """อ่านค่ามุมข้อต่อทั้งหมด แปลงเป็นองศา และแสดงผล"""
    q_rad = [sim.getJointPosition(j) for j in joint_h]
    q_deg = [round(np.rad2deg(angle), 2) for angle in q_rad]
    
    print(f"\n📊 [JOINT ANGLES] : {label}")
    print(f"   J1: {q_deg[0]:+7.2f}° | J2: {q_deg[1]:+7.2f}° | J3: {q_deg[2]:+7.2f}°")
    print(f"   J4: {q_deg[3]:+7.2f}° | J5: {q_deg[4]:+7.2f}° | J6: {q_deg[5]:+7.2f}°")
    print("-" * 55)

def set_gripper_velocity(velocity):
    """ควบคุมความเร็วการเปิด/ปิด Gripper (บวก = เปิด, ลบ = ปิด)"""
    sim.setJointTargetVelocity(j0, velocity)
    sim.setJointTargetVelocity(j1, velocity)

def attach_cup(is_attached):
    """
    หลักการ: Parenting Bypass
    ปลดฟิสิกส์ชั่วคราวและแปะแก้วเข้ากับมือ เพื่อกันแก้วหลุดหล่นตอนตีลังกา
    """
    if is_attached:
        sim.setObjectInt32Param(cup_h, sim.shapeintparam_static, 1)
        sim.resetDynamicObject(cup_h)
        sim.setObjectParent(cup_h, tip, True)
        print("🔒 Status: CUP ATTACHED")
    else:
        sim.setObjectParent(cup_h, -1, True)
        sim.setObjectInt32Param(cup_h, sim.shapeintparam_static, 0)
        sim.resetDynamicObject(cup_h)
        print("🔓 Status: CUP RELEASED")

def move_robot(target_xyz, target_j5=None, target_j6=None, duration=3.0):
    """
    หลักการ: Numerical Jacobian + Damped Least Squares
    สามารถแยกบังคับข้อมือ (J5, J6) ได้อิสระ ในขณะที่ J1-J4 ยังคงทำหน้าที่เข้าหาเป้าหมาย
    """
    dt = 0.05
    steps = int(duration / dt)
    damping = 0.15
    gain = 1.2

    # จดจำมุมเริ่มต้นของ J5, J6 สำหรับทำ Linear Interpolation
    start_j5 = sim.getJointPosition(joint_h[4])
    start_j6 = sim.getJointPosition(joint_h[5])

    for step in range(steps):
        # 1. อ่านค่าพิกัดปัจจุบัน (ดึงเทียบ World Frame ตามต้นฉบับ)
        q_curr = np.array([sim.getJointPosition(j) for j in joint_h])
        curr_pos = np.array(sim.getObjectPosition(tip, sim.handle_world))
        
        # 2. หา Error ระยะทาง
        error = np.array(target_xyz) - curr_pos
        
        # 3. คำนวณ Numerical Jacobian Matrix (3x6) แบบสดๆ
        J = np.zeros((3, 6))
        epsilon = 1e-4
        for i in range(6):
            orig = sim.getJointPosition(joint_h[i])
            sim.setJointPosition(joint_h[i], orig + epsilon)
            new_p = np.array(sim.getObjectPosition(tip, sim.handle_world))
            J[:, i] = (new_p - curr_pos) / epsilon
            sim.setJointPosition(joint_h[i], orig)

        # 4. แก้สมการ IK ด้วย DLS (Damped Least Squares)
        inv_j = J.T @ np.linalg.inv(J @ J.T + (damping**2) * np.eye(3))
        dq = inv_j @ (error * gain)
        
        # 5. อัปเดตมุมมอเตอร์
        # 5.1 ให้ฐานถึงศอก (J1-J4) วิ่งตามสมการ IK
        for i in range(4):
            sim.setJointPosition(joint_h[i], q_curr[i] + dq[i] * dt)
        
        # 5.2 ข้อพับข้อมือ (J5) - ถ้าสั่ง Override ให้วิ่งตามสัดส่วนเวลา ถ้าไม่สั่งก็ใช้ IK
        if target_j5 is not None:
            progress = (step + 1) / steps
            j5_step = start_j5 + (target_j5 - start_j5) * progress
            sim.setJointPosition(joint_h[4], j5_step)
        else:
            sim.setJointPosition(joint_h[4], q_curr[4] + dq[4] * dt)

        # 5.3 ข้อหมุนข้อมือ (J6) - ควบคุมแยกอิสระเช่นเดียวกับ J5
        if target_j6 is not None:
            progress = (step + 1) / steps
            j6_step = start_j6 + (target_j6 - start_j6) * progress
            sim.setJointPosition(joint_h[5], j6_step)
        else:
            sim.setJointPosition(joint_h[5], q_curr[5] + dq[5] * dt)

        sim.step()

# =========================================================
# 🚀 3. MAIN EXECUTION LOOP
# =========================================================
def main():
    try:
        print("\n🎬 Initializing Simulation...")
        sim.setStepping(True)
        sim.startSimulation()
        print_joint_angles("Start / Home Position")

        # สแกนหาตำแหน่งแก้วปัจจุบัน
        cup_pos = sim.getObjectPosition(cup_h, sim.handle_world)
        

        # ---------------------------------------------------------
        print("\n▶️ STEP 1: APPROACH (เคลื่อนที่ไปรอเหนือแก้ว)")
        set_gripper_velocity(1.0) # อ้ามือ
        curr_j5 = sim.getJointPosition(joint_h[4]) # ไว้จูนระดับการเงยของข้อมือตอนวาง
        move_robot([cup_pos[0], cup_pos[1], cup_pos[2] + 0.10] , target_j5 = curr_j5 + 0.6 , duration=3.0)
        move_robot([cup_pos[0], cup_pos[1], cup_pos[2] - 0.0785] , duration=1.0)
        print_joint_angles("At Cup Position")

        # ---------------------------------------------------------
        print("\n▶️ STEP 2: GRASP (หนีบแก้ว)")
        set_gripper_velocity(-1.0)
        for _ in range(25): sim.step() 
        attach_cup(True)

        # ---------------------------------------------------------
        print("\n▶️ STEP 3: LIFT (ดึงแก้วขึ้นในแนวดิ่ง)")
        lift_height = [cup_pos[0], cup_pos[1], cup_pos[2] + 0.35]
        move_robot(lift_height, duration=2.0)
        print_joint_angles("After Lift")

        # ---------------------------------------------------------
        print("\n▶️ STEP 4: FLIP (บิดข้อมือ J6 ตีลังกา 180 องศา)")
        curr_j6 = sim.getJointPosition(joint_h[5])
        final_j6 = curr_j6 + np.pi  # บิดเพิ่ม 180 องศา (Pi Radians)
        move_robot(lift_height, target_j6=final_j6, duration=1.5)
        print_joint_angles("After Flip (J6)")

        # ---------------------------------------------------------
        print("\n▶️ STEP 5: TILT UP (เงยข้อมือ J5 ขึ้นเพื่อเตรียมวาง)")
        curr_j5 = sim.getJointPosition(joint_h[4])
        final_j5 = curr_j5 - 0.4
        move_robot(lift_height, target_j6=final_j6, target_j5=final_j5, duration=1.5)
        print_joint_angles("After Tilt (J5)")

        # ---------------------------------------------------------
        print(f"\n▶️ STEP 6: MOVE TO DELIVERY (นำทางไปยังพิกัดวาง {DELIVERY_XYZ})")
        # ลากแขนไปจุดวาง พร้อมรักษาระดับการตีลังกาของข้อมือ J5, J6 เอาไว้
        move_robot(DELIVERY_XYZ, target_j6=final_j6, target_j5=final_j5 , duration=3.0)
        print_joint_angles("At Delivery Point")

        # ---------------------------------------------------------
        print("\n▶️ STEP 7: RELEASE (ปล่อยแก้ว)")
        attach_cup(False)
        set_gripper_velocity(1.0)
        for _ in range(40): sim.step()
        print_joint_angles("Final Release State")

    finally:
        sim.stopSimulation()
        sim.setStepping(False)
        print("\n🏁 Simulation Process Complete.")

# สั่งรันโปรแกรม
if __name__ == "__main__":
    main()