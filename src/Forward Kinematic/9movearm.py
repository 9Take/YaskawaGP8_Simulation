import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class YaskawaRobot:
    def __init__(self):
        print("🔗 กำลังเชื่อมต่อระบบกับ CoppeliaSim...")
        self.api_client = RemoteAPIClient()
        self.sim = self.api_client.require('sim')
        
        # ตั้งค่าคงที่ของหุ่นยนต์
        self.TARGET_DROP_ZONE = [0.6109, 0.0623, 0.5392]
        self.TIME_STEP = 0.05
        self.IK_DAMPING = 0.15
        self.IK_GAIN = 1.2
        self.DELTA_EPSILON = 1e-4

        self._initialize_handles()

    def _initialize_handles(self):
        """ดึงข้อมูล Handles ทั้งหมดจาก Simulator"""
        self.base = self.sim.getObject('/yaskawa')
        self.end_effector = self.sim.getObject('/yaskawa/gripperEF')
        self.arm_joints = [self.sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
        
        # ระบบ Gripper
        self.finger_m1 = self.sim.getObject('/yaskawa/MicoHand/fingers12_motor1')
        self.finger_m2 = self.sim.getObject('/yaskawa/MicoHand/fingers12_motor2')
            
        # วัตถุเป้าหมาย
        self.target_cup = self.sim.getObject('/20cmHighWallL[1]/Cup')
    
    def show_telemetry(self, phase_name="Current Status"):
        """แสดงผลมุมของข้อต่อบนหน้าจอ"""
        rad_angles = [self.sim.getJointPosition(j) for j in self.arm_joints]
        deg_angles = [np.round(np.degrees(a), 2) for a in rad_angles]
        
        print(f"\n📌 [Phase]: {phase_name}")
        print(f"   [J1-J3]: {deg_angles[0]:+7.2f}° | {deg_angles[1]:+7.2f}° | {deg_angles[2]:+7.2f}°")
        print(f"   [J4-J6]: {deg_angles[3]:+7.2f}° | {deg_angles[4]:+7.2f}° | {deg_angles[5]:+7.2f}°")
        print("-" * 50)

    def actuate_gripper(self, speed):
        """สั่งการกริปเปอร์ (เปิด/ปิด)"""
        self.sim.setJointTargetVelocity(self.finger_m1, speed)
        self.sim.setJointTargetVelocity(self.finger_m2, speed)

    def attach_payload(self, is_attached):
        """
        ระบบความปลอดภัยเสริม: เชื่อมต่อวัตถุเข้ากับ End-effector ทางฟิสิกส์
        (ป้องกันวัตถุร่วงหล่นระหว่างการเคลื่อนที่แบบผาดโผน)
        """
        if is_attached:
            self.sim.setObjectInt32Param(self.target_cup, self.sim.shapeintparam_static, 1)
            self.sim.resetDynamicObject(self.target_cup)
            self.sim.setObjectParent(self.target_cup, self.end_effector, True)
            print("🔒 [Payload Status]: SECURED (ผูกแก้วติดกับมือแล้ว)")
        else:
            self.sim.setObjectParent(self.target_cup, -1, True)
            self.sim.setObjectInt32Param(self.target_cup, self.sim.shapeintparam_static, 0)
            self.sim.resetDynamicObject(self.target_cup)
            print("🔓 [Payload Status]: RELEASED (ปลดล็อกฟิสิกส์แก้ว)")

    def drive_tcp_to(self, goal_xyz, override_j5=None, override_j6=None, move_time=3.0):
        """
        เครื่องยนต์คำนวณ Inverse Kinematics หลัก
        """
        total_loops = int(move_time / self.TIME_STEP)
        
        # เก็บค่าเริ่มต้นของข้อมือสำหรับการทำ Interpolation
        init_j5 = self.sim.getJointPosition(self.arm_joints[4])
        init_j6 = self.sim.getJointPosition(self.arm_joints[5])

        for current_loop in range(total_loops):
            # 1. เช็คสถานะปัจจุบัน
            current_q = np.array([self.sim.getJointPosition(j) for j in self.arm_joints])
            tcp_pos = np.array(self.sim.getObjectPosition(self.end_effector, self.sim.handle_world))
            
            # 2. หาระยะกระจัด (Error Vector)
            dist_error = np.array(goal_xyz) - tcp_pos
            
            # 3. สร้าง Numerical Jacobian Matrix (3x6)
            jacob_mat = np.zeros((3, 6))
            for i in range(6):
                original_q = self.sim.getJointPosition(self.arm_joints[i])
                # แกล้งขยับเพื่อหาความชัน
                self.sim.setJointPosition(self.arm_joints[i], original_q + self.DELTA_EPSILON)
                perturbed_pos = np.array(self.sim.getObjectPosition(self.end_effector, self.sim.handle_world))
                
                jacob_mat[:, i] = (perturbed_pos - tcp_pos) / self.DELTA_EPSILON
                self.sim.setJointPosition(self.arm_joints[i], original_q) # ดึงกลับ

            # 4. อัลกอริทึม DLS (Damped Least Squares)
            identity_mat = np.identity(3) # ใช้ identity แทน eye
            dls_inverse = jacob_mat.T @ np.linalg.inv(jacob_mat @ jacob_mat.T + (self.IK_DAMPING**2) * identity_mat)
            joint_velocities = dls_inverse @ (dist_error * self.IK_GAIN)
            
            # 5. สั่งอัปเดตมอเตอร์
            # ชุดขับเคลื่อนหลัก (J1-J4)
            for idx in range(4):
                self.sim.setJointPosition(self.arm_joints[idx], current_q[idx] + joint_velocities[idx] * self.TIME_STEP)
            
            # ควบคุมข้อมือแยกอิสระ (Decoupled J5, J6)
            ratio = (current_loop + 1) / total_loops
            
            if override_j5 is not None:
                self.sim.setJointPosition(self.arm_joints[4], init_j5 + (override_j5 - init_j5) * ratio)
            else:
                self.sim.setJointPosition(self.arm_joints[4], current_q[4] + joint_velocities[4] * self.TIME_STEP)

            if override_j6 is not None:
                self.sim.setJointPosition(self.arm_joints[5], init_j6 + (override_j6 - init_j6) * ratio)
            else:
                self.sim.setJointPosition(self.arm_joints[5], current_q[5] + joint_velocities[5] * self.TIME_STEP)

            self.sim.step()

    def run_mission(self):
        """รันลำดับขั้นตอน Pick and Place"""
        try:
            self.sim.setStepping(True)
            self.sim.startSimulation()
            self.show_telemetry("System Ready / Home")

            # อ่านพิกัดเป้าหมาย (Position) และมุม (Orientation)
            item_pos = self.sim.getObjectPosition(self.target_cup, self.sim.handle_world)
            item_ori = self.sim.getObjectOrientation(self.target_cup, self.sim.handle_world)
            
            # แปลงมุมจากเรเดียน (Radian) เป็นองศา (Degree) เพื่อให้ดูง่ายขึ้น
            roll_deg, pitch_deg, yaw_deg = np.degrees(item_ori)
            
            # 💡 ปริ้นท์ข้อมูลทั้งหมดสำหรับนำไปใส่ Input Table 1
            print(f"\n🎯 [Target Acquired]: ข้อมูลแก้วน้ำที่ World Frame (สำหรับตาราง Input Table 1)")
            print(f"   📍 Position (m)   -> X: {item_pos[0]:+.4f} | Y: {item_pos[1]:+.4f} | Z: {item_pos[2]:+.4f}")
            print(f"   🔄 Orientation    -> Roll: {item_ori[0]:+.4f} rad ({roll_deg:+.2f}°) | Pitch: {item_ori[1]:+.4f} rad ({pitch_deg:+.2f}°) | Yaw: {item_ori[2]:+.4f} rad ({yaw_deg:+.2f}°)")


            # --- Sequence 1: Approach ---
            print("\n>> 1. Approaching Target...")
            self.actuate_gripper(1.0)
            
            j5_current = self.sim.getJointPosition(self.arm_joints[4])
            self.drive_tcp_to([item_pos[0], item_pos[1], item_pos[2] + 0.10], override_j5=j5_current + 0.6, move_time=3.0)
            self.drive_tcp_to([item_pos[0], item_pos[1], item_pos[2] - 0.0785], move_time=1.0)
            self.show_telemetry("Reached Target")

            # --- Sequence 2: Grasp ---
            print("\n>> 2. Grasping...")
            self.actuate_gripper(-1.0)
            for _ in range(25): self.sim.step()
            self.attach_payload(True) 
        
            # --- Sequence 3: Lift ---
            print("\n>> 3. Lifting Payload...")
            safe_z_height = [item_pos[0], item_pos[1], item_pos[2] + 0.35]
            self.drive_tcp_to(safe_z_height, move_time=2.0)
            self.show_telemetry("Payload Lifted")

            # --- Sequence 4: Flip (J6) ---
            print("\n>> 4. Executing Roll Maneuver (J6)...")
            j6_current = self.sim.getJointPosition(self.arm_joints[5])
            j6_target = j6_current + np.pi
            self.drive_tcp_to(safe_z_height, override_j6=j6_target, move_time=1.5)

            # --- Sequence 5: Tilt (J5) ---
            print("\n>> 5. Executing Pitch Maneuver (J5)...")
            j5_current = self.sim.getJointPosition(self.arm_joints[4])
            j5_target = j5_current - 0.4
            self.drive_tcp_to(safe_z_height, override_j6=j6_target, override_j5=j5_target, move_time=1.5)

            # --- Sequence 6: Deliver ---
            print(f"\n>> 6. Moving to Drop Zone: {self.TARGET_DROP_ZONE}")
            self.drive_tcp_to(self.TARGET_DROP_ZONE, override_j6=j6_target, override_j5=j5_target, move_time=3.0)
            self.show_telemetry("Arrived at Drop Zone")

            # --- Sequence 7: Release ---
            print("\n>> 7. Releasing Payload...") 
            self.attach_payload(False) 
            self.actuate_gripper(1.0)
            for _ in range(40): self.sim.step()
            self.show_telemetry("Mission Accomplished")

        finally:
            self.sim.stopSimulation()
            self.sim.setStepping(False)
            print("\n🛑 System Offline.")

# =========================================================
# 🚀 บังคับรันโปรแกรม
# =========================================================
if __name__ == "__main__":
    controller = YaskawaRobot()
    controller.run_mission()