import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class YaskawaGP8Robot:
    """Yaskawa GP8 Robot with Inverse Kinematics control"""

    def __init__(self):
        """Initialize robot parameters and simulator connection"""
        client = RemoteAPIClient()
        self.sim = client.require('sim')
        self.sim.setStepping(True)

        # --- Robot DH Parameters ---
        self.d1, self.a1 = 0.33, 0.01867    
        self.a2, self.a3, self.a4 = 0.04, 0.345, 0.04
        self.d5 = -0.34

        # --- TCP Offset: Link6 → MicoHand center (108mm in X) ---
        self.T6_7 = np.array([
            [1, 0, 0, 0.108],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # --- Get motor objects ---
        self.motors = [self.sim.getObject(f'/yaskawa/joint{i+1}') for i in range(6)]

    def _dh_matrix(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*sa, ca, d*ca],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, q):
        T01 = self._dh_matrix(q[0],           self.d1, self.a1, 0)
        T12 = self._dh_matrix(q[1] - np.pi/2, 0,       self.a2, -np.pi/2)
        T23 = self._dh_matrix(q[2],           0,       self.a3, np.pi)
        T34 = self._dh_matrix(q[3] + np.pi,   self.d5, self.a4, -np.pi/2)
        T45 = self._dh_matrix(q[4],           0,       0,       -np.pi/2)
        T56 = self._dh_matrix(q[5],           0,       0,       np.pi/2)

        T02 = T01 @ T12
        T03 = T02 @ T23
        T04 = T03 @ T34
        T05 = T04 @ T45
        T06 = T05 @ T56
        T07 = T06 @ self.T6_7  # TCP frame

        return [np.eye(4), T01, T02, T03, T04, T05, T06], T07

    def jacobian(self, q):
        J = np.zeros((6, 6))
        eps = 1e-5
        T_list, T_curr = self.forward_kinematics(q)
        p_curr = T_curr[:3, 3]

        for i in range(6):
            q_eps = np.array(q, dtype=float)
            q_eps[i] += eps
            _, T_eps = self.forward_kinematics(q_eps)
            J[:3, i] = (T_eps[:3, 3] - p_curr) / eps
            J[3:, i] = T_list[i][:3, 2]
        return J

    def quintic_trajectory(self, x0, xf, duration, current_time):
        if duration == 0:
            return xf, 0.0
        tau = min(current_time / duration, 1.0)
        s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
        x = x0 + (xf - x0) * s
        ds_dtau = 30 * (tau**2) - 60 * (tau**3) + 30 * (tau**4)
        v = (xf - x0) * (ds_dtau / duration)
        return x, v

    def get_current_config(self):
        return np.array([self.sim.getJointPosition(motor) for motor in self.motors])

    def set_joint_positions(self, q):
        for j in range(6):
            self.sim.setJointPosition(self.motors[j], float(q[j]))

    def move_to(self, target_position, duration=5.0):
        """Move robot TCP to target position using Damped Least Squares (DLS) IK"""
        dt = self.sim.getSimulationTimeStep()
        q_current = self.get_current_config()
        
        _, T_start = self.forward_kinematics(q_current)
        p0 = T_start[:3, 3]
        pf = np.array(target_position)

        total_steps = int(duration / dt)
        print(f"Moving from {np.round(p0, 4)} to {pf.round(4)} in {duration}s")

        # --- ทริคป้องกันหุ่นสะบัด (อิงจากโค้ดอาจารย์) ---
        LAMBDA = 0.005     # Damping factor: หน่วงสมการไม่ให้พังเวลาเจอจุดบอด (Singularity)
        QD_MAX = 3.5       # Max joint velocity: ลิมิตความเร็วข้อต่อไม่ให้กระชากเกิน 3.5 rad/s
        Kp = 5.0           # Position feedback gain: สปริงคอยดึงให้ EF วิ่งตามเป้าหมาย

        for i in range(total_steps):
            t = i * dt
            
            # 1. วางแผนเส้นทาง (หาตำแหน่งและความเร็วที่ควรจะเป็น ณ วินาทีที่ t)
            p_des_x, vx = self.quintic_trajectory(p0[0], pf[0], duration, t)
            p_des_y, vy = self.quintic_trajectory(p0[1], pf[1], duration, t)
            p_des_z, vz = self.quintic_trajectory(p0[2], pf[2], duration, t)
            
            p_des = np.array([p_des_x, p_des_y, p_des_z])
            V_ff = np.array([vx, vy, vz])

            # 2. อ่านตำแหน่งปัจจุบันจริงๆ ของหุ่น
            _, T_curr = self.forward_kinematics(q_current)
            p_curr = T_curr[:3, 3]

            # 3. คำนวณ Error แล้วดึงกลับเข้าเป้าหมาย (สำคัญมาก ป้องกันการสะสม Error จนหุ่นเพี้ยน)
            V_pos = V_ff + Kp * (p_des - p_curr)
            
            # บังคับไม่ให้ข้อมือหมุนมั่ว
            V_ori = np.array([0.0, 0.0, 0.0]) 
            x_dot = np.concatenate([V_pos, V_ori])

            # 4. คำนวณ Inverse Kinematics ด้วย Damped Least Squares (DLS)
            J = self.jacobian(q_current)
            
            # JJT = J * J^T + (Lambda * I) ป้องกันค่าพุ่งเป็นอนันต์
            JJT = J @ J.T + LAMBDA * np.eye(6)
            q_dot = J.T @ np.linalg.solve(JJT, x_dot)

            # 5. ลิมิตความเร็วมอเตอร์ไม่ให้เกินขีดจำกัด (ตัดหัวกราฟความเร็วที่กระชาก)
            q_dot = np.clip(q_dot, -QD_MAX, QD_MAX)

            # 6. อัปเดตและสั่งงานเข้า CoppeliaSim
            q_current = q_current + q_dot * dt
            self.set_joint_positions(q_current)
            self.sim.step()
            
        print("  -> Phase completed.")


# ============================================================
# Main Execution (Part 2: Trajectory Tracking)
# ============================================================
if __name__ == "__main__":
    print("Initializing Robot Control...")
    robot = YaskawaGP8Robot()

    # 1. โหลดข้อมูลจากไฟล์ที่รันไว้ใน Part 1
    # filename = 'trajectory_cup_data.npz'
    # เพิ่ม ../../ เพื่อถอยโฟลเดอร์กลับไปหาไฟล์ที่อยู่ข้างนอกสุด
    filename = '../../trajectory_cup_data.npz'
    try:
        data = np.load(filename)
        print(f"[OK] Loaded data from {filename}")
    except FileNotFoundError:
        print(f"[ERROR] ไม่พบไฟล์ {filename} รันโค้ด Part 1 เพื่อสร้างไฟล์ก่อนครับ")
        exit()

    grab_step = int(data['grab_step'])
    cup_px = data['px']
    cup_py = data['py']
    cup_pz = data['pz']
    place_pos = data['place_pos']

    # 2. คำนวณจุดสำคัญ (Waypoints)
    # จุดที่จะหยิบแก้ว (อ้างอิงจาก Step ที่ดีที่สุด)
    target_grab = np.array([cup_px[grab_step], cup_py[grab_step], cup_pz[grab_step]])
    
    # จุดเตรียมหยิบ (เหนือแก้ว หรือ ถอยออกมาเล็กน้อยเพื่อไม่ให้ชน)
    # สมมติให้แขนเข้าจากด้านบน (Z + 0.15m)
    pre_grab = target_grab + np.array([0.0, 0.0, 0.15])
    
    # จุดยกขึ้นหลังจับแก้วเสร็จ
    lift_up = target_grab + np.array([0.0, 0.0, 0.25])
    
    # จุดเตรียมวาง (เหนือสายพานวาง)
    pre_place = place_pos + np.array([0.0, 0.0, 0.15])

    # 3. เริ่มควบคุมหุ่นยนต์
    robot.sim.startSimulation()
    # ปล่อยให้ Physics เซ็ตตัวสักนิด
    for _ in range(5):
        robot.sim.step()

    try:
        print("\n--- Phase 1: APPROACH (Move to Pre-Grab) ---")
        robot.move_to(pre_grab, duration=3.0)

        print("\n--- Phase 2: DESCEND (Move to Grab Target) ---")
        robot.move_to(target_grab, duration=2.0)
        
        # ---------------------------------------------------------
        # TODO: ตรงนี้คุณสามารถเพิ่มฟังก์ชันสั่งปิด Gripper (MicoHand)
        # ---------------------------------------------------------
        print("  [ACTION] *** CLOSING GRIPPER ***")
        for _ in range(10): # หน่วงเวลาให้ Gripper ทำงาน
            robot.sim.step()

        print("\n--- Phase 3: LIFT (Lift the Cup) ---")
        robot.move_to(lift_up, duration=2.0)

        print("\n--- Phase 4: TRANSPORT (Move to Place Position) ---")
        robot.move_to(pre_place, duration=4.0)

        print("\n--- Phase 5: PLACE (Descend to Conveyor) ---")
        robot.move_to(place_pos, duration=2.0)

        # ---------------------------------------------------------
        # TODO: ตรงนี้คุณสามารถเพิ่มฟังก์ชันสั่งเปิด Gripper
        # ---------------------------------------------------------
        print("  [ACTION] *** OPENING GRIPPER ***")
        for _ in range(10):
            robot.sim.step()

        print("\n--- Phase 6: RETURN (Move back to Safe Position) ---")
        robot.move_to(pre_place, duration=2.0)

        print("\n[SUCCESS] Trajectory completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] เกิดข้อผิดพลาดระหว่างรัน: {e}")
    finally:
        # สั่งหยุดซิมูเลชันเมื่อทำงานเสร็จหรือเกิด Error
        robot.sim.stopSimulation()