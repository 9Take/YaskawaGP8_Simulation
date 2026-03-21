"""
gp8_kinematics.py
=================
Library คำนวณ Kinematics ของ Yaskawa GP8
ประกอบด้วย:
  - DH Parameters ของ GP8 จริง (ค่า L เป็น mm แปลงเป็น m แล้ว)
  - Forward Kinematics (FK)  : joint angles → end-effector pose
  - Geometric Jacobian (J)   : joint velocities → EE velocity
  - Inverse Kinematics (IK)  : target position → joint angles (DLS)
  - Manipulability Index      : วัดว่าห่างจาก singularity แค่ไหน

วิธีใช้:
    from gp8_kinematics import GP8Kinematics
    kin = GP8Kinematics()
    T   = kin.fk(q)                   # FK: matrix 4x4
    J   = kin.jacobian(q)             # Jacobian: matrix 6x6
    q, ok, err = kin.ik(target_pos)   # IK
    m   = kin.manipulability(q)       # Manipulability index
"""

import numpy as np

# ============================================================
# ความยาว link ของ GP8 (หน่วย: เมตร)
# ============================================================
L1 = 0.330      # 330 mm  (d1: ความสูงฐาน)
L2 = 0.345      # 345 mm  (a2: แขนท่อน 2)
L3 = 0.040      # 40  mm  (a3: offset)
L4 = 0.040      # 40  mm  (ไม่ใช้ใน DH แต่เก็บไว้อ้างอิง)
L5 = 0.340      # 340 mm  (d4: ความยาวแขนท่อนกลาง)
L6 = 0.080      # 80  mm  (d6: tool flange)
L7 = 0.16133    # 161.33 mm (ไม่ใช้ใน DH แต่เก็บไว้อ้างอิง)

d1 = 0.330      # 330 mm  (d1: ความสูงฐาน)
d5 = 0.34       # 340 mm  (d5: ความยาวแขนท่อนกลาง)
d6 = 0.255      # 255 mm  (d6: tool flange)

# ============================================================
# Classical DH Parameter Table ของ GP8
# ============================================================
# แต่ละแถว: [a (m), alpha (rad), d (m), theta_offset (rad)]
#
#  Joint | a    | alpha   | d   | theta_offset  | ชื่อ
#  ------+------+---------+-----+---------------+------
#    1   | L4   | +pi/2   | d1  | theta_1       | S-axis
#    2   | L2   |  0      | 0   | theta_2       | L-axis
#    3   | L3   | +pi/2   | 0   | theta_3       | U-axis
#    4   | 0    | -pi/2   | d5  | theta_4       | R-axis
#    5   | 0    | +pi/2   | 0   | theta_5       | B-axis
#    6   | 0    |  0      | d6  | theta_6       | T-axis
#
# Joint limits (องศา):
#   S: ±180,  L: -90/+135,  U: -175/+255
#   R: ±200,  B: ±135,      T: ±360

def std_dh(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])


_Q_MIN = np.radians([-180,  -90, -175, -200, -135, -360])
_Q_MAX = np.radians([ 180,  135,  255,  200,  135,  360])


class GP8Kinematics:
    """
    Kinematics solver สำหรับ Yaskawa GP8 (6-DOF revolute joints)

    ใช้งาน:
        kin = GP8Kinematics()    ← ไม่ต้องส่ง argument
    """

    def __init__(self):
        self.dh       = _DH_TABLE.copy()   # shape (6, 4)
        self.q_min    = _Q_MIN.copy()
        self.q_max    = _Q_MAX.copy()
        self.n_joints = 6

    # ----------------------------------------------------------------
    # DH Transform matrix ของ 1 link
    # ----------------------------------------------------------------
    @staticmethod
    def _dh_matrix(a, alpha, d, theta):
        """
        Homogeneous Transform จาก DH convention (Craig):
        T = Rz(theta) · Tz(d) · Tx(a) · Rx(alpha)

        [ cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)   a·cos(θ) ]
        [ sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)   a·sin(θ) ]
        [   0        sin(α)          cos(α)           d     ]
        [   0          0               0              1     ]
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct,  -st*ca,   st*sa,  a*ct],
            [st,   ct*ca,  -ct*sa,  a*st],
            [0,    sa,      ca,     d   ],
            [0,    0,       0,      1   ],
        ])

    # ----------------------------------------------------------------
    # Forward Kinematics
    # ----------------------------------------------------------------
    def fk(self, q, up_to_joint=None):
        """
        FK: joint angles → Homogeneous Transform 4×4

        Args:
            q           : array-like (6,)  มุม joint (เรเดียน)
            up_to_joint : int หรือ None    คำนวณถึง joint นี้

        Returns:
            T : np.ndarray (4,4)
                T[:3,:3] = rotation matrix
                T[:3, 3] = position [x, y, z] เมตร
        """
        n = self.n_joints if up_to_joint is None else up_to_joint
        T = np.eye(4)
        for i in range(n):
            a, alpha, d, off = self.dh[i]
            T = T @ self._dh_matrix(a, alpha, d, q[i] + off)
        return T

    def fk_position(self, q):
        """FK คืนแค่ position [x, y, z] (เมตร)"""
        return self.fk(q)[:3, 3]

    def fk_all_frames(self, q):
        """
        คืน list ของ T ทุก frame:
        [T_base, T_1, T_2, ..., T_EE]   (7 matrices)
        ใช้วาด stick diagram
        """
        frames, T = [np.eye(4)], np.eye(4)
        for i in range(self.n_joints):
            a, alpha, d, off = self.dh[i]
            T = T @ self._dh_matrix(a, alpha, d, q[i] + off)
            frames.append(T.copy())
        return frames

    # ----------------------------------------------------------------
    # Geometric Jacobian
    # ----------------------------------------------------------------
    def jacobian(self, q):
        """
        Geometric Jacobian (6×6):
            Jv[i] = z_{i-1} × (p_e - p_{i-1})   ← linear part
            Jw[i] = z_{i-1}                       ← angular part

        Returns:
            J : np.ndarray (6, 6)
        """
        p_ee = self.fk(q)[:3, 3]
        J    = np.zeros((6, self.n_joints))
        T_i  = np.eye(4)

        for i in range(self.n_joints):
            z_i  = T_i[:3, 2]
            p_i  = T_i[:3, 3]
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i
            a, alpha, d, off = self.dh[i]
            T_i = T_i @ self._dh_matrix(a, alpha, d, q[i] + off)

        return J

    def jacobian_position(self, q):
        """Jacobian เฉพาะ linear velocity (3×6)"""
        return self.jacobian(q)[:3, :]

    # ----------------------------------------------------------------
    # Manipulability
    # ----------------------------------------------------------------
    def manipulability(self, q):
        """
        Manipulability index = sqrt(det(Jv · Jv^T))
        ยิ่งใกล้ 0 = ยิ่งใกล้ singularity

        Returns:
            float ≥ 0
        """
        Jv = self.jacobian_position(q)
        return float(np.sqrt(max(np.linalg.det(Jv @ Jv.T), 0.0)))

    def is_near_singularity(self, q, threshold=0.001):
        """True ถ้า manipulability < threshold"""
        return self.manipulability(q) < threshold

    # ----------------------------------------------------------------
    # Inverse Kinematics — Damped Least Squares
    # ----------------------------------------------------------------
    def ik(self, target_pos, q_init=None,
           max_iter=200, tol=1e-4,
           alpha=0.5, lambda_damp=0.01):
        """
        IK ด้วย Damped Least Squares (DLS):
            Δq = α · Jv^T (Jv Jv^T + λ²I)^{-1} Δp

        Args:
            target_pos  : array(3,) ตำแหน่งเป้าหมาย [x,y,z] เมตร
            q_init      : array(6,) มุม joint เริ่มต้น (None = zeros)
            max_iter    : int   iteration สูงสุด
            tol         : float ความคลาดเคลื่อนที่รับได้ (เมตร)
            alpha       : float step size (0 < α ≤ 1)
            lambda_damp : float damping factor

        Returns:
            q       : array(6,) joint angles (เรเดียน)
            success : bool
            error   : float  ระยะที่เหลือ (เมตร)
        """
        target = np.asarray(target_pos, dtype=float)
        q = np.zeros(self.n_joints) if q_init is None \
            else np.array(q_init, dtype=float).copy()
        I3 = np.eye(3)

        for _ in range(max_iter):
            dp    = target - self.fk_position(q)
            error = np.linalg.norm(dp)
            if error < tol:
                return q, True, float(error)
            Jv = self.jacobian_position(q)
            A  = Jv @ Jv.T + (lambda_damp**2) * I3
            q  = np.clip(q + alpha * Jv.T @ np.linalg.solve(A, dp),
                         self.q_min, self.q_max)

        error = float(np.linalg.norm(target - self.fk_position(q)))
        return q, False, error

    # ----------------------------------------------------------------
    # Velocity IK — real-time tracking
    # ----------------------------------------------------------------
    def velocity_ik(self, q, target_vel, lambda_damp=0.05):
        """
        แปลง EE velocity → joint velocity (DLS):
            q_dot = Jv^T (Jv Jv^T + λ²I)^{-1} · v_des

        Args:
            q          : array(6,) มุม joint ปัจจุบัน (เรเดียน)
            target_vel : array(3,) [vx, vy, vz] (m/s)
            lambda_damp: float    damping

        Returns:
            q_dot : array(6,) (rad/s)
        """
        Jv = self.jacobian_position(q)
        A  = Jv @ Jv.T + (lambda_damp**2) * np.eye(3)
        return Jv.T @ np.linalg.solve(A, np.asarray(target_vel, dtype=float))

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------
    def clip_joints(self, q):
        """บังคับ joint limits"""
        return np.clip(q, self.q_min, self.q_max)

    def random_q(self):
        """สุ่ม joint angles ภายใน limits"""
        return np.random.uniform(self.q_min, self.q_max)

    def print_info(self, q):
        """พิมพ์ FK + Manipulability + Condition number"""
        T  = self.fk(q)
        m  = self.manipulability(q)
        cn = np.linalg.cond(self.jacobian_position(q))
        ns = self.is_near_singularity(q)
        print("=== GP8 Kinematics Info ===")
        print(f"Joint angles (°) : {np.degrees(q).round(2)}")
        print(f"EE Position  (m) : {T[:3,3].round(4)}")
        print(f"EE Rotation      :\n{T[:3,:3].round(4)}")
        print(f"Manipulability   : {m:.6f}  {'[NEAR SINGULARITY!]' if ns else '[OK]'}")
        print(f"Condition number : {cn:.2f}")
        print("===========================")


# ============================================================
# Self-test — รันไฟล์นี้โดยตรง: python gp8_kinematics.py
# ============================================================
if __name__ == '__main__':
    kin = GP8Kinematics()   # ← ไม่ต้องส่ง argument

    print("--- FK (q = all zeros) ---")
    q0 = np.zeros(6)
    kin.print_info(q0)

    print("\n--- IK test ---")
    target = np.array([0.5, 0.2, 0.8])
    print(f"Target : {target}")
    q_ik, ok, err = kin.ik(target)
    print(f"Result : {'สำเร็จ' if ok else 'ไม่ converge'}  error = {err*1000:.3f} mm")
    print(f"q_ik   : {np.degrees(q_ik).round(2)} °")
    print(f"FK check: {kin.fk_position(q_ik).round(4)}")

    print("\n--- Jacobian ---")
    J = kin.jacobian(q0)
    print(f"Shape  : {J.shape}")
    print(f"Jv:\n{J[:3].round(4)}")
    print(f"Jw:\n{J[3:].round(4)}")

    print("\n--- Velocity IK ---")
    v_des = np.array([0.1, 0.0, 0.0])
    q_dot = kin.velocity_ik(q0, v_des)
    v_chk = kin.jacobian_position(q0) @ q_dot
    print(f"v_des   = {v_des}")
    print(f"q_dot   = {q_dot.round(4)} rad/s")
    print(f"v_check = {v_chk.round(4)}  (ควรใกล้ v_des)")