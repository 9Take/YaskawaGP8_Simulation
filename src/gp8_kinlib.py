"""
gp8_kinematics.py  (Geometric IK version)
==========================================
IK แบบ Geometric (closed-form) ไม่ต้อง DH table
คำนวณ q1-q3 จากสมการ trigonometry โดยตรง
ไม่มี FK error ปัญหา

ค่า link lengths จากการวัดจริงใน CoppeliaSim:
  BASE = [0.01867, 0.00023]  offset ของ joint1 จาก world origin
  a1   = 0.040   (j1→j2, แนว X)
  d1   = 0.330   (ความสูงฐาน, แนว Z)
  L2   = 0.345   (j2→j3, แนว Z)
  L3   = 0.340   (j3→j4, แนว X)
  d3   = 0.040   (j3→j4, แนว Z offset)
  L6   = 0.1336  (j6→EE, tool length)
"""

import numpy as np

# ============================================================
# Link lengths 
# ============================================================
BASE_X = 0.01867
BASE_Y = 0.00023
a1     = 0.040
d1     = 0.330
L2     = 0.345
L3     = 0.340
d3     = 0.040
L6     = 0.1336

# Joint limits จาก datasheet (หน่วย: เรเดียน)
# Joint | q_min (°) | q_max (°)
# ------+------------+-----------
#   1   | -170       | 170
#   2   | -65        | 145
#   3   | -70        | 190
#   4   | -190       | 190
#   5   | -135       | 135
#   6   | -360       | 360

Q_MIN = np.radians([-170,  -65, -70, -190, -135, -360])
Q_MAX = np.radians([ 170,  145,  190,  190,  135,  360])



class GP8Kinematics:
    """
    Geometric IK solver สำหรับ Yaskawa GP8

    ใช้งาน:
        kin = GP8Kinematics()
        q, ok, msg = kin.ik([x, y, z])
    """

    def __init__(self):
        self.q_min    = Q_MIN.copy()
        self.q_max    = Q_MAX.copy()
        self.n_joints = 6

    # ----------------------------------------------------------------
    # Inverse Kinematics — Geometric (closed-form)
    # ----------------------------------------------------------------
    def ik(self, target_pos, elbow_up=True, q_init=None):
        """
        คำนวณ joint angles จาก target position โดยใช้ geometry ล้วนๆ

        หลักการ:
          q1 = atan2(y, x)              ← หมุนฐานให้ชี้เป้า
          q3 = ±acos(law of cosines)    ← มุม elbow
          q2 = atan2(h,r) - atan2(...)  ← มุม shoulder

        Args:
            target_pos : array(3,) [x, y, z] เมตร (world frame)
            elbow_up   : bool  True = elbow up, False = elbow down
            q_init     : ไม่ใช้ (ใส่ไว้ให้ compatible กับ interface เดิม)

        Returns:
            q       : array(6,) joint angles (rad)  หรือ None ถ้าทำไม่ได้
            success : bool
            message : str  'ok' หรือ error message
        """
        px = float(target_pos[0]) - BASE_X
        py = float(target_pos[1]) - BASE_Y
        pz = float(target_pos[2])

        # --- q1: หมุนฐาน ---
        q1 = np.arctan2(py, px)

        # --- ระยะแนวราบถึง wrist (หัก tool length) ---
        r_total = np.sqrt(px**2 + py**2)
        r       = r_total - L6   # ระยะถึง wrist center

        # --- ความสูง wrist ---
        h = pz - d1 - d3

        # --- ระยะตรงจาก shoulder ถึง wrist ---
        D = np.sqrt(r**2 + h**2)

        # --- ตรวจ reachability ---
        if D > L2 + L3:
            return None, False, f"out of reach (D={D:.3f} > {L2+L3:.3f})"
        if D < abs(L2 - L3):
            return None, False, f"too close (D={D:.3f} < {abs(L2-L3):.3f})"

        # --- q3: law of cosines ---
        cos_q3 = (D**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3_abs = np.arccos(cos_q3)
        q3 = -q3_abs if elbow_up else q3_abs

        # --- q2: shoulder angle ---
        alpha = np.arctan2(h, r)
        beta  = np.arctan2(
            L3 * np.sin(abs(q3)),
            L2 + L3 * np.cos(abs(q3))
        )
        q2 = (alpha - beta) if elbow_up else (alpha + beta)

        # --- q4, q5, q6: wrist straight ---
        q4, q5, q6 = 0.0, 0.0, 0.0

        q = np.array([q1, q2, q3, q4, q5, q6])

        # ตรวจ joint limits
        if np.any(q < self.q_min) or np.any(q > self.q_max):
            # ลอง elbow อีกด้าน
            q3_alt = q3_abs if elbow_up else -q3_abs
            alpha_alt = np.arctan2(h, r)
            beta_alt  = np.arctan2(
                L3 * np.sin(abs(q3_alt)),
                L2 + L3 * np.cos(abs(q3_alt))
            )
            q2_alt = (alpha_alt + beta_alt) if elbow_up else (alpha_alt - beta_alt)
            q_alt  = np.array([q1, q2_alt, q3_alt, 0.0, 0.0, 0.0])
            if np.all(q_alt >= self.q_min) and np.all(q_alt <= self.q_max):
                q = q_alt
            else:
                return q, False, "joint limit exceeded"

        return q, True, "ok"

    # ----------------------------------------------------------------
    # Manipulability (ใช้ numerical Jacobian — ไม่ต้อง DH)
    # ----------------------------------------------------------------
    def manipulability(self, q, delta=1e-4):
        """
        คำนวณ manipulability index โดย perturb q แล้ว numerical diff
        ใกล้ 0 = ใกล้ singularity

        ใช้ CoppeliaSim อ่านตำแหน่ง EE จริง แทน FK จาก DH
        → แต่ในกรณีนี้ approximate ด้วย geometric FK อย่างง่าย
        """
        p0 = self._approx_fk(q)
        J  = np.zeros((3, 6))
        for i in range(6):
            q_p = q.copy(); q_p[i] += delta
            q_m = q.copy(); q_m[i] -= delta
            J[:, i] = (self._approx_fk(q_p) - self._approx_fk(q_m)) / (2*delta)
        return float(np.sqrt(max(np.linalg.det(J[:, :3] @ J[:, :3].T), 0.0)))

    def is_near_singularity(self, q, threshold=0.001):
        return self.manipulability(q) < threshold

    def _approx_fk(self, q):
        """FK แบบง่าย (เฉพาะ q1-q3) สำหรับ manipulability"""
        q1, q2, q3 = q[0], q[1], q[2]
        r = (L2 * np.cos(q2) + L3 * np.cos(q2 + q3) + L6)
        x = BASE_X + (a1 + r) * np.cos(q1)
        y = BASE_Y + (a1 + r) * np.sin(q1)
        z = d1 + d3 + L2 * np.sin(q2) + L3 * np.sin(q2 + q3)
        return np.array([x, y, z])

    # ----------------------------------------------------------------
    # Velocity IK — Jacobian numerical
    # ----------------------------------------------------------------
    def velocity_ik(self, q, target_vel, lambda_damp=0.05, delta=1e-4):
        """
        แปลง EE velocity → joint velocity
        ใช้ numerical Jacobian จาก _approx_fk

        Args:
            q          : array(6,) joint angles ปัจจุบัน (rad)
            target_vel : array(3,) [vx, vy, vz] (m/s)
            lambda_damp: float damping

        Returns:
            q_dot : array(6,) (rad/s)
        """
        p0 = self._approx_fk(q)
        Jv = np.zeros((3, 6))
        for i in range(6):
            q_p = q.copy(); q_p[i] += delta
            q_m = q.copy(); q_m[i] -= delta
            Jv[:, i] = (self._approx_fk(q_p) - self._approx_fk(q_m)) / (2*delta)
        v   = np.asarray(target_vel, dtype=float)
        A   = Jv @ Jv.T + (lambda_damp**2) * np.eye(3)
        return Jv.T @ np.linalg.solve(A, v)

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------
    def clip_joints(self, q):
        return np.clip(q, self.q_min, self.q_max)

    def print_info(self, q):
        p  = self._approx_fk(q)
        m  = self.manipulability(q)
        ns = self.is_near_singularity(q)
        print("=== GP8 Geometric IK Info ===")
        print(f"q (°)          : {np.degrees(q).round(2)}")
        print(f"EE pos (approx): {p.round(4)} m")
        print(f"Manipulability : {m:.5f}  {'[NEAR SINGULARITY!]' if ns else '[OK]'}")
        print("=============================")


# ============================================================
# Self-test
# ============================================================
if __name__ == '__main__':
    kin = GP8Kinematics()

    print("=== Geometric IK Self-test ===\n")

    tests = [
        ("CoppeliaSim q=zeros EE", [0.5323,  0.0002,  0.7150]),
        ("CoppeliaSim j1=90  EE",  [0.0180,  0.5139,  0.7150]),
        ("CoppeliaSim j2=45  EE",  [0.6662,  0.0006,  0.2677]),
        ("Intercept point",         [0.5830, -0.5010,  0.4350]),
    ]

    print(f"{'Target':<28} {'q1°':>7} {'q2°':>7} {'q3°':>7} "
          f"{'FK_x':>7} {'FK_y':>7} {'FK_z':>7} {'status':>8}")
    print("-"*80)

    for name, pos in tests:
        q, ok, msg = kin.ik(pos)
        if ok:
            fk  = kin._approx_fk(q)
            err = np.linalg.norm(fk - pos) * 1000
            print(f"{name:<28} {np.degrees(q[0]):>7.1f} {np.degrees(q[1]):>7.1f} "
                  f"{np.degrees(q[2]):>7.1f} "
                  f"{fk[0]:>7.4f} {fk[1]:>7.4f} {fk[2]:>7.4f} "
                  f"err={err:>5.1f}mm")
        else:
            print(f"{name:<28} {'—':>7} {'—':>7} {'—':>7} {msg:>8}")

    print("\n--- Velocity IK test ---")
    q0    = np.array([0, np.radians(30), np.radians(-60), 0, 0, 0])
    v_des = np.array([0.1, 0.05, 0.0])
    q_dot = kin.velocity_ik(q0, v_des)
    Jv    = np.zeros((3,6))
    d     = 1e-4
    for i in range(6):
        qp = q0.copy(); qp[i] += d
        qm = q0.copy(); qm[i] -= d
        Jv[:,i] = (kin._approx_fk(qp)-kin._approx_fk(qm))/(2*d)
    v_chk = Jv @ q_dot
    print(f"v_des   = {v_des}")
    print(f"q_dot   = {q_dot.round(4)} rad/s")
    print(f"v_check = {v_chk.round(4)}  (ควรใกล้ v_des)")