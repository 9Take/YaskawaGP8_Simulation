from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

print("กำลังเชื่อมต่อกับ CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

j_handles = []
for i in range(1, 7):
    j_handles.append(sim.getObject(f'/yaskawa/joint{i}'))

# ==========================================
# 1. DH Parameters Yaskawa GP8 (mm)
# ==========================================
L1, L2, L3, L4 = 330, 345, 40, 40
L5, L6, L7 = 340, 80, 161.33

def std_dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

def get_jacobian_and_fk(q):
    pi2 = np.pi / 2.0
    T = [np.eye(4)] * 7
    T[1] = T[0] @ std_dh(q[0], L1,    L3,  pi2)
    T[2] = T[1] @ std_dh(q[1], 0,     L2,  0.0)
    T[3] = T[2] @ std_dh(q[2], 0,     L4,  pi2)
    T[4] = T[3] @ std_dh(q[3], L5,    0,  -pi2)
    T[5] = T[4] @ std_dh(q[4], 0,     0,   pi2)
    T[6] = T[5] @ std_dh(q[5], L6+L7, 0,   0.0)
    P_ee = T[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z_i = T[i][:3, 2]
        p_i = T[i][:3, 3]
        J[:3, i] = np.cross(z_i, P_ee - p_i)
        J[3:,  i] = z_i
    return J, P_ee, T[6]

# ==========================================
# 2. Home position
#    q1 = -36 deg หันหน้าไปทาง conveyor
#    (target angle = -36.4 deg)
#    หลีกเลี่ยง singularity: q5 = -75 deg
# ==========================================
q_home = np.array([
    np.radians(-36),   # j1: หันไปทาง conveyor วงกลมเลย
    np.radians(-30),   # j2
    np.radians(60),    # j3
    0.0,               # j4
    np.radians(-75),   # j5: หลีกเลี่ยง -90 singularity
    0.0                # j6
])
_, P_start, _ = get_jacobian_and_fk(q_home)
print(f"Home EF: x={P_start[0]:.1f}  y={P_start[1]:.1f}  z={P_start[2]:.1f} mm")
print(f"Home radius: {np.sqrt(P_start[0]**2+P_start[1]**2):.1f} mm")

# ==========================================
# 3. Waypoints + Timing
#
# จาก getCup_trajectory_loop.py:
#   แก้วเข้า workspace ที่ t=5.05s (step 101)
#   ตำแหน่ง pick: x=639.78, y=-515.41, z=453.25 mm
#   velocity: Vx=-189.9, Vy=52.7 mm/s
#   radius pick = 821.6 mm
#
# hover position:
#   back-calculate จาก pick → (734.7, -541.8) mm
#   แต่ radius=912mm เกิน workspace 888mm
#   → clamp ให้ radius=780mm (ปลอดภัย)
#   → (627.8, -462.9, 703.25) mm
#
# Timeline:
#   0.0  → 3.5s  : home → hover
#   3.5  → 4.55s : รอที่ hover (แก้ววิ่งเข้ามา)
#   4.55 → 5.05s : ลง + track xy แก้ว
#   5.05 → 5.55s : grab (เดินตามแก้ว)
#   5.55 → 7.0s  : lift ขึ้น
#   7.0  → 20.0s : ค้าง
# ==========================================
dt   = 0.05
step = 400

# pick position (ณ T_pick)
T_pick   = 5.05
px_pick  = 639.78
py_pick  = -515.41
pz_pick  = 453.25
pz_hover = pz_pick + 250.0   # 703.25 mm

cup_vx = -189.9
cup_vy =   52.7

# hover position (clamped radius=780mm)
# back-calc: แก้วอยู่ที่นี่ตอน T_wait_end=4.55s
dt_wait_to_pick = T_pick - 4.55   # = 0.5s
px_hover_raw = px_pick - cup_vx * dt_wait_to_pick   # 734.7
py_hover_raw = py_pick - cup_vy * dt_wait_to_pick   # -541.8
r_raw = np.sqrt(px_hover_raw**2 + py_hover_raw**2)  # 912mm

# clamp radius ให้ไม่เกิน 780mm
r_clamp  = 780.0
scale    = min(1.0, r_clamp / r_raw)
px_hover = px_hover_raw * scale   # 627.8
py_hover = py_hover_raw * scale   # -462.9

print(f"Hover: x={px_hover:.1f}  y={py_hover:.1f}  z={pz_hover:.1f}  "
      f"r={np.sqrt(px_hover**2+py_hover**2):.1f} mm")
print(f"Pick : x={px_pick:.1f}  y={py_pick:.1f}  z={pz_pick:.1f}  "
      f"r={np.sqrt(px_pick**2+py_pick**2):.1f} mm")

p0 = P_start.copy()

# Phase timing
T_arrive   = 3.5
T_wait_end = 4.55
T_grab_end = T_pick + 0.5   # 5.55s
T_lift_end = 7.0

def get_cmd(t_now):
    p_hover_3d = np.array([px_hover, py_hover, pz_hover])

    if t_now < T_arrive:
        # Phase 1: home → hover
        frac  = t_now / T_arrive
        p_tgt = p0 + frac * (p_hover_3d - p0)
        v_des = (p_hover_3d - p0) / T_arrive
        return v_des, p_tgt

    elif t_now < T_wait_end:
        # Phase 2: รอที่ hover
        return np.zeros(3), p_hover_3d

    elif t_now < T_pick:
        # Phase 3: ลง + track xy แก้ว
        dur   = T_pick - T_wait_end
        frac  = (t_now - T_wait_end) / dur
        dt_e  = t_now - T_wait_end
        p_tgt = np.array([
            px_hover + cup_vx * dt_e,
            py_hover + cup_vy * dt_e,
            pz_hover + frac * (pz_pick - pz_hover)
        ])
        v_des = np.array([cup_vx, cup_vy, (pz_pick - pz_hover) / dur])
        return v_des, p_tgt

    elif t_now < T_grab_end:
        # Phase 4: grab เดินตามแก้ว คง z
        dt_e  = t_now - T_wait_end
        p_tgt = np.array([
            px_hover + cup_vx * dt_e,
            py_hover + cup_vy * dt_e,
            pz_pick
        ])
        return np.array([cup_vx, cup_vy, 0.0]), p_tgt

    elif t_now < T_lift_end:
        # Phase 5: lift ขึ้น เดินตาม xy
        dur   = T_lift_end - T_grab_end
        frac  = (t_now - T_grab_end) / dur
        dt_e  = t_now - T_wait_end
        p_tgt = np.array([
            px_hover + cup_vx * dt_e,
            py_hover + cup_vy * dt_e,
            pz_pick + frac * (pz_hover - pz_pick)
        ])
        v_des = np.array([cup_vx, cup_vy, (pz_hover - pz_pick) / dur])
        return v_des, p_tgt

    else:
        # Phase 6: ค้างที่ lift position
        dt_e   = T_lift_end - T_wait_end
        p_hold = np.array([
            px_hover + cup_vx * dt_e,
            py_hover + cup_vy * dt_e,
            pz_hover
        ])
        return np.zeros(3), p_hold

# ==========================================
# 4. คำนวณ joint trajectory (DLS IK)
# ==========================================
q_curr = q_home.copy()
j_list = []
Kp  = 6.0
lam = 0.5

print("คำนวณ trajectory...")
for step_i in range(step):
    t_now = step_i * dt
    V_des, P_tgt = get_cmd(t_now)

    J, P_curr, _ = get_jacobian_and_fk(q_curr)
    err   = P_tgt - P_curr
    V_cmd = np.concatenate([V_des + Kp * err, [0.0, 0.0, 0.0]])

    JJT   = J @ J.T
    J_dls = J.T @ np.linalg.inv(JJT + (lam**2) * np.eye(6))

    k_ns    = 0.05
    q_dot_0 = -k_ns * (q_curr - q_home)
    q_dot   = J_dls @ V_cmd + (np.eye(6) - J_dls @ J) @ q_dot_0
    q_dot   = np.clip(q_dot, -2.0, 2.0)

    q_curr = q_curr + q_dot * dt
    j_list.append(q_curr.copy())

# ==========================================
# 5. รัน simulation
# ==========================================
print("กำลังเริ่มขยับหุ่นยนต์...")
sim.setStepping(True)
sim.startSimulation()

for q in j_list:
    sim.setJointPosition(j_handles[0], float(q[0]))
    sim.setJointPosition(j_handles[1], float(q[1]))
    sim.setJointPosition(j_handles[2], float(q[2]))
    sim.setJointPosition(j_handles[3], float(q[3]))
    sim.setJointPosition(j_handles[4], float(q[4]))
    sim.setJointPosition(j_handles[5], float(q[5]))
    sim.step()

sim.setStepping(False)
sim.stopSimulation()
print("เสร็จสิ้น!")