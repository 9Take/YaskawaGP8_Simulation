"""
010243427: INDUSTRIAL MOBILE ROBOTS — Class Project 2/68
Yaskawa GP8 + Kinova KG-2 Gripper
Scene: yaskawaGP8_loopConveyor.ttt

Pipeline ตาม tutorial:
  Step 1: ได้ cup trajectory จาก getCup_trajectory_loop.py
  Step 2: Design EF trajectory (waypoints + timing)
  Step 3: Jacobian → joints' trajectory  ← ไฟล์นี้
  Step 4: Run simulation
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

# ══════════════════════════════════════════════════════════════
# 1. DH Parameters — Yaskawa GP8 (mm)
#    verified: FK(q=[0,0,0,0,0,0]) → x≈541, z=715
#    joint offset: q2_dh = q2_real + 90°
# ══════════════════════════════════════════════════════════════
L1, L2, L3, L4 = 330, 345, 40, 40
L5, L_TOOL     = 340, 161.33
J_OFFSET = np.array([0, np.pi/2, 0, 0, 0, 0])

def std_dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

def fk(q_real):
    """FK — q_real คือ joint angles จาก CoppeliaSim (rad)"""
    q = q_real + J_OFFSET
    T = np.eye(4)
    for (th,d,a,al) in [(q[0],L1,L3,np.pi/2),(q[1],0,L2,0),
                        (q[2],0,L4,np.pi/2),(q[3],L5,0,-np.pi/2),
                        (q[4],0,0,np.pi/2),(q[5],L_TOOL,0,0)]:
        T = T @ std_dh(th,d,a,al)
    return T[:3,3], T[:3,:3]

def jacobian(q_real):
    """Geometric Jacobian 6x6"""
    q  = q_real + J_OFFSET
    Ts = [np.eye(4)]*7
    params = [(q[0],L1,L3,np.pi/2),(q[1],0,L2,0),
              (q[2],0,L4,np.pi/2),(q[3],L5,0,-np.pi/2),
              (q[4],0,0,np.pi/2),(q[5],L_TOOL,0,0)]
    for i,p in enumerate(params):
        Ts[i+1] = Ts[i] @ std_dh(*p)
    P_ee = Ts[6][:3,3]
    J = np.zeros((6,6))
    for i in range(6):
        zi = Ts[i][:3,2]; pi = Ts[i][:3,3]
        J[:3,i] = np.cross(zi, P_ee-pi)
        J[3:,i] = zi
    return J

def rot_err(R_cur, R_des):
    Re = R_des @ R_cur.T
    return 0.5*np.array([Re[2,1]-Re[1,2],Re[0,2]-Re[2,0],Re[1,0]-Re[0,1]])

def euler_to_rot(roll, pitch, yaw):
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    return Rz@Ry@Rx

# ══════════════════════════════════════════════════════════════
# 2. Numerical IK Solver (Jacobian DLS)
#    — คำนวณ joint angles จาก target position+orientation
# ══════════════════════════════════════════════════════════════
def solve_ik(p_target, R_target, q_init, max_iter=500, tol=2.0):
    q = q_init.copy()
    for it in range(max_iter):
        p_c, R_c = fk(q)
        dp = p_target - p_c
        if np.linalg.norm(dp) < tol: break
        dr  = rot_err(R_c, R_target)
        J   = jacobian(q)
        J_dls = J.T @ np.linalg.inv(J@J.T + 0.5**2*np.eye(6))
        q   = q + J_dls @ np.concatenate([dp, dr]) * 0.4
    p_r,_ = fk(q)
    err   = np.linalg.norm(p_target - p_r)
    print(f"    IK: err={err:.1f}mm  iter={it+1}  "
          f"q=[{', '.join(f'{np.degrees(x):.1f}' for x in q)}]°")
    return q

# ══════════════════════════════════════════════════════════════
# 3. STEP 2 DATA — Waypoints จาก trajectory design
# ══════════════════════════════════════════════════════════════
# --- Step 1 output (cup data) ---
T_PICK   = 5.05              # วินาทีที่แก้วเข้า workspace
px_pick, py_pick, pz_pick = 639.78, -515.41, 453.25  # mm
cup_vx,  cup_vy            = -189.9, 52.7             # mm/s

# --- Derived waypoints ---
HOVER_Z  = pz_pick + 250.0  # 703.25 mm

# hover: back-calc + clamp r=780mm
T_WAIT   = 4.55
dt_w2p   = T_PICK - T_WAIT
px_hraw  = px_pick - cup_vx*dt_w2p
py_hraw  = py_pick - cup_vy*dt_w2p
r_raw    = np.sqrt(px_hraw**2 + py_hraw**2)
sc       = min(1.0, 780.0/r_raw)
px_hov, py_hov = px_hraw*sc, py_hraw*sc

# place: linear conveyor
px_pl, py_pl, pz_pl = 500.0, 500.0, 500.0
PLACE_HOVER_Z = pz_pl + 250.0

# Orientations
angle_to_cup = np.arctan2(py_pick, px_pick)   # -38.9°
R_pick  = euler_to_rot(np.radians(-135.1), 0.0,           angle_to_cup)
R_place = euler_to_rot(0.0,               np.radians(90), np.radians(-180))

# ══════════════════════════════════════════════════════════════
# 4. STEP 3 — คำนวณ Joint Angles ด้วย IK (Jacobian)
# ══════════════════════════════════════════════════════════════
q_home = np.zeros(6)   # CoppeliaSim home = all joints zero

# q_init ที่เหมาะสม (ใกล้ solution จริง ช่วยให้ IK converge เร็ว)
q_init_pick  = np.array([np.radians(-36), np.radians(-45),
                          np.radians(60),  0, np.radians(-75), 0])
q_init_place = np.array([np.radians(60),  np.radians(-45),
                          np.radians(60),  0, np.radians(-75), 0])

print("="*55)
print("STEP 3: Jacobian IK — คำนวณ joint angles")
print("="*55)

print("\n[1] Hover (เหนือจุด pick):")
q_hov = solve_ik(np.array([px_hov, py_hov, HOVER_Z]), R_pick, q_init_pick)

print("\n[2] Pick (ระดับแก้ว):")
q_pck = solve_ik(np.array([px_pick, py_pick, pz_pick]), R_pick, q_hov)

print("\n[3] Place hover:")
q_plh = solve_ik(np.array([px_pl, py_pl, PLACE_HOVER_Z]), R_place, q_init_place)

print("\n[4] Place:")
q_pl  = solve_ik(np.array([px_pl, py_pl, pz_pl]), R_place, q_plh)

# ══════════════════════════════════════════════════════════════
# 5. STEP 3 — Generate joint trajectory arrays (tutorial slide 19)
#    t, j1, j2, j3, j4, j5, j6, gripper — ทุก dt=0.05s
# ══════════════════════════════════════════════════════════════
dt   = 0.05
N    = 400   # 20 วินาที
t_arr = [i*dt for i in range(N)]

# LFPB interpolation
def lfpb(t, q0, qf, tf, tb=None):
    if tb is None: tb = tf*0.2
    if tf < 1e-9:  return qf
    a = (qf-q0)/(tb*(tf-tb))
    if   t <= 0:       return q0
    elif t <= tb:      return q0 + 0.5*a*t**2
    elif t <= tf-tb:   return q0 + a*tb*(t-tb/2)
    elif t <= tf:      return qf - 0.5*a*(tf-t)**2
    else:              return qf

def interp_q(t_now, t0, t1, qa, qb):
    """LFPB interpolate ระหว่าง qa→qb ในช่วง [t0,t1]"""
    if t_now <= t0: return qa.copy()
    if t_now >= t1: return qb.copy()
    dur = t1 - t0
    return np.array([lfpb(t_now-t0, qa[i], qb[i], dur) for i in range(6)])

# Timing
T0  = 0.0    # home
T1  = 3.5    # → hover
T2  = T_WAIT # → เริ่มลง
T3  = T_PICK # → pick (grab)
T4  = 5.55   # → lift
T5  = 7.0    # → hover สิ้นสุด
T6  = 10.0   # → place hover
T7  = 11.5   # → place
T8  = 12.5   # → lift
T9  = 16.0   # → home

# Gripper velocity (tutorial slide 21)
V_OPEN  =  0.04   # m/s เปิด
V_CLOSE = -0.04   # m/s ปิด (จับแก้ว)
V_HOLD  = -0.04   # m/s ถือแก้วไว้

print("\nสร้าง joint trajectory arrays...")
j1_arr = []; j2_arr = []; j3_arr = []
j4_arr = []; j5_arr = []; j6_arr = []
grip_arr = []

for t_now in t_arr:
    # เลือก joint angles ตาม phase
    if   t_now < T1:  q = interp_q(t_now, T0, T1,  q_home, q_hov)
    elif t_now < T2:  q = q_hov.copy()
    elif t_now < T3:  q = interp_q(t_now, T2, T3,  q_hov,  q_pck)
    elif t_now < T4:  q = q_pck.copy()   # grab — คง position
    elif t_now < T5:  q = interp_q(t_now, T4, T5,  q_pck,  q_hov)
    elif t_now < T6:  q = interp_q(t_now, T5, T6,  q_hov,  q_plh)
    elif t_now < T7:  q = interp_q(t_now, T6, T7,  q_plh,  q_pl)
    elif t_now < T8:  q = q_pl.copy()    # place — คง position
    elif t_now < T9:  q = interp_q(t_now, T8, T9,  q_pl,   q_home)
    else:             q = q_home.copy()

    # gripper velocity
    if   t_now < T1:          v_grip = V_OPEN    # เปิดก่อนไปรอ
    elif t_now < T3:          v_grip = V_OPEN    # รอที่ hover (เปิดค้าง)
    elif t_now < T4:          v_grip = V_CLOSE   # GRAB
    elif t_now < T7:          v_grip = V_HOLD    # ถือระหว่างเดินทาง
    elif t_now < T8:          v_grip = V_CLOSE   # กด hold ตอนวาง
    else:                     v_grip = V_OPEN    # เปิดปล่อย+กลับ home

    j1_arr.append(q[0]); j2_arr.append(q[1]); j3_arr.append(q[2])
    j4_arr.append(q[3]); j5_arr.append(q[4]); j6_arr.append(q[5])
    grip_arr.append(v_grip)

print(f"  สร้าง {N} steps ({N*dt:.1f} วินาที) เสร็จแล้ว")
print(f"\nJoint angles ที่ keypoints (deg):")
print(f"  home : [{', '.join(f'{np.degrees(x):5.1f}' for x in q_home)}]")
print(f"  hover: [{', '.join(f'{np.degrees(x):5.1f}' for x in q_hov)}]")
print(f"  pick : [{', '.join(f'{np.degrees(x):5.1f}' for x in q_pck)}]")
print(f"  p.hov: [{', '.join(f'{np.degrees(x):5.1f}' for x in q_plh)}]")
print(f"  place: [{', '.join(f'{np.degrees(x):5.1f}' for x in q_pl)}]")

# ══════════════════════════════════════════════════════════════
# 6. STEP 4 — Run Simulation
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 4: Run Simulation")
print("="*55)

client = RemoteAPIClient()
sim    = client.require('sim')

joints = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1,7)]
m1 = sim.getObject('/yaskawa/MicoHand/fingers12_motor1')
m2 = sim.getObject('/yaskawa/MicoHand/fingers12_motor2')

sim.setStepping(True)
sim.startSimulation()

try:
    for step_i in range(N):
        # set joint positions (tutorial slide 18-19)
        sim.setJointPosition(joints[0], float(j1_arr[step_i]))
        sim.setJointPosition(joints[1], float(j2_arr[step_i]))
        sim.setJointPosition(joints[2], float(j3_arr[step_i]))
        sim.setJointPosition(joints[3], float(j4_arr[step_i]))
        sim.setJointPosition(joints[4], float(j5_arr[step_i]))
        sim.setJointPosition(joints[5], float(j6_arr[step_i]))

        # set gripper velocity (tutorial slide 21)
        sim.setJointTargetVelocity(m1, float(grip_arr[step_i]))
        sim.setJointTargetVelocity(m2, float(grip_arr[step_i]))

        sim.step()

    print("✓ Simulation เสร็จสิ้น!")

finally:
    sim.setStepping(False)
    sim.stopSimulation()