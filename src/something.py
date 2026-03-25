"""
step4_track_and_intercept.py
============================
รวมทุกอย่างในไฟล์เดียว:
  1. เชื่อมต่อ CoppeliaSim + ดึง joint/EE/cup handles
  2. บันทึกข้อมูล 20 วินาที (position, velocity, joint angles)
  3. Fit sine wave → predict cup trajectory
  4. หา intercept point ด้วย IK + manipulability check
  5. พล็อตผลทั้งหมด

ต้องมีไฟล์ gp8_kinematics.py อยู่ใน folder เดียวกัน
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from gp8_kinlib import GP8Kinematics

# ============================================================
# ตั้งค่า
# ============================================================
RECORD_DURATION = 20    # วินาทีที่บันทึก
SAMPLE_DT       = 0.05  # วินาทีต่อ sample

# ชื่อ joint จริง — แก้ให้ตรงกับที่ find_joints.py พิมพ์ออกมา
JOINT_NAMES = [
    '/yaskawa/joint1',
    '/yaskawa/joint2',
    '/yaskawa/joint3',
    '/yaskawa/joint4',
    '/yaskawa/joint5',
    '/yaskawa/joint6',
]
CUP_NAME = '/Cup'
EE_NAME  = '/yaskawa/MicoHand'

# ============================================================
# เชื่อมต่อ CoppeliaSim
# ============================================================
client = RemoteAPIClient()
sim    = client.require('sim')
sim.startSimulation()

def get_handle(name):
    """ดึง handle — ถ้าไม่เจอพิมพ์ชื่อที่ผิด แต่ไม่หยุดโปรแกรม"""
    try:
        h = sim.getObject(name)
        print(f"  [OK] {name}")
        return h
    except:
        print(f"  [WARN] หา '{name}' ไม่พบ — ตรวจชื่อใน CoppeliaSim")
        return None

print("=== ดึง Object handles ===")
cup_handle = get_handle(CUP_NAME)
ee_handle  = get_handle(EE_NAME)

joint_handles = []
for name in JOINT_NAMES:
    h = get_handle(name)
    joint_handles.append(h)   # append ทั้ง None และ handle จริง

has_joints = all(h is not None for h in joint_handles)
if not has_joints:
    print("\n[WARN] joint บางตัวหาไม่เจอ")
    print("       รัน find_joints.py ก่อน แล้วนำชื่อมาใส่ใน JOINT_NAMES")

if cup_handle is None or ee_handle is None:
    print("[ERROR] หา Cup หรือ MicoHand ไม่พบ หยุดโปรแกรม")
    sim.stopSimulation()
    exit()

# ============================================================
# บันทึกข้อมูล
# ============================================================
print(f"\n=== บันทึกข้อมูล {RECORD_DURATION} วินาที ===")

time_data = []
cup_pos   = []   # (N, 3)
cup_vel   = []   # (N, 3)
ee_pos    = []   # (N, 3)
q_data    = []   # (N, 6) — joint angles จริงจาก sim

q_last = np.zeros(6)   # fallback ถ้าหา joint ไม่ได้

start_time = sim.getSimulationTime()

while True:
    t_now = sim.getSimulationTime()
    elapsed = t_now - start_time
    if elapsed > RECORD_DURATION:
        break

    # --- ตำแหน่ง ---
    p_cup = sim.getObjectPosition(cup_handle, sim.handle_world)
    p_ee  = sim.getObjectPosition(ee_handle,  sim.handle_world)

    # --- ความเร็วแก้ว ---
    v_cup_lin, _ = sim.getObjectVelocity(cup_handle)

    # --- joint angles จริง ---
    if has_joints:
        q_now = np.array([sim.getJointPosition(jh) for jh in joint_handles])
        q_last = q_now
    else:
        q_now = q_last   # ใช้ค่าเดิมถ้าหา joint ไม่ได้

    time_data.append(elapsed)
    cup_pos.append(p_cup[:3])
    cup_vel.append(v_cup_lin[:3])
    ee_pos.append(p_ee[:3])
    q_data.append(q_now.copy())

    time.sleep(SAMPLE_DT)

sim.stopSimulation()
print(f"บันทึกเสร็จ: {len(time_data)} จุด")

# แปลง numpy
t_arr    = np.array(time_data)
cup_arr  = np.array(cup_pos)    # (N, 3)
vcp_arr  = np.array(cup_vel)    # (N, 3)
ee_arr   = np.array(ee_pos)     # (N, 3)
q_arr    = np.array(q_data)     # (N, 6)

# joint angles สุดท้าย (ใช้เป็น q_init ของ IK)
q_current = q_arr[-1]
print(f"q_current (°): {np.degrees(q_current).round(2)}")

# ============================================================
# ขั้น 3A: FK check — เปรียบ FK ของเราและ CoppeliaSim
# ============================================================
kin = GP8Kinematics()
fk_pos = np.array([kin.fk_position(q) for q in q_arr])
fk_err = np.linalg.norm(ee_arr - fk_pos, axis=1)
print(f"\nFK error เฉลี่ย: {fk_err.mean()*1000:.2f} mm  (ถ้า > 50mm แสดงว่า DH ไม่ตรง)")

# ============================================================
# ขั้น 3B: Fit sine wave
# ============================================================
def sine_model(t, A, omega, phi, offset):
    return A * np.sin(omega * t + phi) + offset

def fit_sine(t_data, y_data, axis_name):
    A0      = (y_data.max() - y_data.min()) / 2
    offset0 = y_data.mean()
    fft_v   = np.abs(np.fft.rfft(y_data - offset0))
    freqs   = np.fft.rfftfreq(len(y_data), d=(t_data[1]-t_data[0]))
    omega0  = 2 * np.pi * freqs[np.argmax(fft_v[1:]) + 1]
    try:
        popt, _ = curve_fit(sine_model, t_data, y_data,
                            p0=[A0, omega0, 0.0, offset0], maxfev=10000)
        y_fit = sine_model(t_data, *popt)
        r2    = 1 - np.sum((y_data-y_fit)**2) / np.sum((y_data-offset0)**2)
        A, omega, phi, offset = popt
        print(f"  {axis_name}: A={A:.4f}m  ω={omega:.4f}rad/s"
              f"  φ={phi:.4f}  offset={offset:.4f}  R²={r2:.4f}")
        return popt
    except Exception as e:
        print(f"  {axis_name}: fit ไม่ได้ ({e})")
        return [A0, omega0, 0.0, offset0]

print("\n=== Fit Sine Wave ===")
popt_x = fit_sine(t_arr, cup_arr[:, 0], 'X')
popt_y = fit_sine(t_arr, cup_arr[:, 1], 'Y')
popt_z = fit_sine(t_arr, cup_arr[:, 2], 'Z')

def predict_cup(t_future):
    """Predict ตำแหน่ง + ความเร็วแก้ว ณ เวลา t_future"""
    pos = np.array([
        sine_model(t_future, *popt_x),
        sine_model(t_future, *popt_y),
        sine_model(t_future, *popt_z),
    ])
    Ax, wx, phix, _ = popt_x
    Ay, wy, phiy, _ = popt_y
    Az, wz, phiz, _ = popt_z
    vel = np.array([
        Ax * wx * np.cos(wx*t_future + phix),
        Ay * wy * np.cos(wy*t_future + phiy),
        Az * wz * np.cos(wz*t_future + phiz),
    ])
    return pos, vel

# ============================================================
# ขั้น 3C: หา Intercept Point
# ============================================================
ARM_SPEED   = 0.5    # m/s
SING_THRESH = 0.005
MAX_IK_ERR  = 0.02   # m
t_now_val   = t_arr[-1]

print(f"\n=== หา Intercept Point (จาก t={t_now_val:.2f}s) ===")
best = None

for dt_ahead in np.arange(0.5, 20.0, 0.1):
    pos_pred, vel_pred = predict_cup(t_now_val + dt_ahead)

    # แขนไปทัน?
    ee_now = kin.fk_position(q_current)
    dist   = np.linalg.norm(pos_pred - ee_now)
    if dist / ARM_SPEED > dt_ahead:
        continue

    # IK
    q_sol, ok, err = kin.ik(pos_pred, q_init=q_current)
    if not ok or err > MAX_IK_ERR:
        continue

    # Manipulability
    m = kin.manipulability(q_sol)
    if m < SING_THRESH:
        print(f"  +{dt_ahead:.1f}s singularity (m={m:.5f})")
        continue

    best = dict(dt=dt_ahead, t=t_now_val+dt_ahead,
                pos=pos_pred, vel=vel_pred, q=q_sol, m=m, err_mm=err*1000)
    print(f"  +{dt_ahead:.1f}s [OK] pos={pos_pred.round(3)}"
          f"  m={m:.4f}  IK={err*1000:.2f}mm")
    break

if best:
    print(f"\n=== Intercept Point ===")
    print(f"ไปรอที่ : {best['pos'].round(4)} m")
    print(f"เวลา    : t = {best['t']:.2f}s  (อีก {best['dt']:.1f}s)")
    print(f"q (°)   : {np.degrees(best['q']).round(2)}")
    print(f"Manip   : {best['m']:.5f}")
    print(f"IK err  : {best['err_mm']:.3f} mm")
    print(f"cup vel : {best['vel'].round(4)} m/s")
else:
    print("\n[WARN] หาไม่ได้ — ลองเพิ่ม ARM_SPEED หรือ MAX_IK_ERR")

# ============================================================
# พล็อต
# ============================================================
t_ext  = np.linspace(0, t_now_val + 15, 600)
pred   = np.array([predict_cup(tt)[0] for tt in t_ext])
v_pred = np.array([predict_cup(tt)[1] for tt in t_ext])

# --- Position + Velocity + Intercept ---
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Cup Trajectory — Sine Fit + Intercept Point', fontsize=13, fontweight='bold')
col_names = ['X', 'Y', 'Z']

for col in range(3):
    ax = axes[0, col]
    ax.set_title(f'Position {col_names[col]} (m)', fontsize=10)
    ax.set_xlabel('t (s)'); ax.grid(True, alpha=0.3)
    ax.plot(t_arr, cup_arr[:, col], 'r--', lw=1.5, label='วัดได้')
    ax.plot(t_ext, pred[:, col], 'b-', lw=1.5, alpha=0.7, label='Sine fit')
    if best:
        ax.axvline(best['t'], color='green', ls='--', lw=1.5, label='Intercept')
        ax.scatter(best['t'], best['pos'][col], c='green', s=150, zorder=10)
    if col == 0:
        ax.legend(fontsize=8)

    ax2 = axes[1, col]
    ax2.set_title(f'Velocity {col_names[col]} (m/s)', fontsize=10)
    ax2.set_xlabel('t (s)'); ax2.grid(True, alpha=0.3)
    ax2.plot(t_arr, vcp_arr[:, col], 'r--', lw=1.5, label='วัดได้')
    ax2.plot(t_ext, v_pred[:, col], 'b-', lw=1.5, alpha=0.7, label='Sine fit')
    if best:
        ax2.axvline(best['t'], color='green', ls='--', lw=1.5)
        ax2.scatter(best['t'], best['vel'][col], c='green', s=150, zorder=10)

plt.tight_layout()
plt.savefig('step4_intercept.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FK error ---
fig2, ax3 = plt.subplots(figsize=(10, 4))
ax3.set_title('FK Error (library vs CoppeliaSim)', fontweight='bold')
ax3.plot(t_arr, fk_err * 1000, color='coral', lw=1.5)
ax3.axhline(fk_err.mean() * 1000, color='red', ls='--',
            label=f'เฉลี่ย = {fk_err.mean()*1000:.2f} mm')
ax3.set_xlabel('t (s)'); ax3.set_ylabel('Error (mm)')
ax3.grid(True, alpha=0.3); ax3.legend(fontsize=9)
ax3.text(0.02, 0.95,
         'ถ้า error < 30mm = DH ใช้ได้\nถ้า error สูง = ต้องปรับ DH ใน gp8_kinematics.py',
         transform=ax3.transAxes, va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.show()

# --- 3D ---
fig3d = plt.figure(figsize=(9, 7))
ax3d  = fig3d.add_subplot(111, projection='3d')
ax3d.set_title('Cup Path + EE Path + Intercept Point', fontweight='bold')
ax3d.plot(*cup_arr.T, 'r--', lw=1, alpha=0.5, label='Cup วัดได้')
ax3d.plot(*pred.T, 'b-', lw=2, alpha=0.8, label='Cup predicted')
ax3d.plot(*ee_arr.T, 'c-', lw=1.5, alpha=0.6, label='EE path')
if best:
    ax3d.scatter(*best['pos'], c='green', s=250, marker='*', zorder=10, label='Intercept')
    ax3d.plot([ee_arr[-1, 0], best['pos'][0]],
              [ee_arr[-1, 1], best['pos'][1]],
              [ee_arr[-1, 2], best['pos'][2]],
              'g--', lw=1.5, alpha=0.7, label='EE → Intercept')
ax3d.scatter(*ee_arr[-1], c='blue', s=100, zorder=10, label='EE ตอนนี้')
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ============================================================
# สรุป — สิ่งที่เอาไปใช้ขั้น 5 (ส่งแขนไป intercept)
# ============================================================
print("\n" + "="*50)
print("สรุปสิ่งที่ได้จาก step 4:")
print("="*50)
if best:
    print(f"  intercept_pos = {best['pos'].round(4)}")
    print(f"  intercept_q   = {np.degrees(best['q']).round(2)} °")
    print(f"  cup_vel_at_intercept = {best['vel'].round(4)} m/s")
    print(f"  manipulability = {best['m']:.5f}")
    print(f"\nขั้นต่อไป (step 5):")
    print(f"  1. ส่ง joint angles intercept_q ไปที่ CoppeliaSim")
    print(f"  2. ใช้ velocity_ik ติดตามแก้วแบบ real-time")
    print(f"  3. Grasp แล้วไปวางที่สายพานปลายทาง")
print("="*50)