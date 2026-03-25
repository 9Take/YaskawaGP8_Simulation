'''
Scene: yaskawaGP8_linearConveyor.ttt

Description: 
- get the cup position, velocity and acceleration
- get the cup orientation roll, pitch and yaw
'''

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
#from zmqRemoteApi import RemoteAPIClient
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import threading
import getch
import math
Pi = math.pi

robotNum = 'GP8'
client = RemoteAPIClient()
sim = client.require('sim')

executedMovId = 'notReady'
targetArm = '/yaskawa'
stringSignalName = targetArm + '_executedMovId'
sim.setStepping(True)

sim.startSimulation()

cup=sim.getObject('./Cup')
cup_acc_sensor = sim.getObject('./Accelerometer')

t = []
# linear
cupPx = []
cupPy = []
cupPz = []
cupVx = []
cupVy = []
cupVz = []
cupAx = []
cupAy = []
cupAz = []
# angular
cupP_alpha = []
cupP_beta = []
cupP_gamma = []
cupV_alpha = []
cupV_beta = []
cupV_gamma = []

i = 0
step = 380
dt = sim.getSimulationTimeStep()

while not sim.getSimulationStopping():
    print('t %d: %.2f sec' % (i, sim.getSimulationTime()))
    cupPos = sim.getObjectPosition(cup, sim.handle_world)
    alpha, beta, gamma = sim.getObjectOrientation(cup, sim.handle_world)
    cupL_Vel, cupA_Vel = sim.getObjectVelocity(cup, sim.handle_world)
    cup_accelX = sim.getFloatSignal("accelerometerX")
    cup_accelY = sim.getFloatSignal("accelerometerY")
    cup_accelZ = sim.getFloatSignal("accelerometerZ")
    # linear
    print('Cup linear')
    print('Cup pos %d: x=%.3f, y=%.3f, z=%.3f ' % (i, cupPos[0], cupPos[1], cupPos[2]))
    print('Cup vel %d: x=%.3f, y=%.3f, z=%.3f ' % (i, cupL_Vel[0], cupL_Vel[1], cupL_Vel[2]))
    print('Cup acc %d: x=%s, y=%s, z=%s ' % (i, cup_accelX, cup_accelY, cup_accelZ))
    # angular
    print('Cup angular')
    print('Cup angular pos %d: x=%.3f, y=%.3f, z=%.3f ' % (i, alpha, beta, gamma))
    print('Cup angular vel %d: x=%.3f, y=%.3f, z=%.3f ' % (i, cupA_Vel[0], cupA_Vel[1], cupA_Vel[2]))
    
    # get data to list
    t.append(sim.getSimulationTime())
    # linear
    cupPx.append(cupPos[0])
    cupPy.append(cupPos[1])
    cupPz.append(cupPos[2])
    cupVx.append(cupL_Vel[0])
    cupVy.append(cupL_Vel[1])
    cupVz.append(cupL_Vel[2])
    cupAx.append(cup_accelX)
    cupAy.append(cup_accelY)
    cupAz.append(cup_accelZ)
    # angular
    cupP_alpha.append(alpha)
    cupP_beta.append(beta)
    cupP_gamma.append(gamma)
    cupV_alpha.append(cupA_Vel[0])
    cupV_beta.append(cupA_Vel[1])
    cupV_gamma.append(cupA_Vel[2])
    
    if i==step:
        break
    i = i + 1
    sim.step()
    
print(dt)  

# plot -------------------------------------------------
plt.figure # plot linear position

plt.subplot(2,3,1)
plt.title('t and linear position x')
plt.xlabel('t(s)')
plt.ylabel('m')
plt.plot(t,cupPx, '--r', label='cup x')
plt.legend(title='where:')

plt.subplot(2,3,2)
plt.title('t and linear position y')
plt.xlabel('t(s)')
plt.ylabel('m')
plt.legend()
plt.plot(t,cupPy, '--r', label='cup y')
plt.legend(title='where:')

plt.subplot(2,3,3)
plt.title('t and linear position z')
plt.xlabel('t(s)')
plt.ylabel('m')
plt.legend()
plt.plot(t,cupPz, '--r', label='cup z')
plt.legend(title='where:')

plt.subplot(2,3,4)
plt.title('t and linear velocity x')
plt.xlabel('t(s)')
plt.ylabel('m/s')
plt.legend()
plt.plot(t,cupVx, '--r', label='cup Vx')
plt.legend(title='where:')

plt.subplot(2,3,5)
plt.title('t and linear velocity y')
plt.xlabel('t(s)')
plt.ylabel('m/s')
plt.legend()
plt.plot(t,cupVy, '--r', label='cup Vy')
plt.legend(title='where:')

plt.subplot(2,3,6)
plt.title('t and linear velocity z')
plt.xlabel('t(s)')
plt.ylabel('m/s')
plt.legend()
plt.plot(t,cupVz, '--r', label='cup Vz')
plt.legend(title='where:')

plt.show()

plt.figure # plot angular position

plt.subplot(2,3,1)
plt.title('t and angular position x of the cup')
plt.xlabel('t(s)')
plt.ylabel('rad')
plt.legend()
plt.plot(t,cupP_alpha, '--r')

plt.subplot(2,3,2)
plt.title('t and angular position y of the cup')
plt.xlabel('t(s)')
plt.ylabel('rad')
plt.legend()
plt.plot(t,cupP_beta, '--r')

plt.subplot(2,3,3)
plt.title('t and angular position z of the cup')
plt.xlabel('t(s)')
plt.ylabel('rad')
plt.legend()
plt.plot(t,cupP_gamma, '--r')

plt.subplot(2,3,4)
plt.title('t and angular velocity x of the cup')
plt.xlabel('t(s)')
plt.ylabel('rad/s')
plt.legend()
plt.plot(t,cupV_alpha, '--r')

plt.subplot(2,3,5)
plt.title('t and angular velocity y of the cup')
plt.xlabel('t(s)')
plt.ylabel('rad/s')
plt.legend()
plt.plot(t,cupV_beta, '--r')

plt.subplot(2,3,6)
plt.title('t and angular velocity z of the cup')
plt.xlabel('t(s)')
plt.ylabel('rad/s')
plt.legend()
plt.plot(t,cupV_gamma, '--r')

plt.show()

fig = plt.figure() # 3D line plot
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
ax.set_aspect('auto')
ax.plot3D(cupPx, cupPy, cupPz, 'green')
ax.set_title('3D line plot the cup (x,y,z)')

plt.show()

plt.figure # acceleration
plt.subplot(1,3,1)
plt.title('t and acceleration x of the cup')
plt.plot(t,cupAx, '--r')
plt.xlabel('t(s)')
plt.ylabel('m/s^2')
plt.legend()
plt.subplot(1,3,2)
plt.title('t and acceleration y of the cup')
plt.plot(t,cupAy, '--r')
plt.xlabel('t(s)')
plt.ylabel('m/s^2')
plt.legend()
plt.subplot(1,3,3)
plt.title('t and acceleration z of the cup')
plt.plot(t,cupAz, '--r')
plt.xlabel('t(s)')
plt.ylabel('m/s^2')
plt.legend()
plt.show()

# stop program -----------------------------------------
sim.setStepping(False)
sim.stopSimulation()

"""
step3_intercept.py
==================
วางไว้หลัง sim.stopSimulation() ของโค้ดอาจารย์

สิ่งที่ทำ:
  1. Fit sine wave จาก cupPx, cupPy, cupPz
  2. Predict ตำแหน่งแก้วในอนาคต
  3. หา intercept point ที่ EE ไปรอได้ (IK + manipulability check)
  4. พล็อตให้เห็นภาพ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from gp8_kinematics import GP8Kinematics

# ============================================================
# รับข้อมูลจากโค้ดอาจารย์
# cupPx, cupPy, cupPz, cupVx, cupVy, cupVz, t ต้องมีอยู่แล้ว
# ============================================================
px    = np.array(cupPx)
py    = np.array(cupPy)
pz    = np.array(cupPz)
t_arr = np.array(t)

# ============================================================
# ขั้น 3A: Fit sine wave แต่ละแกน
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
        print(f"  {axis_name}: A={A:.4f}  omega={omega:.4f}rad/s"
              f"  phi={phi:.4f}  offset={offset:.4f}  R²={r2:.4f}")
        return popt
    except Exception as e:
        print(f"  {axis_name}: fit ไม่ได้ ({e})")
        return [A0, omega0, 0.0, offset0]

print("=== Fit Sine Wave ===")
popt_x = fit_sine(t_arr, px, 'X')
popt_y = fit_sine(t_arr, py, 'Y')
popt_z = fit_sine(t_arr, pz, 'Z')

# ============================================================
# ขั้น 3B: ฟังก์ชัน predict
# ============================================================
def predict_cup(t_future):
    """
    Predict ตำแหน่ง + ความเร็วแก้ว ณ เวลา t_future
    Returns: pos array(3,), vel array(3,)
    """
    Ax, wx, phix, cx = popt_x
    Ay, wy, phiy, cy = popt_y
    Az, wz, phiz, cz = popt_z
    pos = np.array([
        Ax * np.sin(wx*t_future + phix) + cx,
        Ay * np.sin(wy*t_future + phiy) + cy,
        Az * np.sin(wz*t_future + phiz) + cz,
    ])
    vel = np.array([
        Ax * wx * np.cos(wx*t_future + phix),
        Ay * wy * np.cos(wy*t_future + phiy),
        Az * wz * np.cos(wz*t_future + phiz),
    ])
    return pos, vel

def predict_now(t_current, dt_ahead):
    """ใช้ใน real-time control loop"""
    return predict_cup(t_current + dt_ahead)

# ============================================================
# ขั้น 3C: หา Intercept Point
# ============================================================
kin          = GP8Kinematics()
q_current    = np.zeros(6)   # ← ใส่ค่า joint จริงถ้ามี
ARM_SPEED    = 0.5            # m/s (ประมาณความเร็วแขน)
SING_THRESH  = 0.005          # manipulability ต่ำกว่านี้ = singularity
MAX_IK_ERR   = 0.02           # m
t_now        = t_arr[-1]

print(f"\n=== หา Intercept Point (จาก t={t_now:.2f}s) ===")
best = None

for dt_ahead in np.arange(0.5, 15.0, 0.1):
    pos_pred, vel_pred = predict_cup(t_now + dt_ahead)

    # ตรวจแขนไปทัน
    ee_now   = kin.fk_position(q_current)
    dist     = np.linalg.norm(pos_pred - ee_now)
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

    best = dict(dt=dt_ahead, t=t_now+dt_ahead,
                pos=pos_pred, vel=vel_pred, q=q_sol, m=m, err=err*1000)
    print(f"  +{dt_ahead:.1f}s OK  pos={pos_pred.round(3)}"
          f"  m={m:.4f}  IK={err*1000:.2f}mm")
    break

if best:
    print(f"\n=== ผลลัพธ์ ===")
    print(f"ไปรอที่  : {best['pos'].round(4)} m")
    print(f"เวลา     : t = {best['t']:.2f}s  (อีก {best['dt']:.1f}s)")
    print(f"q (°)    : {np.degrees(best['q']).round(2)}")
    print(f"Manip    : {best['m']:.5f}")
    print(f"IK error : {best['err']:.3f} mm")
    print(f"ความเร็วแก้วตอนนั้น: {best['vel'].round(4)} m/s")
else:
    print("\n[WARN] หาไม่ได้ — ลองเพิ่ม ARM_SPEED หรือ MAX_IK_ERR")

# ============================================================
# พล็อต
# ============================================================
t_ext = np.linspace(0, t_now + 15, 600)
pred  = np.array([predict_cup(tt)[0] for tt in t_ext])
v_ext = np.array([predict_cup(tt)[1] for tt in t_ext])

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Cup Trajectory — Sine Fit + Intercept Point',
             fontsize=13, fontweight='bold')

data_pos  = [px, py, pz]
data_vel  = [np.array(cupVx), np.array(cupVy), np.array(cupVz)]
col_names = ['X', 'Y', 'Z']

for col in range(3):
    # --- position ---
    ax = axes[0, col]
    ax.set_title(f'Position {col_names[col]} (m)', fontsize=10)
    ax.set_xlabel('t (s)'); ax.grid(True, alpha=0.3)
    ax.plot(t_arr, data_pos[col], 'r--', lw=1.5, label='วัดได้')
    ax.plot(t_ext, pred[:, col], 'b-', lw=1.5, alpha=0.7, label='Sine fit')
    if best:
        ax.axvline(best['t'], color='green', ls='--', lw=1.5, label='Intercept')
        ax.scatter(best['t'], best['pos'][col], c='green', s=120, zorder=10)
    if col == 0:
        ax.legend(fontsize=8)

    # --- velocity ---
    ax2 = axes[1, col]
    ax2.set_title(f'Velocity {col_names[col]} (m/s)', fontsize=10)
    ax2.set_xlabel('t (s)'); ax2.grid(True, alpha=0.3)
    ax2.plot(t_arr, data_vel[col], 'r--', lw=1.5, label='วัดได้')
    ax2.plot(t_ext, v_ext[:, col], 'b-', lw=1.5, alpha=0.7, label='Sine fit')
    if best:
        ax2.axvline(best['t'], color='green', ls='--', lw=1.5)
        ax2.scatter(best['t'], best['vel'][col], c='green', s=120, zorder=10,
                    label=f"v={best['vel'][col]:.3f} m/s")
    if col == 0:
        ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('step3_intercept.png', dpi=150, bbox_inches='tight')
plt.show()

# 3D
fig3d = plt.figure(figsize=(9, 7))
ax3d  = fig3d.add_subplot(111, projection='3d')
ax3d.set_title('Cup Path + Intercept Point', fontweight='bold')
ax3d.plot(px, py, pz, 'r--', lw=1, alpha=0.6, label='วัดได้')
ax3d.plot(pred[:,0], pred[:,1], pred[:,2], 'b-', lw=2, alpha=0.8, label='Predicted')
if best:
    ax3d.scatter(*best['pos'], c='green', s=250, marker='*', zorder=10, label='Intercept')
    ee = kin.fk_position(q_current)
    ax3d.scatter(*ee, c='blue', s=100, zorder=10, label='EE ตอนนี้')
    ax3d.plot([ee[0], best['pos'][0]], [ee[1], best['pos'][1]], [ee[2], best['pos'][2]],
              'g--', lw=1.5, alpha=0.6)
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.legend(fontsize=8)
plt.tight_layout()
plt.show()

print("\n[READY] predict_now(t_current, dt_ahead) พร้อมใช้ใน real-time loop")

