"""
Part 3 : Execute trajectory — DIRECT Python Control (Lua bypass)
=================================================================
Root cause (confirmed by test_deep_diag.py):
  The Lua child script on /yaskawa does NOT execute in CoppeliaSim
  v4.10 — sysCall_init never fires, callScriptFunction returns
  "script does not exist".  ALL signal-based approaches fail.

Solution:
  Control everything from Python via the ZMQ Remote API in
  **stepping mode**, the same approach used by Previous_Year65:
    • Arm joints  → setJointTargetPosition / setJointPosition
    • Gripper     → setJointTargetVelocity on finger motors
    • Cup attach  → setObjectParent  (toggle USE_ATTACH below)
    • Physics     → sim.step() per iteration

Usage
-----
  1. Open  yaskawaGP8_group19.ttt  in CoppeliaSim  (don't press Play)
  2. python part3_run.py
"""

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial.transform import Rotation as R

# ╔═══════════════════════════════════════════════════════════════╗
# ║  TUNABLE  — adjust and re-run (no IK recomputation)         ║
# ╚═══════════════════════════════════════════════════════════════╝
USE_ATTACH       = True    # True  = setObjectParent to guarantee cup sticks
                           # False = rely on physics friction only
ATTACH_DELAY     = 5       # steps after gripper-close starts → attach cup
DETACH_ADVANCE   = 0       # steps before gripper-open starts → detach cup
GRIP_VEL_SCALE   = 1.0     # multiply all gripper_cmd velocities (>1 = faster)
# ════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────
#  1.  LOAD TRAJECTORY  (from part2_ik.py)
# ──────────────────────────────────────────────────────────────
d = np.load('trajectory.npz', allow_pickle=True)

j_traj      = [np.array(d[f'j{i}']) for i in range(1, 7)]   # j_traj[0]=j1 …
gripper_cmd = np.array(d['gripper_cmd'])
ef_x_plan   = np.array(d['ef_x'])
ef_y_plan   = np.array(d['ef_y'])
ef_z_plan   = np.array(d['ef_z'])
GRAB_STEP   = int(d['GRAB_STEP'])
DT          = float(d['DT'])
STEPS       = int(d['STEPS'])
ef_home     = d['ef_home']
robot_base  = d['robot_base']
place_pos   = d['place_pos']

print(f"  Trajectory : {STEPS} steps  ({STEPS*DT:.1f} s),  DT = {DT}")
print(f"  GRAB_STEP  = {GRAB_STEP}  ({GRAB_STEP*DT:.1f} s)")

# Identify gripper phase changes
close_start = None   # first step where cmd < -0.1 (closing)
open_start  = None   # first step where cmd > +0.1 (opening after close)
for i in range(STEPS):
    if close_start is None and gripper_cmd[i] < -0.1:
        close_start = i
    if close_start is not None and open_start is None and gripper_cmd[i] > 0.1:
        open_start = i
        break

attach_step = (close_start + ATTACH_DELAY) if close_start else GRAB_STEP + ATTACH_DELAY
detach_step = (open_start - DETACH_ADVANCE) if open_start else STEPS

print(f"  Gripper    : close_start={close_start}  open_start={open_start}")
print(f"  Cup        : attach@step {attach_step}   detach@step {detach_step}")
print(f"  USE_ATTACH = {USE_ATTACH}")


# ──────────────────────────────────────────────────────────────
#  2.  CONNECT  &  HANDLES
# ──────────────────────────────────────────────────────────────
client = RemoteAPIClient()
sim    = client.require('sim')

joint_h  = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
ef_h     = sim.getObject('/yaskawa/gripperEF')
finger1  = sim.getObject('/yaskawa/MicoHand/fingers12_motor1')
finger2  = sim.getObject('/yaskawa/MicoHand/fingers12_motor2')
cup_h    = sim.getObject('/conveyorSystem/Cup')

print(f"  joints     = {joint_h}")
print(f"  fingers    = ({finger1}, {finger2})")
print(f"  gripperEF  = {ef_h},  cup = {cup_h}")


# ──────────────────────────────────────────────────────────────
#  3.  TRY TO ENABLE  /yaskawa  SCRIPT  (might help child scripts)
# ──────────────────────────────────────────────────────────────
try:
    robot_h  = sim.getObject('/yaskawa')
    script_h = sim.getScript(sim.scripttype_childscript, robot_h)
    enabled  = sim.getScriptInt32Param(script_h, sim.scriptintparam_enabled)
    print(f"  /yaskawa script : handle={script_h}  enabled={enabled}")
    if not enabled:
        sim.setScriptInt32Param(script_h, sim.scriptintparam_enabled, 1)
        print("  → Enabled /yaskawa script")
except Exception as e:
    print(f"  Script enable : {e}")

# Check MicoHand / gripperEF scripts
for name in ['/yaskawa/MicoHand', '/yaskawa/gripperEF']:
    try:
        oh = sim.getObject(name)
        sh = sim.getScript(sim.scripttype_childscript, oh)
        en = sim.getScriptInt32Param(sh, sim.scriptintparam_enabled)
        print(f"  {name:30s}  script={sh}  enabled={en}")
    except Exception as e:
        print(f"  {name:30s}  {e}")


# ──────────────────────────────────────────────────────────────
#  4.  START SIMULATION  (stepping mode)
# ──────────────────────────────────────────────────────────────
sim.setStepping(True)
sim.startSimulation()

# Cache joint dynamic flags (don't re-query every step)
arm_dyn = []
for j in range(6):
    mode = sim.getJointMode(joint_h[j])
    dyn  = sim.isDynamicallyEnabled(joint_h[j])
    is_dyn = (mode == sim.jointmode_force and dyn)
    arm_dyn.append(is_dyn)
    label = 'setJointTargetPosition' if is_dyn else 'setJointPosition'
    print(f"  joint{j+1} → {label}  (mode={mode}, dyn={dyn})")

f1_dyn = sim.isDynamicallyEnabled(finger1)
print(f"  finger1 → setJointTargetVelocity  (dyn={f1_dyn})")

# Warm-up (let physics settle)
for _ in range(5):
    sim.step()

# Quick Lua signal check (just in case enabling helped)
sig = sim.getStringSignal('/yaskawa_executedMovId')
if sig is not None:
    sstr = sig.decode() if isinstance(sig, bytes) else str(sig)
    print(f"  Lua signal after enable: '{sstr}' — nice, but using direct control")
else:
    print(f"  Lua signal still None — confirming: direct control mode")


# ──────────────────────────────────────────────────────────────
#  5.  EXECUTE  TRAJECTORY
# ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  EXECUTING — Direct Python Control  ({STEPS} steps)")
print(f"{'='*60}")

rec_t   = []
rec_ef  = []
rec_cup = []
rec_j   = []
rec_fg  = []
rec_ef_ori = []   # EF orientation (Euler XYZ in degrees)

cup_attached = False
cup_detached = False

t_wall_0 = time.time()

for step in range(STEPS):

    # ── 5a. Arm joints ──────────────────────────────────────
    for j in range(6):
        pos = float(j_traj[j][step])
        if arm_dyn[j]:
            sim.setJointTargetPosition(joint_h[j], pos)
        else:
            sim.setJointPosition(joint_h[j], pos)

    # ── 5b. Gripper motors ──────────────────────────────────
    gv = float(gripper_cmd[step]) * GRIP_VEL_SCALE
    sim.setJointTargetVelocity(finger1, gv)
    sim.setJointTargetVelocity(finger2, gv)

    # ── 5c. Cup attachment / detachment ─────────────────────
    if USE_ATTACH:
        if step == attach_step and not cup_attached:
            try:
                # Disable cup dynamics so it follows the EF kinematically
                sim.setObjectInt32Param(cup_h, sim.shapeintparam_static, 1)
                sim.resetDynamicObject(cup_h)
                sim.setObjectParent(cup_h, ef_h, True)
                cup_attached = True
                print(f"    ★ step {step} ({step*DT:.2f}s): CUP ATTACHED to gripperEF (dynamics OFF)")
            except Exception as e:
                print(f"    ✗ Attach error: {e}")

        if step == detach_step and cup_attached and not cup_detached:
            try:
                sim.setObjectParent(cup_h, -1, True)
                # Re-enable cup dynamics so it falls naturally
                sim.setObjectInt32Param(cup_h, sim.shapeintparam_static, 0)
                sim.resetDynamicObject(cup_h)
                cup_detached = True
                print(f"    ★ step {step} ({step*DT:.2f}s): CUP RELEASED (dynamics ON)")
            except Exception as e:
                print(f"    ✗ Detach error: {e}")

    # ── 5d. Advance physics ─────────────────────────────────
    sim.step()

    # ── 5e. Record data ─────────────────────────────────────
    try:
        t_now   = sim.getSimulationTime()
        ef_pos  = sim.getObjectPosition(ef_h, sim.handle_world)
        cup_pos = sim.getObjectPosition(cup_h, sim.handle_world)
        jp      = [sim.getJointPosition(jh) for jh in joint_h]
        fp      = sim.getJointPosition(finger1)
        ef_ori  = sim.getObjectOrientation(ef_h, sim.handle_world)  # Euler XYZ rad

        rec_t.append(t_now)
        rec_ef.append(ef_pos)
        rec_cup.append(cup_pos)
        rec_j.append(jp)
        rec_fg.append(fp)
        rec_ef_ori.append([np.degrees(a) for a in ef_ori])  # store as degrees
    except Exception:
        pass

    # ── 5f. Progress ────────────────────────────────────────
    if step % 50 == 0 or step == STEPS - 1:
        ep = rec_ef[-1]  if rec_ef  else [0, 0, 0]
        cp = rec_cup[-1] if rec_cup else [0, 0, 0]
        fg = rec_fg[-1]  if rec_fg  else 0
        tag = ""
        if   cup_attached and not cup_detached: tag = " [HOLDING]"
        elif cup_detached:                      tag = " [RELEASED]"
        print(f"    step {step:4d}/{STEPS}  t={step*DT:5.1f}s"
              f"  ef=({ep[0]:+.3f},{ep[1]:+.3f},{ep[2]:+.3f})"
              f"  cup=({cp[0]:+.3f},{cp[1]:+.3f},{cp[2]:+.3f})"
              f"  fg={fg:+.5f}{tag}")


wall = time.time() - t_wall_0
print(f"\n  Done in {wall:.1f}s wall-time.  Recorded {len(rec_t)} samples.")

input("\n  Press Enter to stop simulation and show plots...")
sim.stopSimulation()


# ──────────────────────────────────────────────────────────────
#  6.  PLOTS
# ──────────────────────────────────────────────────────────────
if len(rec_t) < 5:
    print("  Not enough data to plot.")
    exit()

rec_ef  = np.array(rec_ef)
rec_cup = np.array(rec_cup)
rec_j   = np.array(rec_j)
rec_fg  = np.array(rec_fg)
rec_t   = np.array(rec_t)
rec_ef_ori = np.array(rec_ef_ori)   # (N, 3) degrees

# -- Derived quantities for rubric plots --
# EF linear velocity  (central differences)
ef_vel = np.gradient(rec_ef, rec_t, axis=0)          # (N, 3) m/s
ef_speed = np.linalg.norm(ef_vel, axis=1)             # (N,) m/s

# Cup linear velocity
cup_vel = np.gradient(rec_cup, rec_t, axis=0)         # (N, 3) m/s
cup_speed = np.linalg.norm(cup_vel, axis=1)

# EF angular velocity (derivative of orientation, simple finite diff)
ef_ang_vel = np.gradient(rec_ef_ori, rec_t, axis=0)   # deg/s

# Joint velocities (from recorded joint positions)
joint_vel = np.gradient(rec_j, rec_t, axis=0)          # (N, 6) rad/s

# Gripper finger velocity
finger_vel = np.gradient(rec_fg, rec_t)                # m/s

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Work 1 — Yaskawa GP8 Pick & Place — Group 19  (Direct Control)',
             fontsize=14)

# ── Row 0: EF position x, y, z ──
plans = [ef_x_plan, ef_y_plan, ef_z_plan]
for col, (label, plan) in enumerate(zip(['x', 'y', 'z'], plans)):
    ax = axes[0, col]
    ax.plot(rec_t, rec_ef[:, col],   '-b',  lw=1.5, label='EF actual')
    ax.plot(rec_t, rec_cup[:, col],  '--r', lw=1,   label='Cup')
    # planned (may have slightly different length, clip)
    n = min(len(rec_t), len(plan))
    plan_t = np.arange(n) * DT
    ax.plot(plan_t, plan[:n], ':g', lw=1, label='EF plan')
    # phase markers
    ax.axvline(attach_step * DT, color='m',      ls=':', alpha=.6, label='attach')
    ax.axvline(detach_step * DT, color='orange',  ls=':', alpha=.6, label='detach')
    ax.set_title(f'Position {label} (m)')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)

# ── Row 1, col 0: Joints 1-3 ──
for j in range(3):
    axes[1, 0].plot(rec_t, rec_j[:, j], label=f'j{j+1}')
axes[1, 0].set_title('Joints 1-3 (rad)')
axes[1, 0].set_xlabel('t (s)')
axes[1, 0].legend()

# ── Row 1, col 1: Joints 4-6 ──
for j in range(3, 6):
    axes[1, 1].plot(rec_t, rec_j[:, j], label=f'j{j+1}')
axes[1, 1].set_title('Joints 4-6 (rad)')
axes[1, 1].set_xlabel('t (s)')
axes[1, 1].legend()

# ── Row 1, col 2: Finger1 position ──
axes[1, 2].plot(rec_t, rec_fg, '-g', lw=1.5)
axes[1, 2].set_title('Finger1 motor position')
axes[1, 2].set_xlabel('t (s)')
axes[1, 2].set_ylabel('pos (m)')
ax12 = axes[1, 2]
ax12.axvline(attach_step * DT, color='m',     ls=':', label='attach')
ax12.axvline(detach_step * DT, color='orange', ls=':', label='detach')
if close_start is not None:
    ax12.axvline(close_start * DT, color='r', ls='--', alpha=.5, label='close cmd')
if open_start is not None:
    ax12.axvline(open_start * DT, color='b', ls='--', alpha=.5, label='open cmd')
ax12.legend(fontsize=7)

plt.tight_layout()
plt.savefig('plot_execution.png', dpi=120)
print("  Saved → plot_execution.png")
plt.show()


# ===================================================================
#  Rubric Plots  (matching Sagar reference structure)
# ===================================================================

# ── Figure R1: Cup Data (Position & Velocity) ──
figR1, axR1 = plt.subplots(2, 1, figsize=(12, 9))
figR1.suptitle('Rubric 1 — Cup Position & Velocity', fontsize=13)

for col, lbl, clr in [(0,'X','r'),(1,'Y','g'),(2,'Z','b')]:
    axR1[0].plot(rec_t, rec_cup[:, col], f'{clr}-', label=lbl)
axR1[0].set_ylabel('Position (m)'); axR1[0].set_title('Cup Position vs Time')
axR1[0].axvline(attach_step*DT, color='m', ls=':', alpha=.6, label='attach')
axR1[0].axvline(detach_step*DT, color='orange', ls=':', alpha=.6, label='detach')
axR1[0].legend(); axR1[0].grid(True, alpha=0.3)

for col, lbl, clr in [(0,'Vx','r'),(1,'Vy','g'),(2,'Vz','b')]:
    axR1[1].plot(rec_t, cup_vel[:, col], f'{clr}-', label=lbl)
axR1[1].set_xlabel('t (s)'); axR1[1].set_ylabel('Velocity (m/s)')
axR1[1].set_title('Cup Velocity vs Time')
axR1[1].legend(); axR1[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_rubric1_cup.png', dpi=120)
print("  Saved → plot_rubric1_cup.png")
plt.show()


# ── Figure R2: EF Linear Trajectory (Position + Velocity + 3D) ──
figR2 = plt.figure(figsize=(15, 14))
figR2.suptitle('Rubric 2.2 — End-Effector Linear Trajectory', fontsize=13)

axR2a = figR2.add_subplot(3, 1, 1)
for col, lbl, clr in [(0,'X','r'),(1,'Y','g'),(2,'Z','b')]:
    axR2a.plot(rec_t, rec_ef[:, col], f'{clr}-', label=lbl)
axR2a.set_ylabel('Position (m)'); axR2a.set_title('EF Position vs Time')
axR2a.legend(); axR2a.grid(True, alpha=0.3)

axR2b = figR2.add_subplot(3, 1, 2, projection='3d')
axR2b.plot(rec_ef[:,0], rec_ef[:,1], rec_ef[:,2], 'b-', lw=2, label='EF')
axR2b.plot(rec_cup[:,0], rec_cup[:,1], rec_cup[:,2], 'r--', alpha=0.5, label='Cup')
axR2b.scatter(*robot_base, c='r', marker='^', s=100, label='Base')
axR2b.set_xlabel('X'); axR2b.set_ylabel('Y'); axR2b.set_zlabel('Z')
axR2b.set_title('3D Trajectory'); axR2b.legend(fontsize=7)

axR2c = figR2.add_subplot(3, 1, 3)
for col, lbl, clr in [(0,'Vx','r'),(1,'Vy','g'),(2,'Vz','b')]:
    axR2c.plot(rec_t, ef_vel[:, col], f'{clr}-', label=lbl)
axR2c.plot(rec_t, ef_speed, 'k--', lw=1, alpha=0.6, label='|V|')
axR2c.set_xlabel('t (s)'); axR2c.set_ylabel('Velocity (m/s)')
axR2c.set_title('EF Linear Velocity vs Time')
axR2c.legend(); axR2c.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_rubric2_ef_linear.png', dpi=120)
print("  Saved → plot_rubric2_ef_linear.png")
plt.show()


# ── Figure R3: EF Angular Trajectory (Orientation + Angular Velocity) ──
figR3, axR3 = plt.subplots(2, 1, figsize=(12, 9))
figR3.suptitle('Rubric 2.4 — End-Effector Orientation & Angular Velocity', fontsize=13)

for col, lbl, clr in [(0,'Alpha (Roll)','r'),(1,'Beta (Pitch)','g'),(2,'Gamma (Yaw)','b')]:
    axR3[0].plot(rec_t, rec_ef_ori[:, col], f'{clr}-', label=lbl)
axR3[0].set_ylabel('Orientation (deg)'); axR3[0].set_title('EF Orientation vs Time')
axR3[0].legend(); axR3[0].grid(True, alpha=0.3)

for col, lbl, clr in [(0,'Alpha_dot','r'),(1,'Beta_dot','g'),(2,'Gamma_dot','b')]:
    axR3[1].plot(rec_t, ef_ang_vel[:, col], f'{clr}-', label=lbl)
axR3[1].set_xlabel('t (s)'); axR3[1].set_ylabel('Angular Velocity (deg/s)')
axR3[1].set_title('EF Angular Velocity vs Time')
axR3[1].legend(); axR3[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_rubric3_ef_angular.png', dpi=120)
print("  Saved → plot_rubric3_ef_angular.png")
plt.show()


# ── Figure R4: Joint Positions & Velocities ──
figR4, axR4 = plt.subplots(2, 1, figsize=(12, 9))
figR4.suptitle('Rubric 3.2 — Joint Positions & Velocities', fontsize=13)

for j in range(6):
    axR4[0].plot(rec_t, np.degrees(rec_j[:, j]), label=f'q{j+1}')
axR4[0].set_ylabel('Position (deg)'); axR4[0].set_title('Joint Positions vs Time')
axR4[0].legend(); axR4[0].grid(True, alpha=0.3)

for j in range(6):
    axR4[1].plot(rec_t, np.degrees(joint_vel[:, j]), label=f'q{j+1}_dot')
axR4[1].set_xlabel('t (s)'); axR4[1].set_ylabel('Velocity (deg/s)')
axR4[1].set_title('Joint Velocities vs Time')
axR4[1].legend(); axR4[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_rubric4_joints.png', dpi=120)
print("  Saved → plot_rubric4_joints.png")
plt.show()


# ── Figure R5: Gripper Velocity ──
figR5, axR5 = plt.subplots(2, 1, figsize=(12, 7))
figR5.suptitle('Rubric 4.2 — Gripper Data', fontsize=13)

n_gc = min(len(rec_t), len(gripper_cmd))
axR5[0].plot(np.arange(n_gc)*DT, gripper_cmd[:n_gc], 'k-', lw=1.5, label='Gripper cmd vel')
axR5[0].set_ylabel('Velocity cmd (m/s)'); axR5[0].set_title('Gripper Command')
for _t, _c, _l in [(attach_step*DT, 'm', 'attach'), (detach_step*DT, 'orange', 'detach')]:
    axR5[0].axvline(_t, color=_c, ls=':', alpha=.6, label=_l)
axR5[0].legend(); axR5[0].grid(True, alpha=0.3)

axR5[1].plot(rec_t, rec_fg, 'g-', lw=1.5, label='Finger position')
axR5[1].plot(rec_t, finger_vel, 'r--', lw=1, alpha=0.7, label='Finger velocity')
axR5[1].set_xlabel('t (s)'); axR5[1].set_ylabel('Position (m) / Velocity (m/s)')
axR5[1].set_title('Finger Motor Position & Velocity')
axR5[1].legend(); axR5[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_rubric5_gripper.png', dpi=120)
print("  Saved → plot_rubric5_gripper.png")
plt.show()

# ── XY top-down view ──
fig2, ax = plt.subplots(figsize=(8, 8))
ax.plot(rec_ef[:, 0],  rec_ef[:, 1],  'b-',  lw=2, label='EF path')
ax.plot(rec_cup[:, 0], rec_cup[:, 1], 'r--', alpha=0.5, label='Cup')
ax.plot(robot_base[0], robot_base[1], 'r^', ms=12, label='Robot base')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('XY top-down path')
ax.legend()
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('plot_xy.png', dpi=120)
print("  Saved → plot_xy.png")
plt.show()

# ── 3D path ──
fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot(rec_ef[:, 0], rec_ef[:, 1], rec_ef[:, 2], 'b-', lw=2, label='EF')
ax3.plot(rec_cup[:, 0], rec_cup[:, 1], rec_cup[:, 2], 'r--', alpha=0.5, label='Cup')
ax3.scatter(*robot_base, c='r', marker='^', s=100, label='Base')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('3D EF & Cup path')
ax3.legend()
plt.tight_layout()
plt.savefig('plot_3d.png', dpi=120)
print("  Saved → plot_3d.png")
plt.show()

print('\nDone.')
