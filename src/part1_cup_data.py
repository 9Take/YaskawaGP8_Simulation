"""
Part 1 : Collect cup trajectory + Select GRAB_STEP
============================================================
Professor's Step 1:  Get position AND velocity data of the cup

Collects  per step (dt=0.05 s):
  • Position    x, y, z
  • Velocity    vx, vy, vz
  • Orientation roll, pitch, yaw

Saves  →  cup_data.npz   (reused by Part 2 & 3)

Usage:  python part1_cup_data.py
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib.pyplot as plt
import math

# ── Connect ──
client = RemoteAPIClient()
sim    = client.require('sim')

DT            = 0.05
STEPS_COLLECT = 600          # 30 s — enough for 1+ full loop

# ── Handles ──
cup_h    = sim.getObject('/conveyorSystem/Cup')
ef_h     = sim.getObject('/yaskawa/gripperEF')
robot_h  = sim.getObject('/yaskawa')

# ══════════════════════════════════════════════════════════════
#  PHASE 1 : Collect cup trajectory  (robot stays at home)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 1 : Collect cup trajectory")
print("=" * 60)

sim.setStepping(True)
sim.startSimulation()

ef_home    = np.array(sim.getObjectPosition(ef_h, sim.handle_world))
robot_base = np.array(sim.getObjectPosition(robot_h, sim.handle_world))
print(f"  EF home:    {ef_home.round(4)}")
print(f"  Robot base: {robot_base.round(4)}")

try:
    conv_h    = sim.getObject('/conveyor')
    place_pos = np.array(sim.getObjectPosition(conv_h, sim.handle_world))
    place_ori = np.array(sim.getObjectOrientation(conv_h, sim.handle_world))
    print(f"  Place conv: pos={place_pos.round(4)}")
    print(f"              ori(deg)={np.degrees(place_ori).round(1)}")
except Exception:
    place_pos = np.array([0.455, 0.500, 0.400])
    place_ori = np.array([0.0, 0.0, 0.0])
    print(f"  Place conv: (default) {place_pos}")

# Position
cup_px, cup_py, cup_pz = [], [], []
# Velocity
cup_vx, cup_vy, cup_vz = [], [], []
# Orientation (Euler angles)
cup_ox, cup_oy, cup_oz = [], [], []

for i in range(STEPS_COLLECT):
    pos      = sim.getObjectPosition(cup_h, sim.handle_world)
    vel_l, _ = sim.getObjectVelocity(cup_h, sim.handle_world)
    ori      = sim.getObjectOrientation(cup_h, sim.handle_world)

    cup_px.append(pos[0]); cup_py.append(pos[1]); cup_pz.append(pos[2])
    cup_vx.append(vel_l[0]); cup_vy.append(vel_l[1]); cup_vz.append(vel_l[2])
    cup_ox.append(ori[0]); cup_oy.append(ori[1]); cup_oz.append(ori[2])
    sim.step()

sim.setStepping(False)
sim.stopSimulation()

cup_px = np.array(cup_px); cup_py = np.array(cup_py); cup_pz = np.array(cup_pz)
cup_vx = np.array(cup_vx); cup_vy = np.array(cup_vy); cup_vz = np.array(cup_vz)
cup_ox = np.array(cup_ox); cup_oy = np.array(cup_oy); cup_oz = np.array(cup_oz)
cup_speed = np.sqrt(cup_vx**2 + cup_vy**2 + cup_vz**2)

print(f"  Collected {STEPS_COLLECT} steps  cup[0]=({cup_px[0]:.3f},{cup_py[0]:.3f},{cup_pz[0]:.3f})")
print(f"  Peak speed: {np.max(cup_speed):.3f} m/s")


# ══════════════════════════════════════════════════════════════
#  PHASE 2 : Select grab step
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 2 : Select grab step")
print("=" * 60)

dist_base = np.sqrt((cup_px - robot_base[0])**2 + (cup_py - robot_base[1])**2)

# -- Auto-recommend GRAB_STEP (closest approach within arm reach) --
REACH_MAX = 0.65
reachable = np.where(dist_base < REACH_MAX)[0]
if len(reachable) > 0:
    best_idx = int(reachable[np.argmin(dist_base[reachable])])
    print(f"\n  ★ Recommended GRAB_STEP = {best_idx}  "
          f"(dist={dist_base[best_idx]:.3f}m, min within reach)")
    print(f"    Reachable window: step {reachable[0]} – {reachable[-1]}  "
          f"({reachable[0]*DT:.1f}s – {reachable[-1]*DT:.1f}s)")
    print(f"    Cup at {best_idx}: ({cup_px[best_idx]:.3f}, {cup_py[best_idx]:.3f}, {cup_pz[best_idx]:.3f})")
else:
    best_idx = int(np.argmin(dist_base))
    print(f"\n  ⚠ No step within {REACH_MAX}m reach! "
          f"Closest: step {best_idx} (dist={dist_base[best_idx]:.3f}m)")

print(f"\n  {'Step':>5}  {'t(s)':>5}  {'cup_x':>7}  {'cup_y':>7}  {'cup_z':>7}"
      f"  {'dist_xy':>7}  {'|vel|':>6}")
print("  " + "-" * 55)
for i in range(0, STEPS_COLLECT, 10):
    print(f"  {i:>5}  {i*DT:>5.1f}  {cup_px[i]:>7.3f}  {cup_py[i]:>7.3f}"
          f"  {cup_pz[i]:>7.3f}  {dist_base[i]:>7.3f}  {cup_speed[i]:>6.3f}")

# ── Plots ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Cup Trajectory Data — pick GRAB_STEP then close window', fontsize=13)

# XY path
sc = axes[0, 0].scatter(cup_px, cup_py, c=range(STEPS_COLLECT), cmap='viridis', s=10)
plt.colorbar(sc, ax=axes[0, 0], label='step')
axes[0, 0].plot(robot_base[0], robot_base[1], 'r^', ms=12, label='robot base')
axes[0, 0].plot(ef_home[0],    ef_home[1],    'bs', ms=10, label='EF home')
axes[0, 0].plot(place_pos[0],  place_pos[1],  'g*', ms=14, label='place target')
th = np.linspace(0, 2*math.pi, 200)
for r in [0.55, 0.65, 0.85]:
    axes[0, 0].plot(robot_base[0]+r*np.cos(th), robot_base[1]+r*np.sin(th),
                    '--', alpha=0.3, label=f'r={r}m')
for i in range(0, STEPS_COLLECT, 50):
    axes[0, 0].annotate(str(i), (cup_px[i], cup_py[i]), fontsize=7, color='red')
axes[0, 0].set_xlabel('x (m)'); axes[0, 0].set_ylabel('y (m)')
axes[0, 0].set_title('Cup XY path'); axes[0, 0].legend(fontsize=6); axes[0, 0].set_aspect('equal')

# Distance & speed
axes[0, 1].plot(range(STEPS_COLLECT), dist_base, 'b-',  label='dist_base_xy (m)')
axes[0, 1].plot(range(STEPS_COLLECT), cup_speed, 'r--', label='|vel| (m/s)')
axes[0, 1].plot(range(STEPS_COLLECT), cup_pz,    'g:',  label='cup_z (m)')
axes[0, 1].axhline(0.65, color='purple', alpha=0.4, ls='--', label='reach ~0.65m')
axes[0, 1].set_xlabel('step'); axes[0, 1].set_ylabel('value')
axes[0, 1].set_title('Distance / Speed / Z vs step'); axes[0, 1].legend(fontsize=7)

# Cup position (x,y,z)
t_arr = np.arange(STEPS_COLLECT) * DT
axes[0, 2].plot(t_arr, cup_px, label='x')
axes[0, 2].plot(t_arr, cup_py, label='y')
axes[0, 2].plot(t_arr, cup_pz, label='z')
axes[0, 2].set_xlabel('t (s)'); axes[0, 2].set_ylabel('m')
axes[0, 2].set_title('Cup position (x,y,z)'); axes[0, 2].legend()

# Cup velocity (vx,vy,vz)
axes[1, 0].plot(t_arr, cup_vx, label='vx')
axes[1, 0].plot(t_arr, cup_vy, label='vy')
axes[1, 0].plot(t_arr, cup_vz, label='vz')
axes[1, 0].set_xlabel('t (s)'); axes[1, 0].set_ylabel('m/s')
axes[1, 0].set_title('Cup velocity (vx,vy,vz)'); axes[1, 0].legend()

# Cup orientation
axes[1, 1].plot(t_arr, np.degrees(cup_ox), label='roll')
axes[1, 1].plot(t_arr, np.degrees(cup_oy), label='pitch')
axes[1, 1].plot(t_arr, np.degrees(cup_oz), label='yaw')
axes[1, 1].set_xlabel('t (s)'); axes[1, 1].set_ylabel('deg')
axes[1, 1].set_title('Cup orientation (roll,pitch,yaw)'); axes[1, 1].legend()

# Cup speed
axes[1, 2].plot(t_arr, cup_speed, 'k-', lw=1.5)
axes[1, 2].set_xlabel('t (s)'); axes[1, 2].set_ylabel('m/s')
axes[1, 2].set_title('Cup speed |v|')

plt.tight_layout(); plt.show()

# ── User input ──
MIN_GRAB, MAX_GRAB = 80, STEPS_COLLECT - 120
while True:
    try:
        raw = input(f"\n  >>> Enter GRAB_STEP ({MIN_GRAB}–{MAX_GRAB}): ").strip()
        GRAB_STEP = int(raw)
        if MIN_GRAB <= GRAB_STEP <= MAX_GRAB:
            break
        print(f"  Must be between {MIN_GRAB} and {MAX_GRAB}.")
    except ValueError:
        print("  Enter an integer.")

print(f"\n  GRAB_STEP = {GRAB_STEP}")
print(f"  Cup at grab: ({cup_px[GRAB_STEP]:.3f}, {cup_py[GRAB_STEP]:.3f}, {cup_pz[GRAB_STEP]:.3f})")
print(f"  Cup vel at grab: ({cup_vx[GRAB_STEP]:.4f}, {cup_vy[GRAB_STEP]:.4f}, {cup_vz[GRAB_STEP]:.4f})")

# ── Save ──
out = 'cup_data.npz'
np.savez(out,
         cup_px=cup_px, cup_py=cup_py, cup_pz=cup_pz,
         cup_vx=cup_vx, cup_vy=cup_vy, cup_vz=cup_vz,
         cup_ox=cup_ox, cup_oy=cup_oy, cup_oz=cup_oz,
         cup_speed=cup_speed,
         ef_home=ef_home, robot_base=robot_base,
         place_pos=place_pos, place_ori=place_ori,
         GRAB_STEP=GRAB_STEP, DT=DT, STEPS_COLLECT=STEPS_COLLECT)
print(f"\n  Saved → {out}")
print("  Next: python part2_ik.py")
