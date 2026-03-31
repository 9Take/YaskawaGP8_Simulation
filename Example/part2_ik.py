"""
Part 2 : EF Trajectory Design  +  Jacobian Velocity IK
=========================================================
Professor's methodology  (classProjectTutorial_2026.pdf):
  Step 1 -> Cup trajectory data           (loaded from part1)
  Step 2 -> Design robot EF trajectory    (time-synchronized with cup)
  Step 3 -> Jacobian IK:  X_dot -> q_dot = J_pinv * V  ->  q = q + q_dot*dt

Trajectory phases (step index == simulation time index == cup data index):
  0  WAIT      -- EF at home,            cup approaches
  1  APPROACH  -- S-curve home -> above cup path
  2  DESCEND   -- blend to cup x,y  +  S-curve z down
  3  GRAB      -- track cup exactly,  close gripper
  4  LIFT      -- move up
  5  TRANSPORT -- fly to place conveyor
  6  PLACE     -- descend,  open gripper at end
  7  RETURN    -- retreat up  ->  home

Output  ->  trajectory.npz   (compatible with part3_run.py)
Usage   :  python part2_ik.py
"""

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R, Slerp

# ===================================================================
#  TUNABLE PARAMETERS
# ===================================================================

# -- EF z relative to cup z (0 = EF at cup center height)
EF_Z_OFFSET      = 0.0

# -- Height above cup for pre-grab hover  (m)
PRE_GRAB_HEIGHT  = 0.12

# -- Steps per trajectory phase
APPROACH_STEPS   = 120     # home -> pre-grab           (6.0 s)
DESCEND_STEPS    = 15      # pre-grab -> grab level     (0.75 s)
GRAB_STEPS       = 20      # track cup + close gripper  (1.0 s)
LIFT_STEPS       = 50      # lift after grab
TRANSPORT_STEPS  = 90      # fly to place conveyor  (was 60 – more time for big swing)
PLACE_STEPS      = 30      # descend to place target
RETURN_STEPS     = 50      # retreat + home           (was 40)

# -- Heights (m)
LIFT_HEIGHT      = 0.20    # moderate lift — 40cm exceeded GP8 max reach
PLACE_ABOVE_Z    = 0.05    # vertical hover height above place target  (was 0.15)
PLACE_TARGET_Z   = 0.12    # cup land height above place conveyor center  (was 0.05)

# -- Gripper velocity commands (m/s for finger motors)
#    Sagar (Previous_Year65) used ±0.15 — gentler avoids knocking the cup.
GRIP_OPEN  =  0.20
GRIP_CLOSE = -0.20
GRIP_HOLD  = -0.04
GRIP_IDLE  =  0.0

# -- Jacobian IK
Kp          = 10.0     # position-feedback gain
Kp_ori      = 8.0      # orientation feedback gain
W_ORI       = 0.8      # weight for orientation vs position (0=ignore, 1=equal)
DELTA_Q     = 1e-4     # perturbation for numerical Jacobian
LAMBDA_BASE = 0.005    # damping for DLS pseudoinverse (was 1e-3, raised to avoid singularity blowup)
QD_MAX      = 3.5      # max joint velocity (rad/s) -- prevents singularity explosions

# -- Joint limits (rad) -- Yaskawa GP8  (from CoppeliaSim model properties)
Q_MIN = np.array([-2.97, -1.75, -3.14, -3.49, -2.09, -6.28])
Q_MAX = np.array([ 2.97,  2.62,  1.22,  3.49,  2.09,  6.28])

# -- GP8 DH link lengths (from Sagar/fwk_solver.py, confirmed against GP8 spec) --
#    d1=0.33002  a1=0.01867  a2=0.04  a3=0.345  d4=0.34  d7=0.241
#    Max reachable radius ≈ 0.727 m  (from Yaskawa datasheet)
GP8_MAX_REACH = 0.727


# ===================================================================
#  0.  CONNECT TO COPPELIASIM  (scene query — reused in section 4)
# ===================================================================
client = RemoteAPIClient()
sim    = client.require('sim')


# ===================================================================
#  1.  LOAD CUP DATA
# ===================================================================
print("\n" + "=" * 60)
print("  1.  LOAD CUP DATA")
print("=" * 60)

d = np.load('cup_data.npz', allow_pickle=True)

cup_px = np.asarray(d['cup_px'], dtype=float)
cup_py = np.asarray(d['cup_py'], dtype=float)
cup_pz = np.asarray(d['cup_pz'], dtype=float)

ef_home       = np.asarray(d['ef_home'],    dtype=float)
robot_base    = np.asarray(d['robot_base'],  dtype=float)
place_pos     = np.asarray(d['place_pos'],   dtype=float)
GRAB_STEP     = int(d['GRAB_STEP'])
DT            = float(d['DT'])
STEPS_COLLECT = int(d['STEPS_COLLECT'])

# Cup velocities -- load or derive from positions
try:
    cup_vx = np.asarray(d['cup_vx'], dtype=float)
    cup_vy = np.asarray(d['cup_vy'], dtype=float)
    cup_vz = np.asarray(d['cup_vz'], dtype=float)
    print("  Cup velocities loaded from npz")
except KeyError:
    cup_vx = np.gradient(cup_px, DT)
    cup_vy = np.gradient(cup_py, DT)
    cup_vz = np.gradient(cup_pz, DT)
    print("  Cup velocities computed from positions (np.gradient)")

cup_speed = np.sqrt(cup_vx**2 + cup_vy**2 + cup_vz**2)

print(f"  GRAB_STEP      = {GRAB_STEP}  ({GRAB_STEP * DT:.1f} s)")
print(f"  Cup @grab      = ({cup_px[GRAB_STEP]:.4f}, "
      f"{cup_py[GRAB_STEP]:.4f}, {cup_pz[GRAB_STEP]:.4f})")
print(f"  Cup vel @grab  = ({cup_vx[GRAB_STEP]:.4f}, "
      f"{cup_vy[GRAB_STEP]:.4f}, {cup_vz[GRAB_STEP]:.4f})  "
      f"|v|={cup_speed[GRAB_STEP]:.4f} m/s")
print(f"  EF home        = {ef_home.round(4)}")
print(f"  Place pos      = {place_pos.round(4)}")

# ── Cup orientation (Euler XYZ from CoppeliaSim getObjectOrientation) ──
# CoppeliaSim uses EXTRINSIC XYZ convention (= scipy 'XYZ', uppercase)
cup_euler_xyz = np.column_stack([
    np.asarray(d['cup_ox'], dtype=float),
    np.asarray(d['cup_oy'], dtype=float),
    np.asarray(d['cup_oz'], dtype=float),
])                                                  # (STEPS_COLLECT, 3)
cup_rots   = R.from_euler('XYZ', cup_euler_xyz)     # Rotation array (extrinsic XYZ)
cup_z_axes = cup_rots.apply([0, 0, 1])              # cup local-Z in world (N,3)

# ── Destination conveyor '/conveyor' — query world-frame position ──
# The CoppeliaSim client is connected from section 0 above.
try:
    _conv_h = sim.getObject('/conveyor')
    _conv_world_pos = np.array(sim.getObjectPosition(_conv_h, sim.handle_world))
    _conv_euler = np.array(sim.getObjectOrientation(_conv_h, sim.handle_world))
    R_conv = R.from_euler('XYZ', _conv_euler)   # extrinsic XYZ = CoppeliaSim convention
    conv_normal = R_conv.apply([0, 0, 1])          # conveyor surface normal in world
    print(f"  /conveyor world pos  = {_conv_world_pos.round(4)}")
    print(f"  /conveyor euler(xyz) = {np.degrees(_conv_euler).round(1)}°")
    print(f"  /conveyor normal     = {conv_normal.round(4)}")
    # Use conveyor world position directly (Z = belt surface height)
    place_pos = _conv_world_pos.copy()
    print(f"  Place pos (world)    = {place_pos.round(4)}")
except Exception as e:
    print(f"  ⚠ Could not query /conveyor: {e}")
    print(f"  Using place_pos from npz: {place_pos.round(4)}")
    conv_normal = np.array([0.0, 0.0, 1.0])        # default: flat surface

# Diagnostic: cup tilt at GRAB_STEP
cup_z_grab = cup_z_axes[GRAB_STEP]
tilt_deg   = np.degrees(np.arccos(np.clip(cup_z_grab[2], -1, 1)))
print(f"  Cup Z @grab    = {cup_z_grab.round(4)}")
print(f"  Cup tilt @grab = {tilt_deg:.1f}° from vertical")

# 180° flip around local-X  (used in Step 2 to get EF target from cup rot)
R_flip = R.from_euler('x', np.pi)

    
# ===================================================================
#  2.  DESIGN EF TRAJECTORY
# ===================================================================
print("\n" + "=" * 60)
print("  2.  DESIGN EF TRAJECTORY")
print("=" * 60)

# -- Phase boundaries (absolute step indices) --
descend_start  = GRAB_STEP - DESCEND_STEPS
approach_start = descend_start - APPROACH_STEPS
if approach_start < 0:
    APPROACH_STEPS = descend_start
    approach_start = 0
    print(f"  >> APPROACH_STEPS trimmed to {APPROACH_STEPS}")

wait_end        = approach_start
grab_end        = GRAB_STEP + GRAB_STEPS
lift_end        = grab_end   + LIFT_STEPS
transport_end   = lift_end   + TRANSPORT_STEPS
place_end       = transport_end + PLACE_STEPS
STEPS           = place_end  + RETURN_STEPS

# Ensure cup data covers the grab window
need_idx = GRAB_STEP + GRAB_STEPS - 1
if need_idx >= STEPS_COLLECT:
    GRAB_STEPS = STEPS_COLLECT - GRAB_STEP
    grab_end   = GRAB_STEP + GRAB_STEPS
    lift_end   = grab_end  + LIFT_STEPS
    transport_end = lift_end + TRANSPORT_STEPS
    place_end  = transport_end + PLACE_STEPS
    STEPS      = place_end + RETURN_STEPS
    print(f"  >> GRAB_STEPS reduced to {GRAB_STEPS} (cup data limit)")

print(f"  Phase 0  WAIT      :  0       -> {wait_end:4d}   ({wait_end} steps,  {wait_end*DT:.1f}s)")
print(f"  Phase 1  APPROACH  :  {approach_start:4d} -> {descend_start:4d}   ({APPROACH_STEPS} steps)")
print(f"  Phase 2  DESCEND   :  {descend_start:4d} -> {GRAB_STEP:4d}   ({DESCEND_STEPS} steps)")
print(f"  Phase 3  GRAB      :  {GRAB_STEP:4d} -> {grab_end:4d}   ({GRAB_STEPS} steps)")
print(f"  Phase 4  LIFT      :  {grab_end:4d} -> {lift_end:4d}   ({LIFT_STEPS} steps)")
print(f"  Phase 5  TRANSPORT :  {lift_end:4d} -> {transport_end:4d}   ({TRANSPORT_STEPS} steps)")
print(f"  Phase 6  PLACE     :  {transport_end:4d} -> {place_end:4d}   ({PLACE_STEPS} steps)")
print(f"  Phase 7  RETURN    :  {place_end:4d} -> {STEPS:4d}   ({RETURN_STEPS} steps)")
print(f"  TOTAL              :  {STEPS} steps  ({STEPS * DT:.1f} s)")


# -- S-curve helper (cosine 0-to-1, zero velocity at endpoints) --
def scurve(a):
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(a, 0.0, 1.0)))


# -- Horizontal Side-Grasp via Yaw + Roll + optional Pitch  -----------------
#
# GP8 home:  EF  Z ≈ [1, 0, 0]  (approach axis, pointing outward)
#                +X ≈ [0, 0, 1]  (gripper UP direction)
#                 Y ≈ [0,-1, 0]  (finger opening direction)
#
# Strategy:
#   1. YAW around world-Z to face target → R_face
#   2. ROLL around local-Z  → align gripper UP with the surface normal's
#      component PERPENDICULAR to the approach axis  (~12° cup, ~3.7° conv)
#   3. Optional PITCH around local-Y → tilt approach axis to capture the
#      normal's component ALONG the approach axis.
#
# IMPORTANT: Pitch is DISABLED for GRAB (max_pitch=0) because at 81% reach
# even 10° pitch causes 0.14m position errors.  Pitch is ENABLED for PLACE
# (max_pitch=10°) to match the conveyor tilt more accurately.
# During TRANSPORT the Slerp from R_grab (no pitch) to R_place (with pitch)
# introduces the pitch gradually when the robot has more kinematic freedom.
# --------------------------------------------------------------------------

# Design-time home quaternion (section 4 reads the real one — virtually identical)
_ef_h_temp = sim.getObject('/yaskawa/gripperEF')
_quat_actual = sim.getObjectQuaternion(_ef_h_temp, sim.handle_world)
_R_home_approx = R.from_quat(_quat_actual)

def _side_grasp_frame(R_home_ref, target_pos, robot_base, surface_normal, max_pitch=0.0):
    # 1. Ensure normal is pointing up
    n = surface_normal.copy()
    if n[2] < 0: n = -n
    n = n / np.linalg.norm(n)
    
    # 2. Get Gripper's default UP direction (Local X-axis)
    home_x = R_home_ref.apply([1, 0, 0])
    
    # 3. Calculate minimal rotation to perfectly align Gripper UP with Conveyor Normal
    cross_p = np.cross(home_x, n)
    norm_c = np.linalg.norm(cross_p)
    
    if norm_c > 1e-6:
        axis = cross_p / norm_c
        angle = np.arccos(np.clip(np.dot(home_x, n), -1.0, 1.0))
        R_align = R.from_rotvec(axis * angle)
        R_final = R_align * R_home_ref
    else:
        R_final = R_home_ref
        
    z_horizontal = R_final.apply([0, 0, 1])
    return R_final, z_horizontal, 0.0, 0.0, 0.0


# -- GRAB orientation --------------------------------------------------------
cup_pos_grab = np.array([cup_px[GRAB_STEP], cup_py[GRAB_STEP], cup_pz[GRAB_STEP]])

cup_R_grab = R.from_euler('XYZ', cup_euler_xyz[GRAB_STEP])
n_cup = cup_R_grab.apply([0, 0, 1])
if n_cup[2] < 0:
    n_cup = -n_cup

R_grab, z_horiz_grab, yaw_grab, roll_grab, pitch_grab = _side_grasp_frame(
    _R_home_approx, cup_pos_grab, robot_base, n_cup, max_pitch=0)  # NO pitch at grab
z_ee_grab = z_horiz_grab              # HORIZONTAL approach axis for position planning
x_ee_grab = R_grab.apply([1, 0, 0])   # gripper UP (roll only, no pitch)

# Positions: approach from behind along the HORIZONTAL approach axis
grab_pos  = cup_pos_grab.copy()
pre_grab  = grab_pos - z_ee_grab * 0.15          # 15 cm behind cup


# -- PLACE orientation -------------------------------------------------------
pos_place_target = np.array([0.45452, 0.500, 0.462])   # belt(0.4) + 3cm cup height

conv_R_full = R.from_euler('XYZ', [-15, -10, -180], degrees=True)   # extrinsic XYZ
n_conv = conv_R_full.apply([0, 0, 1])
if n_conv[2] < 0:
    n_conv = -n_conv

R_place, z_horiz_place, yaw_place, roll_place, pitch_place = _side_grasp_frame(
    _R_home_approx, pos_place_target, robot_base, n_conv, max_pitch=np.radians(20))  # pitch for conveyor
z_ee_place = z_horiz_place

# -- Cup-gripper offset compensation for PLACE --
# At grab, the gripper doesn't perfectly match cup tilt (only roll, no pitch).
# The cup is held with a small angular offset in gripper frame.
# Compensate at PLACE so the CUP normal (not gripper) matches conveyor normal.
cup_z_in_grip = R_grab.inv().apply(n_cup)   # cup normal in gripper-local frame
cup_z_in_grip /= np.linalg.norm(cup_z_in_grip)

# R_corr rotates cup_z_in_grip -> [1,0,0] in gripper frame
_a = cup_z_in_grip
_b = np.array([1.0, 0.0, 0.0])
_cross = np.cross(_a, _b)
if np.linalg.norm(_cross) > 1e-10:
    _axis = _cross / np.linalg.norm(_cross)
    _angle = np.arccos(np.clip(np.dot(_a, _b), -1, 1))
    R_corr = R.from_rotvec(_axis * _angle)
else:
    R_corr = R.identity()
    _angle = 0.0

R_place = R_place * R_corr   # COMPENSATED: cup normal will match convoy normal
x_ee_place = R_place.apply([1, 0, 0])   # gripper UP (slightly different from conv normal)

print(f"  Cup-gripper offset = {np.degrees(_angle):.1f} deg  (compensated at PLACE)")

place_target = pos_place_target.copy()
# Pull back horizontally along approach axis (NOT straight up in +Z)
# to avoid the lower finger uppercutting the cup on retreat.
z_approach_place = np.array([0, 0, -1])  # Fake approach
# Force safe hover directly straight UP over the place target
place_above = place_target + np.array([0.0, 0.0, 0.15])

# Diagnostic: tilt angle = angle between x_ee (gripper UP) and [0,0,1]
tilt_grab_deg  = np.degrees(np.arccos(np.clip(np.dot(x_ee_grab, [0,0,1]), -1, 1)))
tilt_place_deg = np.degrees(np.arccos(np.clip(np.dot(x_ee_place, [0,0,1]), -1, 1)))

print(f"\n  -- Horizontal Side-Grasp (Yaw + Roll; Pitch only at PLACE) --")
print(f"  yaw_grab  = {np.degrees(yaw_grab):.1f}°   roll_grab  = {np.degrees(roll_grab):.1f}°   pitch_grab  = {np.degrees(pitch_grab):.1f}° (no pitch)")
print(f"  yaw_place = {np.degrees(yaw_place):.1f}°   roll_place = {np.degrees(roll_place):.1f}°   pitch_place = {np.degrees(pitch_place):.1f}°")
print(f"  x_ee_grab (UP) = {x_ee_grab.round(4)}  tilt from vert = {tilt_grab_deg:.1f}°  (cup={tilt_deg:.1f}°)")
print(f"  x_ee_place(UP) = {x_ee_place.round(4)}  tilt from vert = {tilt_place_deg:.1f}°")
print(f"  R_grab euler   = {np.degrees(R_grab.as_euler('XYZ')).round(1)}°")
print(f"  R_place euler  = {np.degrees(R_place.as_euler('XYZ')).round(1)}°")
print(f"  z_ee_grab      = {z_ee_grab.round(4)}  (approach, horizontal for pos planning)")
print(f"  z_ee_place     = {z_ee_place.round(4)}  (approach, horizontal for pos planning)")
print(f"  n_cup          = {n_cup.round(4)}")
print(f"  n_conv         = {n_conv.round(4)}")
print(f"  Pre-grab       = {pre_grab.round(4)}")
print(f"  Grab pos       = {grab_pos.round(4)}")
print(f"  Place target   = {place_target.round(4)}")
print(f"  Place above    = {place_above.round(4)}")

# -- Workspace reachability check (from GP8 DH params) --
def _check_reach(label, pt):
    """Warn if a waypoint is near or beyond GP8 max reach."""
    dist = np.sqrt(pt[0]**2 + pt[1]**2 + (pt[2] - 0.33)**2)   # distance from J1 axis
    pct  = dist / GP8_MAX_REACH * 100
    flag = '*** OUT OF REACH ***' if dist > GP8_MAX_REACH else ''
    print(f"  Reach {label:16s}: {dist:.3f}m  ({pct:.0f}% of max)  {flag}")
    return dist <= GP8_MAX_REACH

print()
for _lbl, _pt in [('pre_grab', pre_grab), ('grab_pos', grab_pos),
                   ('place_target', place_target), ('place_above', place_above),
                   ('ef_home', ef_home)]:
    _check_reach(_lbl, _pt)

# -- Allocate trajectory arrays --
ef_x = np.zeros(STEPS)
ef_y = np.zeros(STEPS)
ef_z = np.zeros(STEPS)
gripper_cmd = np.zeros(STEPS)


# ------------ Phase 0  WAIT ------------
for k in range(wait_end):
    ef_x[k], ef_y[k], ef_z[k] = ef_home
    gripper_cmd[k] = GRIP_OPEN

# ------------ Phase 1  APPROACH (S-curve home -> pre_grab) ------------
for k in range(APPROACH_STEPS):
    step = approach_start + k
    s    = scurve(k / max(APPROACH_STEPS - 1, 1))
    ef_x[step] = ef_home[0] + s * (pre_grab[0] - ef_home[0])
    ef_y[step] = ef_home[1] + s * (pre_grab[1] - ef_home[1])
    ef_z[step] = ef_home[2] + s * (pre_grab[2] - ef_home[2])
    gripper_cmd[step] = GRIP_OPEN

# ------------ Phase 2  DESCEND  (horizontal slide-in along approach axis) ----
# Slide from pre_grab (15cm behind) INTO the cup,
# while tracking the cup's XY movement on the carousel.
EARLY_CLOSE = 5
for k in range(DESCEND_STEPS):
    step = descend_start + k
    s    = scurve(k / max(DESCEND_STEPS - 1, 1))
    # Cup position at this step
    cup_pos_k = np.array([cup_px[step], cup_py[step], cup_pz[step]])
    # Approach axis for current cup position (HORIZONTAL for position planning)
    R_k, z_k_h, _, _, _ = _side_grasp_frame(_R_home_approx, cup_pos_k, robot_base,
                                R.from_euler('XYZ', cup_euler_xyz[step]).apply([0,0,1]),
                                max_pitch=0)  # NO pitch during DESCEND
    z_k = z_k_h   # horizontal approach direction
    pre_k = cup_pos_k - z_k * 0.15
    # Blend from pre_grab toward cup center
    ef_x[step] = pre_k[0] * (1.0 - s) + cup_pos_k[0] * s
    ef_y[step] = pre_k[1] * (1.0 - s) + cup_pos_k[1] * s
    ef_z[step] = pre_k[2] * (1.0 - s) + cup_pos_k[2] * s
    gripper_cmd[step] = GRIP_CLOSE if k >= DESCEND_STEPS - EARLY_CLOSE else GRIP_OPEN

# ------------ Phase 3  GRAB (track cup center, close gripper) ----------------
for k in range(GRAB_STEPS):
    step = GRAB_STEP + k
    idx  = min(step, STEPS_COLLECT - 1)
    ef_x[step] = cup_px[idx]
    ef_y[step] = cup_py[idx]
    ef_z[step] = cup_pz[idx]
    gripper_cmd[step] = GRIP_CLOSE

# ------------ Phase 4  LIFT  (straight UP in global Z) -----
lift_s = np.array([ef_x[grab_end - 1], ef_y[grab_end - 1], ef_z[grab_end - 1]])
lift_e = lift_s + np.array([0.0, 0.0, LIFT_HEIGHT])   # global Z up
lift_e_pos = lift_e.copy()  # update lift_e_pos with actual IK start point
for k in range(LIFT_STEPS):
    step = grab_end + k
    s    = scurve(k / max(LIFT_STEPS - 1, 1))
    ef_x[step] = lift_s[0] + s * (lift_e[0] - lift_s[0])
    ef_y[step] = lift_s[1] + s * (lift_e[1] - lift_s[1])
    ef_z[step] = lift_s[2] + s * (lift_e[2] - lift_s[2])
    gripper_cmd[step] = GRIP_HOLD

# ------------ Phase 5  TRANSPORT (via-point arch: lift_e -> via -> place_above) ----
# Via-point: X outward for elbow room, Y midway between lift and place for a
# smoother arc (was fixed at Y=0 which forced a huge lateral jump in seg B).
via_x = 0.45
via_y = (lift_e[1] + place_above[1]) / 2.0          # midway Y
via_z = max(lift_e[2], place_above[2]) + 0.05
pos_via = np.array([via_x, via_y, via_z])
print(f"  Via-point      = {pos_via.round(4)}")
_check_reach('lift_e', lift_e)
_check_reach('via_point', pos_via)

seg_A_steps = TRANSPORT_STEPS // 2
seg_B_steps = TRANSPORT_STEPS - seg_A_steps

# Segment A: lift_e -> pos_via (first half)
for k in range(seg_A_steps):
    step = lift_end + k
    s    = scurve(k / max(seg_A_steps - 1, 1))
    ef_x[step] = lift_e[0] + s * (pos_via[0] - lift_e[0])
    ef_y[step] = lift_e[1] + s * (pos_via[1] - lift_e[1])
    ef_z[step] = lift_e[2] + s * (pos_via[2] - lift_e[2])
    gripper_cmd[step] = GRIP_HOLD

# Segment B: pos_via -> place_above (second half)
seg_B_start = lift_end + seg_A_steps
for k in range(seg_B_steps):
    step = seg_B_start + k
    s    = scurve(k / max(seg_B_steps - 1, 1))
    ef_x[step] = pos_via[0] + s * (place_above[0] - pos_via[0])
    ef_y[step] = pos_via[1] + s * (place_above[1] - pos_via[1])
    ef_z[step] = pos_via[2] + s * (place_above[2] - pos_via[2])
    gripper_cmd[step] = GRIP_HOLD

# ------------ Phase 6  PLACE (descend, open gripper at end) ------------
for k in range(PLACE_STEPS):
    step = transport_end + k
    s    = scurve(k / max(PLACE_STEPS - 1, 1))
    ef_x[step] = place_above[0] + s * (place_target[0] - place_above[0])
    ef_y[step] = place_above[1] + s * (place_target[1] - place_above[1])
    ef_z[step] = place_above[2] + s * (place_target[2] - place_above[2])
    gripper_cmd[step] = GRIP_OPEN if k >= PLACE_STEPS - 5 else GRIP_HOLD

# ------------ Phase 7  RETURN (retreat → safe via → home) ------------------
# ef_home is at 103% of GP8 max reach, so flying directly causes large IK
# errors.  Route through a safe via-point well within the workspace.
retreat_n   = min(15, RETURN_STEPS // 4)
via_ret_n   = (RETURN_STEPS - retreat_n) // 2
home_n      = RETURN_STEPS - retreat_n - via_ret_n

# Safe via-point for return (pull arm inward, keep Z moderate)
ret_via = np.array([0.45452, 0.0, 0.60])   # well within 83% reach

# 7a  retreat up to place_above
for k in range(retreat_n):
    step = place_end + k
    s    = scurve(k / max(retreat_n - 1, 1))
    ef_x[step] = place_target[0] + s * (place_above[0] - place_target[0])
    ef_y[step] = place_target[1] + s * (place_above[1] - place_target[1])
    ef_z[step] = place_target[2] + s * (place_above[2] - place_target[2])
    gripper_cmd[step] = GRIP_IDLE

# 7b  fly from place_above to safe via-point
ret_seg_b_start = place_end + retreat_n
for k in range(via_ret_n):
    step = ret_seg_b_start + k
    s    = scurve(k / max(via_ret_n - 1, 1))
    ef_x[step] = place_above[0] + s * (ret_via[0] - place_above[0])
    ef_y[step] = place_above[1] + s * (ret_via[1] - place_above[1])
    ef_z[step] = place_above[2] + s * (ret_via[2] - place_above[2])
    gripper_cmd[step] = GRIP_IDLE

# 7c  fly from safe via-point to home
ret_seg_c_start = ret_seg_b_start + via_ret_n
for k in range(home_n):
    step = ret_seg_c_start + k
    s    = scurve(k / max(home_n - 1, 1))
    ef_x[step] = ret_via[0] + s * (ef_home[0] - ret_via[0])
    ef_y[step] = ret_via[1] + s * (ef_home[1] - ret_via[1])
    ef_z[step] = ret_via[2] + s * (ef_home[2] - ret_via[2])
    gripper_cmd[step] = GRIP_IDLE

print("\n  >> EF trajectory designed")


# ===================================================================
#  3.  PLOT -- EF TRAJECTORY vs CUP  (pre-IK verification)
# ===================================================================
t_traj = np.arange(STEPS) * DT
t_cup  = np.arange(STEPS_COLLECT) * DT

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('EF Trajectory Design -- verify timing before IK', fontsize=13)

for ci, (lab, ef_arr, cup_arr) in enumerate([
        ('X (m)', ef_x, cup_px), ('Y (m)', ef_y, cup_py), ('Z (m)', ef_z, cup_pz)]):
    ax = axes.flat[ci]
    ax.plot(t_cup,  cup_arr, 'r-',  alpha=0.5, label=f'Cup {lab[0].lower()}')
    ax.plot(t_traj, ef_arr,  'b-',  lw=1.5,   label=f'EF {lab[0].lower()} plan')
    ax.axvline(GRAB_STEP * DT, color='green', ls='--', alpha=.7,
               label=f'GRAB @ step {GRAB_STEP}')
    for bd, clr in [(approach_start, 'purple'), (descend_start, 'orange'),
                     (grab_end, 'red'), (lift_end, 'cyan'),
                     (transport_end, 'brown'), (place_end, 'gray')]:
        ax.axvline(bd * DT, color=clr, ls=':', alpha=0.35)
    ax.set_xlabel('t (s)'); ax.set_ylabel(lab); ax.legend(fontsize=7)

ax = axes[1, 1]
ax.plot(t_traj, gripper_cmd, 'k-', lw=1.5)
ax.axvline(GRAB_STEP * DT, color='green', ls='--', alpha=.7)
ax.set_xlabel('t (s)'); ax.set_ylabel('vel (m/s)')
ax.set_title('Gripper command')

plt.tight_layout()
plt.savefig('plot_traj_design.png', dpi=120)
print("  Saved -> plot_traj_design.png")
plt.show()


# ===================================================================
#  4.  JACOBIAN VELOCITY IK  (6x6: position + FULL orientation)
# ===================================================================
#   6-DOF control: position (XYZ) + full rotation (axis-angle)
#
#   UPGRADE: replaced Z-axis-only tracking (2 effective DOF) with
#   full 3-DOF rotation error (axis-angle / rotation vector).
#   → Joint 6 (wrist roll) is now actively driven.
#   → Orientation error = rotvec(R_target * R_current^{-1})
#
#   At each step k:
#     V_pos  = feedforward + Kp * (P_des - P_cur)              (3D)
#     V_ori  = Kp_ori * rotvec(R_target * R_cur^{-1})          (3D)
#     J      = numerical Jacobian  (6x6)
#              rows 0-2: dPosition/dq
#              rows 3-5: d(Rotation)/dq  (axis-angle perturbation)
#     V_full = [V_pos; W_ORI * V_ori]                   (weighted)
#     q_dot  = J_w^T (J_w*J_w^T + lambda*I)^{-1} V_w   (damped pseudo-inv)
#     q_new  = q_cur + q_dot * dt                        (integrate)
# ===================================================================

print("\n" + "=" * 60)
print("  4.  JACOBIAN VELOCITY IK  (6x6 pos+ori, stepping mode)")
print("=" * 60)

# (CoppeliaSim client already connected in section 0)

ef_h    = sim.getObject('/yaskawa/gripperEF')
joint_h = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
print(f"  gripperEF = {ef_h}")
print(f"  joints    = {joint_h}")

sim.setStepping(True)
sim.startSimulation()

# let physics settle
for _ in range(3):
    sim.step()

# read home joint config and EF orientation
q_cur = np.array([sim.getJointPosition(j) for j in joint_h])
P_cur = np.array(sim.getObjectPosition(ef_h, sim.handle_world))

# Use quaternion instead of getObjectMatrix to avoid row/column-major ambiguity
# CoppeliaSim getObjectQuaternion returns [x, y, z, w] = scipy convention
ef_quat = sim.getObjectQuaternion(ef_h, sim.handle_world)
R_home  = R.from_quat(ef_quat)          # full rotation  (for Slerp later)
ez_home = R_home.apply([0, 0, 1])       # EF Z-axis in world frame

print(f"  Home q       = {np.round(q_cur, 4)}")
print(f"  Home EF      = {np.round(P_cur, 4)}")
print(f"  Home EF Z-ax = {np.round(ez_home, 4)}")
print(f"  Home quat    = {np.round(ef_quat, 4)}")

# -- Full orientation target per step  (Yaw + Roll + bounded Pitch) -----------
#
#   Phase behaviour:
#     WAIT      : R_home
#     APPROACH  : Slerp  R_home  ->  R_grab
#     DESCEND   : per-step side-grasp frame (cup moving on carousel)
#     GRAB      : per-step side-grasp frame (tracking cup)
#     LIFT      : hold R_grab  (position priority)
#     TRANSPORT : Slerp  R_grab  ->  R_place
#     PLACE     : R_place  (conveyor-tilted)
#     RETURN    : Slerp  R_place  ->  R_home
# --------------------------------------------------------------------------

quat_target = np.zeros((STEPS, 4))   # scipy convention [x, y, z, w]

slerp_home_to_grab  = Slerp([0.0, 1.0], R.concatenate([R_home, R_grab]))
slerp_grab_to_place = Slerp([0.0, 1.0], R.concatenate([R_grab, R_place]))
slerp_place_to_home = Slerp([0.0, 1.0], R.concatenate([R_place, R_home]))

for k in range(STEPS):
    if k < wait_end:
        # ---- WAIT: keep home ----
        quat_target[k] = R_home.as_quat()

    elif k < descend_start:
        # ---- APPROACH: Slerp home -> grab (full rotation) ----
        s = scurve((k - approach_start) / max(APPROACH_STEPS - 1, 1))
        quat_target[k] = slerp_home_to_grab([s])[0].as_quat()

    elif k < GRAB_STEP:
        # ---- DESCEND: per-step side-grasp frame (no pitch) ----
        cup_pos_k = np.array([cup_px[k], cup_py[k], cup_pz[k]])
        n_cup_k   = R.from_euler('XYZ', cup_euler_xyz[k]).apply([0, 0, 1])
        R_k, _, _, _, _ = _side_grasp_frame(_R_home_approx, cup_pos_k, robot_base,
                                            n_cup_k, max_pitch=0)
        quat_target[k] = R_k.as_quat()

    elif k < grab_end:
        # ---- GRAB: per-step side-grasp frame (no pitch) ----
        idx = min(k, STEPS_COLLECT - 1)
        cup_pos_k = np.array([cup_px[idx], cup_py[idx], cup_pz[idx]])
        n_cup_k   = R.from_euler('XYZ', cup_euler_xyz[idx]).apply([0, 0, 1])
        R_k, _, _, _, _ = _side_grasp_frame(_R_home_approx, cup_pos_k, robot_base,
                                            n_cup_k, max_pitch=0)
        quat_target[k] = R_k.as_quat()

    elif k < lift_end:
        # ---- LIFT: hold R_grab (don't fight position tracking) ----
        quat_target[k] = R_grab.as_quat()

    elif k < transport_end:
        # ---- TRANSPORT: Slerp R_grab -> R_place ----
        frac = (k - lift_end) / max(TRANSPORT_STEPS - 1, 1)
        s_rot = scurve(frac)
        quat_target[k] = slerp_grab_to_place([s_rot])[0].as_quat()

    elif k < place_end:
        # ---- PLACE: conveyor-tilted orientation ----
        quat_target[k] = R_place.as_quat()

    else:
        # ---- RETURN: Slerp place -> home ----
        s = scurve((k - place_end) / max(RETURN_STEPS - 1, 1))
        quat_target[k] = slerp_place_to_home([s])[0].as_quat()

# Designed Euler angles (alpha, beta, gamma) for rubric 2.4 plots
euler_des_arr = np.array([R.from_quat(q).as_euler('XYZ') for q in quat_target])
# Also extract Z-axis direction for diagnostic printing
ez_target = np.array([R.from_quat(q).apply([0, 0, 1]) for q in quat_target])

print(f"  R_grab euler      = {np.degrees(R_grab.as_euler('XYZ')).round(1)}°")
print(f"  R_place euler     = {np.degrees(R_place.as_euler('XYZ')).round(1)}°")
print(f"  ez_target[0]      = {ez_target[0].round(3)} (home)")
print(f"  ez_target[{GRAB_STEP}]  = {ez_target[GRAB_STEP].round(3)} (grab)")

# Weight matrix (orientation rows scaled)
W_diag = np.array([1.0, 1.0, 1.0, W_ORI, W_ORI, W_ORI])

# storage
q_all        = np.zeros((STEPS, 6))
qd_all       = np.zeros((STEPS, 6))
ef_actual    = np.zeros((STEPS, 3))
ef_err_arr   = np.zeros(STEPS)
ori_err_arr  = np.zeros(STEPS)
ef_euler_arr = np.zeros((STEPS, 3))    # actual EF Euler XYZ per step (rubric 2.4)

t0 = time.time()

for step in range(STEPS):
    P_des = np.array([ef_x[step], ef_y[step], ef_z[step]])

    # -- feedforward velocity (position) --
    if step == 0:
        V_ff = np.zeros(3)
    else:
        V_ff = np.array([(ef_x[step] - ef_x[step - 1]) / DT,
                          (ef_y[step] - ef_y[step - 1]) / DT,
                          (ef_z[step] - ef_z[step - 1]) / DT])

    # -- current EF state --
    P_cur = np.array(sim.getObjectPosition(ef_h, sim.handle_world))
    R_cur = R.from_quat(sim.getObjectQuaternion(ef_h, sim.handle_world))

    # -- position velocity --
    V_pos = V_ff + Kp * (P_des - P_cur)

    # -- orientation velocity (full rotation error via axis-angle) --
    R_target_k = R.from_quat(quat_target[step])
    ori_err_vec = (R_target_k * R_cur.inv()).as_rotvec()   # rotation vector
    V_ori = Kp_ori * ori_err_vec

    # -- skip Jacobian during WAIT if everything near zero --
    V_full = np.concatenate([V_pos, V_ori])
    if step < wait_end and np.linalg.norm(V_full) < 1e-5:
        q_dot = np.zeros(6)
    else:
        # -- numerical Jacobian (6x6) via central differences --
        # rows 0-2: dPosition/dq
        # rows 3-5: d(Rotation)/dq  (axis-angle perturbation)
        J = np.zeros((6, 6))
        for i in range(6):
            sim.setJointPosition(joint_h[i], float(q_cur[i] + DELTA_Q))
            p_plus = np.array(sim.getObjectPosition(ef_h, sim.handle_world))
            R_plus = R.from_quat(sim.getObjectQuaternion(ef_h, sim.handle_world))

            sim.setJointPosition(joint_h[i], float(q_cur[i] - DELTA_Q))
            p_minus = np.array(sim.getObjectPosition(ef_h, sim.handle_world))
            R_minus = R.from_quat(sim.getObjectQuaternion(ef_h, sim.handle_world))

            sim.setJointPosition(joint_h[i], float(q_cur[i]))   # restore
            J[0:3, i] = (p_plus - p_minus) / (2.0 * DELTA_Q)
            J[3:6, i] = (R_plus * R_minus.inv()).as_rotvec() / (2.0 * DELTA_Q)

        # -- weighted damped pseudo-inverse -- ##Jacobian Velocity IK with orientation weighting##
        # Reduce orientation weight during RETURN (ef_home at 103% reach;   
        # full orientation fights position → prioritise position tracking)  
        W_step = W_diag.copy()                                                
        if step >= place_end:                                                   
            W_step[3:6] *= 0.10          # RETURN: position priority (103% reach)   
        elif step >= transport_end:                                                
            W_step[3:6] *= 1.0           # PLACE: full ori weight — match conveyor tilt 
        elif step >= lift_end:
            W_step[3:6] *= 1.0           # TRANSPORT: full ori weight — J6 must track
        elif step >= grab_end:
            W_step[3:6] *= 0.3           # LIFT: some ori tracking
        J_w    = np.diag(W_step) @ J
        V_w    = W_step * V_full
        JJT    = J_w @ J_w.T + LAMBDA_BASE * np.eye(6)
        q_dot  = J_w.T @ np.linalg.solve(JJT, V_w)

    # -- clamp joint velocities to prevent singularity explosions --
    q_dot = np.clip(q_dot, -QD_MAX, QD_MAX)

    # -- integrate + clamp to joint limits --
    q_new = q_cur + q_dot * DT
    #q_new = np.clip(q_new, Q_MIN, Q_MAX)

    # -- apply to simulation --
    for i in range(6):
        sim.setJointPosition(joint_h[i], float(q_new[i]))

    # -- record --
    P_actual = np.array(sim.getObjectPosition(ef_h, sim.handle_world))
    R_actual = R.from_quat(sim.getObjectQuaternion(ef_h, sim.handle_world))
    q_all[step]        = q_new
    qd_all[step]       = q_dot
    ef_actual[step]    = P_actual
    ef_err_arr[step]   = np.linalg.norm(P_des - P_actual)
    ori_err_arr[step]  = np.linalg.norm((R_target_k * R_actual.inv()).as_rotvec())
    ef_euler_arr[step] = R_actual.as_euler('XYZ')   # extrinsic XYZ = CoppeliaSim convention

    q_cur = q_new

    # -- advance physics (keeps conveyor / cup in sync) --
    sim.step()

    # -- progress --
    if step % 50 == 0 or step == STEPS - 1:
        phase_tag = ""
        if   step < wait_end:        phase_tag = "WAIT"
        elif step < descend_start:   phase_tag = "APPROACH"
        elif step < GRAB_STEP:       phase_tag = "DESCEND"
        elif step < grab_end:        phase_tag = "GRAB"
        elif step < lift_end:        phase_tag = "LIFT"
        elif step < transport_end:   phase_tag = "TRANSPORT"
        elif step < place_end:       phase_tag = "PLACE"
        else:                        phase_tag = "RETURN"
        print(f"    step {step:4d}/{STEPS}  t={step * DT:5.1f}s"
              f"  pos_err={ef_err_arr[step]:.4f}m"
              f"  ori_err={ori_err_arr[step]:.3f}"
              f"  |qd|={np.linalg.norm(q_dot):.3f}"
              f"  [{phase_tag}]")

    # -- warn on big error --
    if ef_err_arr[step] > 0.05:
        print(f"  !! Large position error at step {step}: "
              f"{ef_err_arr[step]:.3f} m")

elapsed = time.time() - t0
sim.stopSimulation()

print(f"\n  >> Jacobian IK complete  ({elapsed:.1f} s)")
print(f"    max  pos error  = {np.max(ef_err_arr):.4f} m")
print(f"    mean pos error  = {np.mean(ef_err_arr):.4f} m")
print(f"    grab max pos err= {np.max(ef_err_arr[GRAB_STEP:grab_end]):.4f} m")
print(f"    max  ori error  = {np.max(ori_err_arr):.4f}")
print(f"    grab max ori err= {np.max(ori_err_arr[GRAB_STEP:grab_end]):.4f}")

# Joint 6 diagnostics — verify wrist roll is changing
print(f"\n  Joint 6 (wrist roll) at key steps:")
print(f"    HOME          q6 = {np.degrees(q_all[0, 5]):7.1f}°")
print(f"    GRAB  (step {GRAB_STEP}) q6 = {np.degrees(q_all[GRAB_STEP, 5]):7.1f}°")
print(f"    PLACE (step {place_end-5}) q6 = {np.degrees(q_all[place_end-5, 5]):7.1f}°")
print(f"    RETURN end    q6 = {np.degrees(q_all[STEPS-1, 5]):7.1f}°")
# Actual EF Euler at key steps
_eul_grab = np.degrees(ef_euler_arr[GRAB_STEP])
_eul_place = np.degrees(ef_euler_arr[min(place_end-1, STEPS-1)])
_eul_des_grab = np.degrees(euler_des_arr[GRAB_STEP])
_eul_des_place = np.degrees(euler_des_arr[min(place_end-1, STEPS-1)])
print(f"    EF euler @GRAB : des=({_eul_des_grab[0]:.1f}, {_eul_des_grab[1]:.1f}, {_eul_des_grab[2]:.1f})  act=({_eul_grab[0]:.1f}, {_eul_grab[1]:.1f}, {_eul_grab[2]:.1f})")
print(f"    EF euler @PLACE: des=({_eul_des_place[0]:.1f}, {_eul_des_place[1]:.1f}, {_eul_des_place[2]:.1f})  act=({_eul_place[0]:.1f}, {_eul_place[1]:.1f}, {_eul_place[2]:.1f})")


# ===================================================================
#  5.  SAVE TRAJECTORY
# ===================================================================
out = 'trajectory.npz'
np.savez(out,
         # joint positions  (part3 reads these)
         j1=q_all[:, 0], j2=q_all[:, 1], j3=q_all[:, 2],
         j4=q_all[:, 3], j5=q_all[:, 4], j6=q_all[:, 5],
         # joint velocities  (professor's Step 3 requirement)
         jd1=qd_all[:, 0], jd2=qd_all[:, 1], jd3=qd_all[:, 2],
         jd4=qd_all[:, 3], jd5=qd_all[:, 4], jd6=qd_all[:, 5],
         # gripper
         gripper_cmd=gripper_cmd,
         # designed EF trajectory (position)
         ef_x=ef_x, ef_y=ef_y, ef_z=ef_z,
         # designed EF orientation (Euler XYZ, radians)  -- rubric 2.4
         euler_des_alpha=euler_des_arr[:, 0],
         euler_des_beta=euler_des_arr[:, 1],
         euler_des_gamma=euler_des_arr[:, 2],
         # actual EF from IK (position)
         ef_actual_x=ef_actual[:, 0],
         ef_actual_y=ef_actual[:, 1],
         ef_actual_z=ef_actual[:, 2],
         # actual EF orientation (Euler XYZ, radians)  -- rubric 2.4
         euler_actual_alpha=ef_euler_arr[:, 0],
         euler_actual_beta=ef_euler_arr[:, 1],
         euler_actual_gamma=ef_euler_arr[:, 2],
         ef_error=ef_err_arr,
         ori_error=ori_err_arr,
         # metadata
         times=np.arange(STEPS) * DT,
         GRAB_STEP=GRAB_STEP, DT=DT, STEPS=STEPS,
         ef_home=ef_home, robot_base=robot_base, place_pos=place_pos)
print(f"\n  Saved -> {out}")


# ===================================================================
#  6.  PLOTS -- JOINTS, VELOCITIES, TRACKING ERROR
# ===================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('Jacobian IK Results -- Joint Trajectories & Error', fontsize=13)

# joints 1-3  (deg)
for j in range(3):
    axes2[0, 0].plot(t_traj, np.degrees(q_all[:, j]), label=f'j{j+1}')
axes2[0, 0].set_title('Joints 1-3 (deg)'); axes2[0, 0].legend()
axes2[0, 0].set_xlabel('t (s)')

# joints 4-6  (deg)
for j in range(3, 6):
    axes2[0, 1].plot(t_traj, np.degrees(q_all[:, j]), label=f'j{j+1}')
axes2[0, 1].set_title('Joints 4-6 (deg)'); axes2[0, 1].legend()
axes2[0, 1].set_xlabel('t (s)')

# joint velocities 1-3
for j in range(3):
    axes2[1, 0].plot(t_traj, qd_all[:, j], label=f'qd{j+1}')
axes2[1, 0].set_title('Joint vel 1-3 (rad/s)'); axes2[1, 0].legend()
axes2[1, 0].set_xlabel('t (s)')

# joint velocities 4-6
for j in range(3, 6):
    axes2[1, 1].plot(t_traj, qd_all[:, j], label=f'qd{j+1}')
axes2[1, 1].set_title('Joint vel 4-6 (rad/s)'); axes2[1, 1].legend()
axes2[1, 1].set_xlabel('t (s)')

# Position tracking error (mm)
axes2[0, 2].plot(t_traj, ef_err_arr * 1000, 'r-', lw=1.2, label='pos (mm)')
axes2[0, 2].plot(t_traj, ori_err_arr * 100, 'b--', lw=1.0, label='ori x100')
axes2[0, 2].set_title('Tracking error')
axes2[0, 2].set_xlabel('t (s)'); axes2[0, 2].set_ylabel('mm / scaled')
axes2[0, 2].legend(fontsize=7)

# per-axis position error (mm)
axes2[1, 2].plot(t_traj, (ef_actual[:, 0] - ef_x) * 1000, label='dx')
axes2[1, 2].plot(t_traj, (ef_actual[:, 1] - ef_y) * 1000, label='dy')
axes2[1, 2].plot(t_traj, (ef_actual[:, 2] - ef_z) * 1000, label='dz')
axes2[1, 2].set_title('Per-axis pos error (mm)'); axes2[1, 2].legend()
axes2[1, 2].set_xlabel('t (s)')

for ax in axes2.flat:
    ax.axvline(GRAB_STEP * DT, color='green', ls='--', alpha=0.4)

plt.tight_layout()
plt.savefig('plot_jacobian_ik.png', dpi=120)
print("  Saved -> plot_jacobian_ik.png")
plt.show()


# ===================================================================
#  6B.  PLOTS -- EF ORIENTATION & ANGULAR VELOCITY  (rubric 2.4)
# ===================================================================
euler_des_deg = np.degrees(euler_des_arr)
euler_act_deg = np.degrees(ef_euler_arr)
omega_des     = np.gradient(euler_des_deg,  DT, axis=0)   # deg/s
omega_act     = np.gradient(euler_act_deg,  DT, axis=0)

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle('EF Orientation & Angular Velocity  (rubric 2.4)', fontsize=13)

_lbl_ori = [r'$\alpha$ (deg)', r'$\beta$ (deg)', r'$\gamma$ (deg)']
_lbl_omg = [r'$\omega_\alpha$ (deg/s)', r'$\omega_\beta$ (deg/s)',
            r'$\omega_\gamma$ (deg/s)']

for ci in range(3):
    ax = axes3[0, ci]
    ax.plot(t_traj, euler_des_deg[:, ci], 'b-',  lw=1.5, label='designed')
    ax.plot(t_traj, euler_act_deg[:, ci], 'r--', lw=1.0, label='actual (IK)')
    ax.set_xlabel('t (s)'); ax.set_ylabel(_lbl_ori[ci])
    ax.set_title(_lbl_ori[ci]); ax.legend(fontsize=7)
    ax.axvline(GRAB_STEP * DT, color='green', ls='--', alpha=0.4)
    for bd, clr in [(descend_start, 'orange'), (grab_end, 'red'),
                     (lift_end, 'cyan'), (transport_end, 'brown'),
                     (place_end, 'gray')]:
        ax.axvline(bd * DT, color=clr, ls=':', alpha=0.35)

    ax = axes3[1, ci]
    ax.plot(t_traj, omega_des[:, ci], 'b-',  lw=1.5, label='designed')
    ax.plot(t_traj, omega_act[:, ci], 'r--', lw=1.0, label='actual (IK)')
    ax.set_xlabel('t (s)'); ax.set_ylabel(_lbl_omg[ci])
    ax.set_title(_lbl_omg[ci]); ax.legend(fontsize=7)
    ax.axvline(GRAB_STEP * DT, color='green', ls='--', alpha=0.4)
    for bd, clr in [(descend_start, 'orange'), (grab_end, 'red'),
                     (lift_end, 'cyan'), (transport_end, 'brown'),
                     (place_end, 'gray')]:
        ax.axvline(bd * DT, color=clr, ls=':', alpha=0.35)

plt.tight_layout()
plt.savefig('plot_orientation.png', dpi=120)
print("  Saved -> plot_orientation.png")
plt.show()


print("\n" + "=" * 60)
print("  Part 2 complete")
print("  Next:  python part3_run.py")
print("=" * 60)
