"""Quick test: verify Euler convention fix and compute cup-gripper offset."""
import numpy as np
from scipy.spatial.transform import Rotation as R

R_home = R.from_quat([0.7069, 0.0109, 0.7072, 0.0011])
rb = np.array([0.0, 0.0, 0.33])

def sgf(Rhr, tp, rb, sn, mp=0.0):
    yaw = np.arctan2(tp[1]-rb[1], tp[0]-rb[0])
    Rf = R.from_euler('z', yaw) * Rhr
    n = sn.copy()
    if n[2] < 0: n = -n
    n /= np.linalg.norm(n)
    nl = Rf.inv().apply(n)
    roll = np.arctan2(nl[1], nl[0])
    Rr = Rf * R.from_euler('z', roll)
    zh = Rr.apply([0,0,1])
    pitch = 0.0
    if mp > 0:
        nl2 = Rr.inv().apply(n)
        pr = np.arctan2(-nl2[2], nl2[0])
        pitch = np.clip(pr, -mp, mp)
    Rfin = Rr * R.from_euler('y', pitch)
    return Rfin, zh, yaw, roll, pitch

# Load actual cup euler at grab
d = np.load('cup_data.npz', allow_pickle=True)
ox = float(d['cup_ox'][236])
oy = float(d['cup_oy'][236])
oz = float(d['cup_oz'][236])
print(f"Cup euler at GRAB (deg): ({np.degrees(ox):.1f}, {np.degrees(oy):.1f}, {np.degrees(oz):.1f})")

n_old = R.from_euler('xyz', [ox,oy,oz]).apply([0,0,1])
n_new = R.from_euler('XYZ', [ox,oy,oz]).apply([0,0,1])
print(f"n_cup OLD (xyz): {n_old.round(4)}")
print(f"n_cup NEW (XYZ): {n_new.round(4)}")

n_conv_old = R.from_euler('xyz', [-15, 10, -180], degrees=True).apply([0,0,1])
n_conv_new = R.from_euler('XYZ', [-15, 10, -180], degrees=True).apply([0,0,1])
print(f"n_conv OLD (xyz): {n_conv_old.round(4)}")
print(f"n_conv NEW (XYZ): {n_conv_new.round(4)}")

# ── GRAB with CORRECT normal ──
cup_pos = np.array([0.2209, -0.5326, 0.446])
Rg, zh, yw, rl, pt = sgf(R_home, cup_pos, rb, n_new, mp=0)
xe = Rg.apply([1,0,0])
ze = Rg.apply([0,0,1])
tilt = np.degrees(np.arccos(np.clip(np.dot(xe,[0,0,1]),-1,1)))
ad = np.degrees(np.linalg.norm((Rg * R_home.inv()).as_rotvec()))
print(f"\n=== GRAB (no pitch, CORRECT normal) ===")
print(f"  yaw={np.degrees(yw):.1f}  roll={np.degrees(rl):.1f}  pitch={np.degrees(pt):.1f}")
print(f"  x_ee (UP)={xe.round(4)}  tilt={tilt:.1f} deg (target=18.0)")
print(f"  z_ee (approach)={ze.round(4)}")
print(f"  euler(XYZ)={np.degrees(Rg.as_euler('XYZ')).round(1)}")
print(f"  angular dist from home={ad:.1f} deg")

# ── PLACE with CORRECT normal ──
pl = np.array([0.4545, 0.5, 0.43])
Rp, zhp, yp, rp, pp = sgf(R_home, pl, rb, n_conv_new, mp=np.radians(20))
xp = Rp.apply([1,0,0])
zp = Rp.apply([0,0,1])
tp = np.degrees(np.arccos(np.clip(np.dot(xp,[0,0,1]),-1,1)))
adp = np.degrees(np.linalg.norm((Rp * R_home.inv()).as_rotvec()))
print(f"\n=== PLACE (max_pitch=20, CORRECT normal) ===")
print(f"  yaw={np.degrees(yp):.1f}  roll={np.degrees(rp):.1f}  pitch={np.degrees(pp):.1f}")
print(f"  x_ee (UP)={xp.round(4)}  tilt={tp:.1f} deg")
print(f"  z_ee (approach)={zp.round(4)}")
print(f"  euler(XYZ)={np.degrees(Rp.as_euler('XYZ')).round(1)}")
print(f"  angular dist from home={adp:.1f} deg")

# ── Cup-gripper offset ──
cup_z_in_grip = Rg.inv().apply(n_new)
print(f"\n=== CUP-GRIPPER OFFSET ===")
print(f"  cup normal in gripper frame: {cup_z_in_grip.round(4)}")
offset_angle = np.degrees(np.arccos(np.clip(np.dot(cup_z_in_grip, [1,0,0]),-1,1)))
print(f"  offset angle (from gripper X): {offset_angle:.1f} deg")

# What cup normal looks like at PLACE with current R_place (no compensation)
cup_world_at_place = Rp.apply(cup_z_in_grip)
angle_cup_conv = np.degrees(np.arccos(np.clip(np.dot(cup_world_at_place, n_conv_new),-1,1)))
print(f"  cup normal at place (uncompensated): {cup_world_at_place.round(4)}")
print(f"  angle cup->conv (MISMATCH): {angle_cup_conv:.1f} deg")

# ── Compensated PLACE ──
# R_corr: rotate cup_z_in_grip -> [1,0,0] in gripper-local frame
a = cup_z_in_grip / np.linalg.norm(cup_z_in_grip)
b = np.array([1.0, 0.0, 0.0])
cross = np.cross(a, b)
if np.linalg.norm(cross) > 1e-10:
    axis = cross / np.linalg.norm(cross)
    angle = np.arccos(np.clip(np.dot(a, b), -1, 1))
    R_corr = R.from_rotvec(axis * angle)
else:
    R_corr = R.identity()

R_place_comp = Rp * R_corr
xpc = R_place_comp.apply([1,0,0])
zpc = R_place_comp.apply([0,0,1])
tpc = np.degrees(np.arccos(np.clip(np.dot(xpc,[0,0,1]),-1,1)))
adpc = np.degrees(np.linalg.norm((R_place_comp * R_home.inv()).as_rotvec()))

cup_world_comp = R_place_comp.apply(cup_z_in_grip)
angle_comp = np.degrees(np.arccos(np.clip(np.dot(cup_world_comp, n_conv_new),-1,1)))

print(f"\n=== COMPENSATED PLACE (R_place * R_corr) ===")
print(f"  R_corr angle: {np.degrees(angle):.1f} deg")
print(f"  x_ee (UP)={xpc.round(4)}  tilt={tpc:.1f} deg")
print(f"  z_ee (approach)={zpc.round(4)}")
print(f"  euler(XYZ)={np.degrees(R_place_comp.as_euler('XYZ')).round(1)}")
print(f"  angular dist from home={adpc:.1f} deg")
print(f"  cup normal at place (COMPENSATED): {cup_world_comp.round(4)}")
print(f"  angle cup->conv (should be ~0): {angle_comp:.1f} deg")
print(f"  conv normal:                     {n_conv_new.round(4)}")

# Angular dist grab->place (for TRANSPORT Slerp)
agp = np.degrees(np.linalg.norm((Rg.inv() * R_place_comp).as_rotvec()))
print(f"\n  angular dist grab->place_comp: {agp:.1f} deg")
agp2 = np.degrees(np.linalg.norm((Rg.inv() * Rp).as_rotvec()))
print(f"  angular dist grab->place_orig: {agp2:.1f} deg")
