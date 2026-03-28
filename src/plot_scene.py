"""
Plot Scene Hierarchy  —  position + orientation frames ของ object ทุกตัวใน CoppeliaSim
=====================================================================================
วิธีรัน:
  1. เปิด yaskawaGP8_group19.ttt ใน CoppeliaSim (ไม่ต้อง Play)
  2. python plot_scene.py

Output:
  - plot_scene_all.png       : 3D view ของทุก object พร้อม frame arrows
  - plot_scene_key.png       : เน้นเฉพาะ gripperEF, Cup, conveyor, conveyorSystem
  - plot_scene_detail.png    : เปรียบเทียบ frame ของ key objects
  - console table            : position + euler (deg) ของทุกตัว
"""

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import sys

# ─── Connect ───────────────────────────────────────────────────
print("  Connecting to CoppeliaSim...")
try:
    client = RemoteAPIClient()
    sim    = client.require('sim')
    print("  Connected OK")
except Exception as e:
    print(f"  ERROR: Cannot connect to CoppeliaSim: {e}")
    print("  Make sure CoppeliaSim is open with yaskawaGP8_group19.ttt")
    print("  (don't press Play)")
    sys.exit(1)

# ─── Enumerate ALL objects using getObjectsInTree ──────────────
all_paths = []
try:
    # Get all objects from the root of the scene
    root_h = sim.getObject('/')    # scene root
    all_handles = sim.getObjectsInTree(root_h, sim.handle_all, 0)
    print(f"  Found {len(all_handles)} objects in scene tree")
    for h in all_handles:
        try:
            alias = sim.getObjectAlias(h, 1)   # full path alias
            all_paths.append(alias)
        except:
            pass
except Exception as e:
    print(f"  getObjectsInTree failed: {e}, using known paths...")
    all_paths = []

# Add known paths as fallback
KNOWN_PATHS = [
    '/yaskawa', '/yaskawa/gripperEF', '/yaskawa/MicoHand',
    '/yaskawa/base_link_base',
    '/conveyorSystem', '/conveyorSystem/Cup',
    '/conveyor', '/Vision_sensor', '/Floor',
]
for p in KNOWN_PATHS:
    if p not in all_paths:
        all_paths.append(p)

# ─── Query each object ─────────────────────────────────────────
results = []
seen_handles = set()
tried_paths = set()

print(f"\n{'='*90}")
print(f"  SCENE OBJECTS  —  Position (world) & Orientation (euler xyz, deg)")
print(f"{'='*90}")
print(f"  {'Object':<55s} {'X':>7s} {'Y':>7s} {'Z':>7s}   {'a':>7s} {'b':>7s} {'g':>7s}")
print(f"  {'-'*55} {'-'*7} {'-'*7} {'-'*7}   {'-'*7} {'-'*7} {'-'*7}")

for path in all_paths:
    if path in tried_paths:
        continue
    tried_paths.add(path)
    try:
        h = sim.getObject(path)
        if h in seen_handles:
            continue
        seen_handles.add(h)
        
        pos   = sim.getObjectPosition(h, sim.handle_world)
        euler = sim.getObjectOrientation(h, sim.handle_world)  # xyz intrinsic
        euler_deg = np.degrees(euler)
        
        # Also get quaternion
        quat = sim.getObjectQuaternion(h, sim.handle_world)  # [x,y,z,w]
        
        results.append({
            'path': path,
            'handle': h,
            'pos': np.array(pos),
            'euler_deg': euler_deg,
            'quat': np.array(quat),
            'rot': R.from_quat(quat),  # scipy uses [x,y,z,w]
        })
        
        print(f"  {path:<55s} {pos[0]:+7.4f} {pos[1]:+7.4f} {pos[2]:+7.4f}   "
              f"{euler_deg[0]:+7.1f} {euler_deg[1]:+7.1f} {euler_deg[2]:+7.1f}")
    except Exception:
        pass  # object not found — skip silently

print(f"\n  Total objects found: {len(results)}")

# ─── Also show frames relative to /yaskawa base ───────────────
try:
    base_h = sim.getObject('/yaskawa')
    print(f"\n{'='*90}")
    print(f"  KEY OBJECTS relative to /yaskawa base")
    print(f"{'='*90}")
    print(f"  {'Object':<55s} {'X':>7s} {'Y':>7s} {'Z':>7s}   {'a':>7s} {'b':>7s} {'g':>7s}")
    print(f"  {'-'*55} {'-'*7} {'-'*7} {'-'*7}   {'-'*7} {'-'*7} {'-'*7}")
    
    key_objects = ['/yaskawa/gripperEF', '/conveyorSystem', '/conveyorSystem/Cup',
                   '/conveyor', '/Vision_sensor']
    for path in key_objects:
        try:
            h = sim.getObject(path)
            pos   = sim.getObjectPosition(h, base_h)
            euler = sim.getObjectOrientation(h, base_h)
            euler_deg = np.degrees(euler)
            print(f"  {path:<55s} {pos[0]:+7.4f} {pos[1]:+7.4f} {pos[2]:+7.4f}   "
                  f"{euler_deg[0]:+7.1f} {euler_deg[1]:+7.1f} {euler_deg[2]:+7.1f}")
        except:
            pass
except:
    pass

# ─── Detailed surface normals ──────────────────────────────────
print(f"\n{'='*90}")
print(f"  SURFACE NORMALS (Z-axis of each object = local 'up')")
print(f"{'='*90}")

for r in results:
    z_world = r['rot'].apply([0, 0, 1])
    x_world = r['rot'].apply([1, 0, 0])
    tilt = np.degrees(np.arccos(np.clip(z_world[2], -1, 1)))
    short = r['path'].split('/')[-1] if '/' in r['path'] else r['path']
    if tilt > 0.5:  # only show tilted objects
        print(f"  {r['path']:<55s}  Z_world={z_world.round(4)}  tilt={tilt:.1f} deg")


# ═══════════════════════════════════════════════════════════════
#  PLOT 1 : ALL objects — 3D scatter + frame arrows
# ═══════════════════════════════════════════════════════════════
def draw_frame(ax, pos, rot, length=0.05, lw=1.5, label=None):
    """Draw XYZ arrows for an object frame."""
    colors = ['r', 'g', 'b']  # X=red, Y=green, Z=blue
    axes_names = ['X', 'Y', 'Z']
    for i, (c, an) in enumerate(zip(colors, axes_names)):
        direction = rot.apply(np.eye(3)[i])
        ax.quiver(pos[0], pos[1], pos[2],
                  direction[0]*length, direction[1]*length, direction[2]*length,
                  color=c, linewidth=lw, arrow_length_ratio=0.2)
    if label:
        ax.text(pos[0], pos[1], pos[2] + length*1.2, label, fontsize=6, ha='center')


fig = plt.figure(figsize=(18, 10))

# ── Plot 1a: All objects top view (XY) ──
ax1 = fig.add_subplot(131)
for r in results:
    short = r['path'].split('/')[-1] if '/' in r['path'] else r['path']
    ax1.plot(r['pos'][0], r['pos'][1], 'ko', markersize=3)
    ax1.annotate(short, (r['pos'][0], r['pos'][1]), fontsize=5, rotation=20)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Top View (XY) — All Objects')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# ── Plot 1b: Side view (XZ) ──
ax2 = fig.add_subplot(132)
for r in results:
    short = r['path'].split('/')[-1] if '/' in r['path'] else r['path']
    ax2.plot(r['pos'][0], r['pos'][2], 'ko', markersize=3)
    ax2.annotate(short, (r['pos'][0], r['pos'][2]), fontsize=5, rotation=20)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Z (m)')
ax2.set_title('Side View (XZ) — All Objects')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# ── Plot 1c: 3D with frames ──
ax3 = fig.add_subplot(133, projection='3d')
for r in results:
    short = r['path'].split('/')[-1] if '/' in r['path'] else r['path']
    ax3.scatter(*r['pos'], s=10, c='black')
    draw_frame(ax3, r['pos'], r['rot'], length=0.06, lw=1.0, label=short)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('3D Scene — All Objects + Frames')

plt.tight_layout()
plt.savefig('plot_scene_all.png', dpi=150)
print(f"\n  Saved -> plot_scene_all.png")


# ═══════════════════════════════════════════════════════════════
#  PLOT 2 : KEY OBJECTS — gripperEF, Cup, conveyor, conveyorSystem
# ═══════════════════════════════════════════════════════════════
KEY_NAMES = ['yaskawa', 'gripperEF', 'Cup', 'conveyor', 'conveyorSystem',
             'Vision_sensor', 'Floor', 'base_link_base']

key_results = [r for r in results 
               if any(k in r['path'] for k in KEY_NAMES)]
# Also add robot joints for reference
joint_results = [r for r in results if 'joint' in r['path'] and 'link' not in r['path']]

fig2 = plt.figure(figsize=(18, 14))

# ── Top view ──
ax1 = fig2.add_subplot(221)
for r in key_results:
    short = r['path'].split('/')[-1]
    ax1.plot(r['pos'][0], r['pos'][1], 'o', markersize=8)
    ax1.annotate(short, (r['pos'][0], r['pos'][1]), fontsize=8,
                 xytext=(5, 5), textcoords='offset points')
    # Draw normal (Z-axis) projection on XY
    z_world = r['rot'].apply([0, 0, 1])
    ax1.arrow(r['pos'][0], r['pos'][1], z_world[0]*0.08, z_world[1]*0.08,
              head_width=0.01, color='blue', alpha=0.6)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Top View (XY) — Key Objects\n(blue arrows = Z-axis projection)')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# ── Side view XZ ──
ax2 = fig2.add_subplot(222)
for r in key_results:
    short = r['path'].split('/')[-1]
    ax2.plot(r['pos'][0], r['pos'][2], 'o', markersize=8)
    ax2.annotate(short, (r['pos'][0], r['pos'][2]), fontsize=8,
                 xytext=(5, 5), textcoords='offset points')
    z_world = r['rot'].apply([0, 0, 1])
    ax2.arrow(r['pos'][0], r['pos'][2], z_world[0]*0.08, z_world[2]*0.08,
              head_width=0.01, color='blue', alpha=0.6)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Z (m)')
ax2.set_title('Side View (XZ) — Key Objects\n(blue arrows = Z-axis/normal)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# ── Side view YZ ──
ax3 = fig2.add_subplot(223)
for r in key_results:
    short = r['path'].split('/')[-1]
    ax3.plot(r['pos'][1], r['pos'][2], 'o', markersize=8)
    ax3.annotate(short, (r['pos'][1], r['pos'][2]), fontsize=8,
                 xytext=(5, 5), textcoords='offset points')
    z_world = r['rot'].apply([0, 0, 1])
    ax3.arrow(r['pos'][1], r['pos'][2], z_world[1]*0.08, z_world[2]*0.08,
              head_width=0.01, color='blue', alpha=0.6)
ax3.set_xlabel('Y (m)')
ax3.set_ylabel('Z (m)')
ax3.set_title('Side View (YZ) — Key Objects\n(blue arrows = Z-axis/normal)')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# ── 3D with large frames ──
ax4 = fig2.add_subplot(224, projection='3d')
for r in key_results:
    short = r['path'].split('/')[-1]
    ax4.scatter(*r['pos'], s=40, zorder=5)
    draw_frame(ax4, r['pos'], r['rot'], length=0.10, lw=2.0, label=short)
# Add robot joints as thin chain
for r in joint_results:
    ax4.scatter(*r['pos'], s=15, c='gray', alpha=0.5)
    draw_frame(ax4, r['pos'], r['rot'], length=0.04, lw=0.8)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('3D — Key Objects + Frames\n(R=X, G=Y, B=Z)')

plt.tight_layout()
plt.savefig('plot_scene_key.png', dpi=150)
print(f"  Saved -> plot_scene_key.png")


# ═══════════════════════════════════════════════════════════════
#  PLOT 3 : Detailed comparison — Cup vs Conveyor normals
# ═══════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(14, 10))
ax = fig3.add_subplot(111, projection='3d')

# Get specific objects
cup_data = next((r for r in results if 'Cup' in r['path']), None)
conv_data = next((r for r in results if r['path'] == '/conveyor'), None)
convsys_data = next((r for r in results if r['path'] == '/conveyorSystem'), None)
ef_data = next((r for r in results if 'gripperEF' in r['path']), None)
base_data = next((r for r in results if r['path'] == '/yaskawa'), None)

# Draw big frames for key objects
important = [
    (cup_data, 'Cup', 'red'),
    (conv_data, '/conveyor', 'blue'),
    (convsys_data, '/conveyorSystem', 'green'),
    (ef_data, 'gripperEF', 'orange'),
    (base_data, '/yaskawa', 'purple'),
]

for data, name, color in important:
    if data is None:
        continue
    p = data['pos']
    rot = data['rot']
    ax.scatter(*p, s=100, c=color, zorder=10, label=name)
    
    # Draw XYZ axes with labels
    axis_colors = ['red', 'green', 'blue']
    axis_labels = ['X', 'Y', 'Z']
    for i in range(3):
        d = rot.apply(np.eye(3)[i])
        ax.quiver(p[0], p[1], p[2], d[0]*0.12, d[1]*0.12, d[2]*0.12,
                  color=axis_colors[i], linewidth=2.5, arrow_length_ratio=0.15)
    
    # Label
    ax.text(p[0], p[1], p[2]+0.15, f"{name}\n({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})",
            fontsize=8, ha='center', fontweight='bold', color=color)
    
    # Print normal (Z-axis)
    z_ax = rot.apply([0, 0, 1])
    tilt = np.degrees(np.arccos(np.clip(z_ax[2], -1, 1)))
    print(f"\n  {name}:")
    print(f"    pos    = {p.round(4)}")
    print(f"    euler  = {np.degrees(rot.as_euler('xyz')).round(1)} deg")
    print(f"    Z-axis = {z_ax.round(4)}  (tilt from vertical = {tilt:.1f} deg)")
    x_ax = rot.apply([1, 0, 0])
    y_ax = rot.apply([0, 1, 0])
    print(f"    X-axis = {x_ax.round(4)}")
    print(f"    Y-axis = {y_ax.round(4)}")

ax.legend(loc='upper left', fontsize=10)
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title('Key Objects — Frames & Normals\n(R=local X, G=local Y, B=local Z)', fontsize=14)

plt.tight_layout()
plt.savefig('plot_scene_detail.png', dpi=150)
print(f"\n  Saved -> plot_scene_detail.png")

# ═══════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  SUMMARY — Cup vs Conveyor orientation analysis")
print(f"{'='*90}")

if cup_data and conv_data:
    cup_z = cup_data['rot'].apply([0, 0, 1])
    conv_z = conv_data['rot'].apply([0, 0, 1])
    
    # Angle between cup normal and conveyor normal
    angle_between = np.degrees(np.arccos(np.clip(np.dot(cup_z, conv_z), -1, 1)))
    
    print(f"  Cup Z (normal)      = {cup_z.round(4)}")
    print(f"  Conveyor Z (normal) = {conv_z.round(4)}")
    print(f"  Angle between normals = {angle_between:.1f} deg")
    print(f"")
    print(f"  Cup tilt from vertical     = {np.degrees(np.arccos(np.clip(cup_z[2],-1,1))):.1f} deg")
    print(f"  Conveyor tilt from vertical = {np.degrees(np.arccos(np.clip(conv_z[2],-1,1))):.1f} deg")

if cup_data and ef_data:
    # Direction from robot base to cup
    if base_data:
        to_cup = cup_data['pos'] - base_data['pos']
        to_cup_horiz = to_cup.copy(); to_cup_horiz[2] = 0
        to_cup_horiz /= np.linalg.norm(to_cup_horiz)
        print(f"\n  Direction base -> cup  (horiz) = {to_cup_horiz.round(4)}")
        
    if base_data and conv_data:
        to_conv = conv_data['pos'] - base_data['pos']
        to_conv_horiz = to_conv.copy(); to_conv_horiz[2] = 0
        to_conv_horiz /= np.linalg.norm(to_conv_horiz)
        print(f"  Direction base -> conv (horiz) = {to_conv_horiz.round(4)}")

print(f"\n  Done. Check plot_scene_all.png, plot_scene_key.png, plot_scene_detail.png")
