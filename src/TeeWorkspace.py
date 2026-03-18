import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================
# 1. เชื่อมต่อกับ CoppeliaSim
# ==========================================
client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()

# --- ชื่อ Object จาก Scene Hierarchy ---
END_EFFECTOR_NAME = '/yaskawa/MicoHand'   # ปลายแขน (End-Effector)
CUP_NAME          = '/Cup'                 # แก้วน้ำ

# ดึง Handle
try:
    ee_handle  = sim.getObject(END_EFFECTOR_NAME)
    cup_handle = sim.getObject(CUP_NAME)
except Exception as e:
    print(f"[ERROR] หา Object ไม่พบ: {e}")
    sim.stopSimulation()
    exit()

# ==========================================
# 2. ลูปเก็บข้อมูลระหว่าง Simulation
# ==========================================
record_duration = 20   # วินาที
OVERLAP_THRESHOLD = 0.15  # เมตร — ระยะที่ถือว่า "ทับซ้อน"

time_data  = []
ee_x, ee_y, ee_z   = [], [], []
cup_x, cup_y, cup_z = [], [], []
overlap_flags = []   # True = ทับซ้อนในช่วงเวลานั้น

print("กำลังบันทึกข้อมูล Workspace และตำแหน่งแก้ว...")
start_time = sim.getSimulationTime()

while True:
    t = sim.getSimulationTime()
    if t - start_time > record_duration:
        break

    pos_ee  = sim.getObjectPosition(ee_handle,  sim.handle_world)
    pos_cup = sim.getObjectPosition(cup_handle, sim.handle_world)

    dist = np.sqrt(
        (pos_ee[0] - pos_cup[0])**2 +
        (pos_ee[1] - pos_cup[1])**2 +
        (pos_ee[2] - pos_cup[2])**2
    )

    time_data.append(t - start_time)
    ee_x.append(pos_ee[0]);  ee_y.append(pos_ee[1]);  ee_z.append(pos_ee[2])
    cup_x.append(pos_cup[0]); cup_y.append(pos_cup[1]); cup_z.append(pos_cup[2])
    overlap_flags.append(dist < OVERLAP_THRESHOLD)

    time.sleep(0.05)

sim.stopSimulation()
print(f"บันทึกข้อมูลเสร็จ: {len(time_data)} จุด")

# แปลงเป็น numpy array
t_arr   = np.array(time_data)
ee_x    = np.array(ee_x);   ee_y  = np.array(ee_y);   ee_z  = np.array(ee_z)
cup_x   = np.array(cup_x);  cup_y = np.array(cup_y);  cup_z = np.array(cup_z)
flags   = np.array(overlap_flags)

overlap_times = t_arr[flags]
print(f"\n=== Overlap Detection ===")
print(f"ระยะ Threshold : {OVERLAP_THRESHOLD} m")
print(f"จำนวนช่วงทับซ้อน: {flags.sum()} จุด ({flags.sum()*0.05:.2f} วินาที)")
if len(overlap_times) > 0:
    print(f"ช่วงเวลาทับซ้อน: {overlap_times[0]:.2f}s – {overlap_times[-1]:.2f}s")

# ==========================================
# 3. คำนวณ Workspace Boundary (Convex Hull)
# ==========================================
workspace_pts = np.column_stack([ee_x, ee_y, ee_z])

# ==========================================
# 4. วาดกราฟทั้งหมด
# ==========================================
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Yaskawa GP8 — Workspace & Cup Overlap Analysis', fontsize=15, fontweight='bold')

# ------ (A) 3D Scatter Plot ------
ax3d = fig.add_subplot(2, 3, 1, projection='3d')
ax3d.scatter(ee_x[~flags], ee_y[~flags], ee_z[~flags],
             c='steelblue', s=6, alpha=0.4, label='End-Effector')
ax3d.scatter(ee_x[flags], ee_y[flags], ee_z[flags],
             c='red', s=20, alpha=0.9, label='Overlap Zone', zorder=5)
ax3d.scatter(cup_x, cup_y, cup_z, c='orange', s=8, alpha=0.5, label='Cup Path')
try:
    hull = ConvexHull(workspace_pts)
    for simplex in hull.simplices:
        ax3d.plot(workspace_pts[simplex, 0],
                  workspace_pts[simplex, 1],
                  workspace_pts[simplex, 2], 'b-', alpha=0.08)
except Exception:
    pass
ax3d.set_title('3D Workspace + Cup Path')
ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
ax3d.legend(fontsize=7)

# ------ (B) Top View (XY Plane) ------
ax_xy = fig.add_subplot(2, 3, 2)
ax_xy.scatter(ee_x[~flags], ee_y[~flags], c='steelblue', s=5, alpha=0.4, label='End-Effector')
ax_xy.scatter(ee_x[flags], ee_y[flags], c='red', s=15, alpha=0.9, label='Overlap')
ax_xy.scatter(cup_x, cup_y, c='orange', s=6, alpha=0.5, label='Cup')

# วาด Convex Hull บน XY (Top View area)
try:
    pts_xy = np.column_stack([ee_x, ee_y])
    hull_xy = ConvexHull(pts_xy)
    verts = pts_xy[hull_xy.vertices]
    verts = np.vstack([verts, verts[0]])   # ปิดวง
    ax_xy.fill(verts[:, 0], verts[:, 1], alpha=0.12, color='steelblue', label='Workspace Area')
    ax_xy.plot(verts[:, 0], verts[:, 1], 'b-', linewidth=1.5)
except Exception:
    pass

ax_xy.set_title('Top View (XY) — Workspace Boundary')
ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
ax_xy.legend(fontsize=7); ax_xy.grid(True); ax_xy.set_aspect('equal')

# ------ (C) XZ Plane (Side View) ------
ax_xz = fig.add_subplot(2, 3, 3)
ax_xz.scatter(ee_x[~flags], ee_z[~flags], c='steelblue', s=5, alpha=0.4)
ax_xz.scatter(ee_x[flags], ee_z[flags], c='red', s=15, alpha=0.9, label='Overlap')
ax_xz.scatter(cup_x, cup_z, c='orange', s=6, alpha=0.5, label='Cup')
try:
    pts_xz = np.column_stack([ee_x, ee_z])
    hull_xz = ConvexHull(pts_xz)
    v = pts_xz[hull_xz.vertices]; v = np.vstack([v, v[0]])
    ax_xz.fill(v[:, 0], v[:, 1], alpha=0.12, color='steelblue')
    ax_xz.plot(v[:, 0], v[:, 1], 'b-', linewidth=1.5)
except Exception:
    pass
ax_xz.set_title('Side View (XZ Plane)')
ax_xz.set_xlabel('X (m)'); ax_xz.set_ylabel('Z (m)')
ax_xz.legend(fontsize=7); ax_xz.grid(True); ax_xz.set_aspect('equal')

# ------ (D) YZ Plane ------
ax_yz = fig.add_subplot(2, 3, 4)
ax_yz.scatter(ee_y[~flags], ee_z[~flags], c='steelblue', s=5, alpha=0.4)
ax_yz.scatter(ee_y[flags], ee_z[flags], c='red', s=15, alpha=0.9, label='Overlap')
ax_yz.scatter(cup_y, cup_z, c='orange', s=6, alpha=0.5, label='Cup')
try:
    pts_yz = np.column_stack([ee_y, ee_z])
    hull_yz = ConvexHull(pts_yz)
    v = pts_yz[hull_yz.vertices]; v = np.vstack([v, v[0]])
    ax_yz.fill(v[:, 0], v[:, 1], alpha=0.12, color='steelblue')
    ax_yz.plot(v[:, 0], v[:, 1], 'b-', linewidth=1.5)
except Exception:
    pass
ax_yz.set_title('Front View (YZ Plane)')
ax_yz.set_xlabel('Y (m)'); ax_yz.set_ylabel('Z (m)')
ax_yz.legend(fontsize=7); ax_yz.grid(True); ax_yz.set_aspect('equal')

# ------ (E) Timeline — Distance + Overlap Band ------
ax_tl = fig.add_subplot(2, 3, 5)
dist_arr = np.sqrt((ee_x - cup_x)**2 + (ee_y - cup_y)**2 + (ee_z - cup_z)**2)
ax_tl.plot(t_arr, dist_arr, color='steelblue', linewidth=1.5, label='Distance EE–Cup')
ax_tl.axhline(OVERLAP_THRESHOLD, color='red', linestyle='--', linewidth=1.2,
              label=f'Threshold = {OVERLAP_THRESHOLD} m')

# แรเงาช่วง Overlap
in_overlap = False
start_ol = None
for i, f in enumerate(flags):
    if f and not in_overlap:
        start_ol = t_arr[i]; in_overlap = True
    elif not f and in_overlap:
        ax_tl.axvspan(start_ol, t_arr[i], color='red', alpha=0.25)
        in_overlap = False
if in_overlap:
    ax_tl.axvspan(start_ol, t_arr[-1], color='red', alpha=0.25)

ax_tl.set_title('Timeline: Distance & Overlap Periods')
ax_tl.set_xlabel('Time (s)'); ax_tl.set_ylabel('Distance (m)')
ax_tl.legend(fontsize=7); ax_tl.grid(True)

# ------ (F) Z Position vs Time (ความสูง) ------
ax_z = fig.add_subplot(2, 3, 6)
ax_z.plot(t_arr, ee_z,  color='steelblue', linewidth=1.5, label='EE Height (Z)')
ax_z.plot(t_arr, cup_z, color='orange',    linewidth=1.5, label='Cup Height (Z)')
ax_z.fill_between(t_arr, ee_z, cup_z,
                  where=flags, color='red', alpha=0.3, label='Overlap Period')
ax_z.set_title('Height (Z) vs Time')
ax_z.set_xlabel('Time (s)'); ax_z.set_ylabel('Z (m)')
ax_z.legend(fontsize=7); ax_z.grid(True)

plt.tight_layout()
plt.savefig('workspace_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nบันทึกภาพ: workspace_analysis.png")

# ==========================================
# 5. Animation — Real-time Top View
# ==========================================
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
ax_anim.set_xlim(min(ee_x.min(), cup_x.min()) - 0.1,
                 max(ee_x.max(), cup_x.max()) + 0.1)
ax_anim.set_ylim(min(ee_y.min(), cup_y.min()) - 0.1,
                 max(ee_y.max(), cup_y.max()) + 0.1)
ax_anim.set_title('Animation: Top View (XY) — Real-time Replay')
ax_anim.set_xlabel('X (m)'); ax_anim.set_ylabel('Y (m)')
ax_anim.set_aspect('equal'); ax_anim.grid(True)

# วาด Workspace boundary (static)
try:
    pts_xy = np.column_stack([ee_x, ee_y])
    hull_xy = ConvexHull(pts_xy)
    v = pts_xy[hull_xy.vertices]; v = np.vstack([v, v[0]])
    ax_anim.fill(v[:, 0], v[:, 1], alpha=0.08, color='steelblue')
    ax_anim.plot(v[:, 0], v[:, 1], 'b--', linewidth=1, alpha=0.5, label='Workspace Boundary')
except Exception:
    pass

trail_ee,  = ax_anim.plot([], [], 'b-', alpha=0.3, linewidth=1)
trail_cup, = ax_anim.plot([], [], '-', color='orange', alpha=0.3, linewidth=1)
dot_ee,    = ax_anim.plot([], [], 'bo', markersize=10, label='End-Effector')
dot_cup,   = ax_anim.plot([], [], 'o', color='orange', markersize=10, label='Cup')
time_text  = ax_anim.text(0.02, 0.96, '', transform=ax_anim.transAxes, fontsize=10)
overlap_text = ax_anim.text(0.5, 0.96, '', transform=ax_anim.transAxes,
                             fontsize=12, ha='center', color='red', fontweight='bold')
ax_anim.legend(loc='lower right', fontsize=8)

TRAIL = 30  # จำนวนจุดหาง

def animate(i):
    s = max(0, i - TRAIL)
    trail_ee.set_data(ee_x[s:i+1],  ee_y[s:i+1])
    trail_cup.set_data(cup_x[s:i+1], cup_y[s:i+1])
    dot_ee.set_data([ee_x[i]],  [ee_y[i]])
    dot_cup.set_data([cup_x[i]], [cup_y[i]])
    dot_ee.set_color('red' if flags[i] else 'steelblue')
    time_text.set_text(f't = {t_arr[i]:.2f} s')
    overlap_text.set_text('⚠ OVERLAP!' if flags[i] else '')
    return trail_ee, trail_cup, dot_ee, dot_cup, time_text, overlap_text

anim = FuncAnimation(fig_anim, animate, frames=len(t_arr),
                     interval=50, blit=True, repeat=True)
plt.show()