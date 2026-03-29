from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib.pyplot as plt

print("📡 Connecting to CoppeliaSim to extract frames & parameters...")
client = RemoteAPIClient()
sim = client.require('sim')

# ==========================================
# 1. GET HANDLES
# ==========================================
joint_handles = [sim.getObjectHandle(f'/yaskawa/joint{i}') for i in range(1, 7)]
ef_handle = sim.getObjectHandle('/yaskawa/gripperEF')
base_handle = sim.getObjectParent(joint_handles[0])

# ==========================================
# 2. FUNCTION ถอดสมการ & พรินต์พารามิเตอร์
# ==========================================
def get_T(handle, ref=-1):
    m = sim.getObjectMatrix(handle, ref)
    return np.array([
        [m[0], m[1], m[2], m[3]],
        [m[4], m[5], m[6], m[7]],
        [m[8], m[9], m[10], m[11]],
        [0,    0,    0,    1]
    ])

def print_kinematic_params(name, handle, ref=-1):
    pos = sim.getObjectPosition(handle, ref)
    ori = sim.getObjectOrientation(handle, ref)
    
    # แปลงเป็น มิลลิเมตร และ องศา เพื่อให้มนุษย์อ่านง่าย
    px, py, pz = np.array(pos) * 1000
    rx, ry, rz = np.degrees(ori)
    
    print(f"{name:<10} | {px:>8.1f} {py:>8.1f} {pz:>8.1f} | {rx:>10.1f} {ry:>10.1f} {rz:>10.1f}")

# ==========================================
# 3. 📊 พิมพ์ตาราง PARAMETERS ลง CONSOLE
# ==========================================
print("\n" + "="*70)
print("🤖 KINEMATIC PARAMETERS (World Frame Coordinates)")
print("="*70)
print(f"{'Frame':<10} | {'X (mm)':<8} {'Y (mm)':<8} {'Z (mm)':<8} | {'Roll (deg)':<10} {'Pitch (deg)':<10} {'Yaw (deg)':<10}")
print("-" * 70)

print_kinematic_params("Base", base_handle)
for i, h in enumerate(joint_handles):
    print_kinematic_params(f"Joint {i+1}", h)
print_kinematic_params("End-Eff", ef_handle)
print("=" * 70 + "\n")

# ==========================================
# 4. วาดแกน 3D (X=Red, Y=Green, Z=Blue)
# ==========================================
def plot_frame(ax, T, name="", scale=0.1):
    origin = T[:3, 3]               
    x_axis = origin + T[:3, 0] * scale  
    y_axis = origin + T[:3, 1] * scale  
    z_axis = origin + T[:3, 2] * scale  

    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r', linewidth=2)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g', linewidth=2)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b', linewidth=2)

    if name:
        ax.text(origin[0], origin[1], origin[2], name, fontsize=10, fontweight='bold')
    return origin

fig = plt.figure(figsize=(16, 10))

# ==========================================
# Main plot (Top-left): Full robot chain
# ==========================================
ax_main = fig.add_subplot(2, 3, (1, 4), projection='3d')
origins = []

T_base = get_T(base_handle)
origins.append(plot_frame(ax_main, T_base, "Base", scale=0.2))

for i, h in enumerate(joint_handles):
    T_j = get_T(h)
    origins.append(plot_frame(ax_main, T_j, f"J{i+1}"))

T_ef = get_T(ef_handle)
origins.append(plot_frame(ax_main, T_ef, "EF", scale=0.15))

origins = np.array(origins)
ax_main.plot(origins[:, 0], origins[:, 1], origins[:, 2], color='k', linestyle='--', linewidth=2, label='Robot Links')

ax_main.set_xlabel('World X')
ax_main.set_ylabel('World Y')
ax_main.set_zlabel('World Z')
ax_main.set_title('Full Robot Kinematic Chain', fontsize=12, fontweight='bold')

max_range = np.array([origins[:,0].max()-origins[:,0].min(), origins[:,1].max()-origins[:,1].min(), origins[:,2].max()-origins[:,2].min()]).max() / 2.0
mid_x = (origins[:,0].max()+origins[:,0].min()) * 0.5
mid_y = (origins[:,1].max()+origins[:,1].min()) * 0.5
mid_z = (origins[:,2].max()+origins[:,2].min()) * 0.5
ax_main.set_xlim(mid_x - max_range, mid_x + max_range)
ax_main.set_ylim(mid_y - max_range, mid_y + max_range)
ax_main.set_zlim(mid_z - max_range, mid_z + max_range)
ax_main.legend()

# ==========================================
# J4 Detail view (Top-right)
# ==========================================
ax_j4 = fig.add_subplot(2, 3, 2, projection='3d')
j4_chain = []
j4_chain.append(plot_frame(ax_j4, get_T(joint_handles[2]), "J3", scale=0.1))
j4_chain.append(plot_frame(ax_j4, get_T(joint_handles[3]), "J4★", scale=0.15))

j4_chain = np.array(j4_chain)
ax_j4.plot(j4_chain[:, 0], j4_chain[:, 1], j4_chain[:, 2], 'r--', linewidth=2.5, label='J3→J4')
ax_j4.set_xlabel('X'); ax_j4.set_ylabel('Y'); ax_j4.set_zlabel('Z')
ax_j4.set_title('J4 Detail View', fontsize=11, fontweight='bold', color='red')
ax_j4.legend(fontsize=9)

# ==========================================
# J5 Detail view (Middle-right)
# ==========================================
ax_j5 = fig.add_subplot(2, 3, 3, projection='3d')
j5_chain = []
j5_chain.append(plot_frame(ax_j5, get_T(joint_handles[4]), "J5★", scale=0.15))
j5_chain.append(plot_frame(ax_j5, get_T(joint_handles[5]), "J6", scale=0.1))

j5_chain = np.array(j5_chain)
ax_j5.plot(j5_chain[:, 0], j5_chain[:, 1], j5_chain[:, 2], 'b--', linewidth=2.5, label='J5→J6')
ax_j5.set_xlabel('X'); ax_j5.set_ylabel('Y'); ax_j5.set_zlabel('Z')
ax_j5.set_title('J5 Detail View', fontsize=11, fontweight='bold', color='blue')
ax_j5.legend(fontsize=9)

# ==========================================
# J4 & J5 Combined view (Bottom-right)
# ==========================================
ax_combined = fig.add_subplot(2, 3, 6, projection='3d')
combined_chain = []
combined_chain.append(plot_frame(ax_combined, get_T(joint_handles[2]), "J3", scale=0.1))
combined_chain.append(plot_frame(ax_combined, get_T(joint_handles[3]), "J4★", scale=0.15))
combined_chain.append(plot_frame(ax_combined, get_T(joint_handles[4]), "J5★", scale=0.15))
combined_chain.append(plot_frame(ax_combined, get_T(joint_handles[5]), "J6", scale=0.1))

combined_chain = np.array(combined_chain)
ax_combined.plot(combined_chain[:, 0], combined_chain[:, 1], combined_chain[:, 2], 'purple',
                 linewidth=3, linestyle='--', label='J3→J4→J5→J6')
ax_combined.set_xlabel('X'); ax_combined.set_ylabel('Y'); ax_combined.set_zlabel('Z')
ax_combined.set_title('J4 & J5 Combined', fontsize=11, fontweight='bold', color='purple')
ax_combined.legend(fontsize=9)

plt.tight_layout()
plt.show()
print("✅ Interactive 3D windows displayed with J4 and J5 detail views")
print("   (rotate with mouse · scroll to zoom)")