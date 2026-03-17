import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# หากมีปัญหาเรื่อง 3D ในเวอร์ชันของคุณ สามารถลบบรรทัด Axes3D ด้านล่างนี้ออกได้
from mpl_toolkits.mplot3d import Axes3D 

# ==========================================
# 1. พารามิเตอร์หุ่นยนต์ Yaskawa GP8 (ใช้ L1-L7 เหมือนเดิม)
# ==========================================
L1 = 330      
L2 = 345      
L3 = 40       
L4 = 40       
L5 = 340      
L6 = 80       
L7 = 161.33   

# ==========================================
# 2. ฟังก์ชันคำนวณ Classical DH Matrix (Standard DH)
# ==========================================
# เรียงลำดับพารามิเตอร์ตามมาตรฐาน: Theta, d, a, alpha
def std_dh(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def get_joint_positions(t1, t2, t3, t4, t5, t6):
    t1, t2, t3 = np.radians(t1), np.radians(t2), np.radians(t3)
    t4, t5, t6 = np.radians(t4), np.radians(t5), np.radians(t6)
    
    pi_half = np.pi / 2
    
    T0 = np.eye(4) # ฐาน (Base)
    
    # ==========================================
    # ⚠️ นำตาราง Classical DH ของคุณมาใส่ตรงนี้ ⚠️
    # รูปแบบการใส่: std_dh( theta, d, a, alpha )
    # ==========================================
    # (โค้ดด้านล่างนี้เป็นเพียงตัวอย่างรอการแก้ไขจากคุณ)
    T1 = T0 @ std_dh(t1, L1, L3,  pi_half)     
    T2 = T1 @ std_dh(t2,  0, L2, 0)          
    T3 = T2 @ std_dh(t3,  0, L4, pi_half)    
    T4 = T3 @ std_dh(t4, L5,  0, -pi_half)   
    T5 = T4 @ std_dh(t5,  0,  0, pi_half)    
    T6 = T5 @ std_dh(t6, L6+L7, 0, 0)        
    
    # ดึงเฉพาะตำแหน่ง X, Y, Z ของทุกจุดเชื่อมต่อมาเรียงกัน
    positions = np.vstack([
        T0[:3, 3], T1[:3, 3], T2[:3, 3], 
        T3[:3, 3], T4[:3, 3], T5[:3, 3], T6[:3, 3]
    ])
    return positions

# ==========================================
# 3. สร้างหน้าต่างกราฟิก 3 มิติ และ Slider (เหมือนเดิม)
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.35) 

init_angles = [0, 0, 0, 0, 0, 0]
pts = get_joint_positions(*init_angles)
line, = ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o-', lw=4, markersize=8, color='b')
ax.plot([0], [0], [0], 'ko', markersize=10) 

limit = 1000
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([0, 1200])
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Yaskawa GP8 Kinematics Viewer (Classical DH)')

axcolor = 'lightgoldenrodyellow'
slider_axes = [plt.axes([0.15, 0.25 - i*0.04, 0.65, 0.03], facecolor=axcolor) for i in range(6)]
sliders = []
names = ['Theta 1', 'Theta 2', 'Theta 3', 'Theta 4', 'Theta 5', 'Theta 6']

for i, sax in enumerate(slider_axes):
    s = Slider(sax, names[i], -180.0, 180.0, valinit=init_angles[i])
    sliders.append(s)

def update(val):
    angles = [s.val for s in sliders]
    new_pts = get_joint_positions(*angles)
    line.set_data(new_pts[:, 0], new_pts[:, 1])
    line.set_3d_properties(new_pts[:, 2])
    fig.canvas.draw_idle()

for s in sliders:
    s.on_changed(update)

plt.show()