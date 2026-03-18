import time
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# ==========================================
# 1. เชื่อมต่อกับ CoppeliaSim
# ==========================================
client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()  # เริ่ม Simulation
# ระบุชื่อ Object ของแก้วน้ำ (Cup) and  ปลายแขน (End-Effector)

cup_name = '/Cup' 
end_effector_name = '/yaskawa/MicoHand'   # ปลายแขน (End-Effector)
joint_names = [
    '/yaskawa/link_5_b_visual',  # Joint 5 
    '/yaskawa/link_4_r_visual',  # Joint 4
    '/yaskawa/link_3_u_visual',  # Joint 3
    '/yaskawa/link_2_l_visual',  # Joint 2
    '/yaskawa/link_1_s_visual',  # Joint 1
    '/yaskawa/base_link_visual'  # Joint base (Joint 0)
]

def get_handle(name):
    try:
        print(f"กำลังค้นหา Object ชื่อ {name}...")
        if sim.getObject(name) is not None:
            print(f"พบ Object ชื่อ {name} แล้ว!")
        return sim.getObject(name)
    except Exception as e:
        print(f"หา Object ชื่อ {name} ไม่พบ กรุณาตรวจสอบชื่อใน Scene")
        exit()

cup_handle = get_handle(cup_name)
ee_handle = get_handle(end_effector_name)
joint_handles = [get_handle(name) for name in joint_names]
# ==========================================
# 2. ตั้งค่าการบันทึกข้อมูล
# ==========================================
record_duration = 20  # บันทึกข้อมูลเป็นเวลา 10 วินาที

time_data  = []
ee_x, ee_y, ee_z   = [], [], [] #ตำแหน่งของปลายแขน (End-Effector)
cup_x, cup_y, cup_z = [], [], [] #ตำแหน่งของแก้วน้ำ
vel_ee_x, vel_ee_y, vel_ee_z = [], [], [] #ความเร็วของปลายแขน (End-Effector)
overlap_flags = []   # True = ทับซ้อนในช่วงเวลานั้น

JOINT_LIMITS_DEG = [
    (180, -180),   # S
    (-90,   135),  # L
    (-175,  255),  # U
    (-200,  200),  # R
    (-135,  135),  # B
    (-360,  360),  # T
]

start_time = sim.getSimulationTime() # 0 second at the start of simulation
# ==========================================
# 3. บันทึกข้อมูลในขณะที่ Simulation กำลังทำงาน
# ==========================================
while True: 
    t = sim.getSimulationTime() # 0 second at the start of simulation
    if t - start_time > record_duration:
        break

    pos_ee  = sim.getObjectPosition(ee_handle,  sim.handle_world)
    pos_cup = sim.getObjectPosition(cup_handle, sim.handle_world)

    # คำนวณระยะห่างระหว่างปลายแขน (End-Effector) กับแก้วน้ำ (Cup)
    dist = np.sqrt(
        (pos_ee[0] - pos_cup[0])**2 +
        (pos_ee[1] - pos_cup[1])**2 +
        (pos_ee[2] - pos_cup[2])**2
    )

    time_data.append(t-start_time)
    ee_x.append(pos_ee[0]);   ee_y.append(pos_ee[1]);   ee_z.append(pos_ee[2])
    cup_x.append(pos_cup[0]); cup_y.append(pos_cup[1]); cup_z.append(pos_cup[2])

    time.sleep(0.05)

sim.stopSimulation()  # หยุด Simulation หลังจากบันทึกข้อมูลเสร็จ


