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
    '/yaskawa/link_5_b_visual',  # Joint 1 
    '/yaskawa/link_4_r_visual',  # Joint 2
    '/yaskawa/link_3_u_visual',  # Joint 3
    '/yaskawa/link_2_l_visual',  # Joint 4
    '/yaskawa/link_1_s_visual',  # Joint 5
    '/yaskawa/base_link_visual'  # Joint base (Joint 6)
]

try:
    cup_handle = sim.getObject(cup_name)
except Exception as e:
    print(f"หา Object ชื่อ {cup_name} ไม่พบ กรุณาตรวจสอบชื่อใน Scene")
    exit()

try:
    end_effector_handle = sim.getObject(end_effector_name)
except Exception as e:
    print(f"หา Object ชื่อ {end_effector_name} ไม่พบ กรุณาตรวจสอบชื่อใน Scene")
    exit()

joint_handles = []
for joint_name in joint_names:
    try:
        joint_handle = sim.getObject(joint_name)
        joint_handles.append(joint_handle)
    except Exception as e:
        print(f"หา Object ชื่อ {joint_name} ไม่พบ กรุณาตรวจสอบชื่อใน Scene")
        exit()

record_duration = 20  # บันทึกข้อมูลเป็นเวลา 10 วินาที
start_time = sim.getSimulationTime() # 0 second at the start of simulation

while True: 
    t = sim.getSimulationTime() # 0 second at the start of simulation
    if t - start_time > record_duration:
        break
    time.sleep(0.05)

sim.stopSimulation()  # หยุด Simulation หลังจากบันทึกข้อมูลเสร็จ


