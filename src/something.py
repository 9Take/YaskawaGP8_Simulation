import time
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================
# 1. เชื่อมต่อกับ CoppeliaSim
# ==========================================
client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()  # เริ่ม Simulation
# ระบุชื่อ Object ของแก้วน้ำ (แก้ '/Cup' ให้ตรงกับชื่อในโปรแกรมของคุณ)
cup_name = '/Cup' 
try:
    cup_handle = sim.getObject(cup_name)
except Exception as e:
    print(f"หา Object ชื่อ {cup_name} ไม่พบ กรุณาตรวจสอบชื่อใน Scene")
    exit()

