import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
Pi = math.pi

# create time
step = 380
dt = 0.05 # s
t = []
tmp = 0
for x in range(step):
    t.append(tmp)
    tmp = tmp + dt

# Design Trajectory of the gripperEF
# p0: จุดเริ่มต้น (Home)
p0x, p0y, p0z = 0.64, 0.00023, 0.71507

# p1: จุดเตรียมหยิบ (Approach) - ให้มาดักรอเหนือแก้ว ลอยขึ้น 10cm ที่เวลา t=8.10
p1x, p1y, p1z = -0.050, -0.601, 0.474 

# p2: จุดหยิบแก้ว (Pick) - โฉบลงมาหนีบแก้วเป๊ะๆ ที่เวลา t=10.10
p2x, p2y, p2z = -0.050, -1.001, 0.374

# p3: จุดยกแก้วขึ้น (Depart) - หยิบเสร็จแล้วดึงขึ้น พร้อมตีคู่ไปกับสายพาน ที่เวลา t=12.10
p3x, p3y, p3z = -0.050, -1.401, 0.474

# p4: จุดวางแก้ว (Place) - วนกลับมาวางจุด Home ที่เวลา t=17.10
p4x, p4y, p4z = 0.64, 0.00023, 0.71507

# ตั้งค่าช่วงเวลา (Time Intervals) ให้สอดคล้องกับตำแหน่ง
t01 = 8.10  # วิ่งจาก Home มาดักรอ ใช้เวลา 8.10 วิ (มาถึงตอน t=8.10)
t12 = 2.00  # วิ่งตีคู่พร้อมโฉบลงมา ใช้เวลา 2.00 วิ (มาถึงตอน t=10.10)
t23 = 2.00  # วิ่งตีคู่พร้อมยกขึ้น ใช้เวลา 2.00 วิ (มาถึงตอน t=12.10)
t34 = 5.00  # วิ่งกลับไปจุดวาง ใช้เวลา 5.00 วิ (มาถึงตอน t=17.10)

vx01, vy01, vz01 = (p1x-p0x)/t01, (p1y-p0y)/t01, (p1z-p0z)/t01
vx12, vy12, vz12 = (p2x-p1x)/t12, (p2y-p1y)/t12, (p2z-p1z)/t12
vx23, vy23, vz23 = (p3x-p2x)/t23, (p3y-p2y)/t23, (p3z-p2z)/t23
vx34, vy34, vz34 = (p4x-p3x)/t34, (p4y-p3y)/t34, (p4z-p3z)/t34

gripperEFx = []
gripperEFy = []
gripperEFz = []
gripperEFvx = []
gripperEFvy = []
gripperEFvz = []

px = p0x
py = p0y
pz = p0z
px_old = px
py_old = py
pz_old = pz

for x in range(step):
    # ปริ้นท์เช็คเวลา (ถ้าขี้เกียจดูให้ใส่ # คอมเมนต์ไว้ได้ครับ)
    # print(t[x]) 
    
    if 12.10 <= t[x]:       # ช่วงที่ 4: จากวินาที 12.10 เป็นต้นไป (วิ่งกลับจุด Home)
        px = px + vx34*dt
        py = py + vy34*dt
        pz = pz + vz34*dt
    
    elif 10.10 <= t[x]:     # ช่วงที่ 3: วินาที 10.10 - 12.10 (หนีบแก้วแล้วดึงขึ้น พร้อมตีคู่สายพาน)
        px = px + vx23*dt
        py = py + vy23*dt
        pz = pz + vz23*dt
    
    elif 8.10 <= t[x]:      # ช่วงที่ 2: วินาที 8.10 - 10.10 (วิ่งตีคู่สายพาน พร้อมโฉบลงมาหนีบ)
        px = px + vx12*dt
        py = py + vy12*dt
        pz = pz + vz12*dt
    
    elif 0.00 <= t[x]:      # ช่วงที่ 1: วินาที 0.00 - 8.10 (ออกจาก Home ไปดักรอเหนือแก้ว)
        px = px + vx01*dt
        py = py + vy01*dt
        pz = pz + vz01*dt 

    gripperEFx.append(px)
    gripperEFy.append(py)
    gripperEFz.append(pz)
    
    gripperEFvx.append((px - px_old)/dt)
    gripperEFvy.append((py - py_old)/dt)
    gripperEFvz.append((pz - pz_old)/dt)
    
    px_old = px
    py_old = py
    pz_old = pz
    
# plot trajectory
plt.figure # plot linear position

plt.subplot(2,3,1)
plt.title('t and linear position x of the gripperEF')
plt.xlabel('t(s)')
plt.ylabel('m')
plt.legend()
plt.plot(t,gripperEFx, 'b')

plt.subplot(2,3,2)
plt.title('t and linear position y of the gripperEF')
plt.xlabel('t(s)')
plt.ylabel('m')
plt.legend()
plt.plot(t,gripperEFy, 'b')

plt.subplot(2,3,3)
plt.title('t and linear position z of the gripperEF')
plt.xlabel('t(s)')
plt.ylabel('m')
plt.legend()
plt.plot(t,gripperEFz, 'b')

plt.subplot(2,3,4)
plt.title('t and linear velocity x of the gripperEF')
plt.xlabel('t(s)')
plt.ylabel('m/s')
plt.legend()
plt.plot(t,gripperEFvx, 'b')

plt.subplot(2,3,5)
plt.title('t and linear velocity y of the gripperEF')
plt.xlabel('t(s)')
plt.ylabel('m/s')
plt.legend()
plt.plot(t,gripperEFvy, 'b')

plt.subplot(2,3,6)
plt.title('t and linear velocity z of the gripperEF')
plt.xlabel('t(s)')
plt.ylabel('m/s')
plt.legend()
plt.plot(t,gripperEFvz, 'b')

plt.show()
