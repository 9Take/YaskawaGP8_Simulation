import math as m
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

print("🔍 กำลังดึงค่า Home Position ปัจจุบันจากหุ่นยนต์...")

home_angles_deg = []
for i in range(1, 7):
    # ดึง Handle ของแต่ละ Joint
    joint_handle = sim.getObject(f'/yaskawa/joint{i}')
    # อ่านค่าองศา (หน่วยเป็นเรเดียน)
    rad = sim.getJointPosition(joint_handle)
    # แปลงเป็นองศา (Degree)
    deg = m.degrees(rad)
    home_angles_deg.append(round(deg, 2))
    
print("\n🎯 ได้ค่า Home Position แล้ว! ก๊อปปี้บรรทัดล่างนี้ไปใส่ในโค้ดหลักได้เลย:")
print(f"home_angles = {home_angles_deg}")