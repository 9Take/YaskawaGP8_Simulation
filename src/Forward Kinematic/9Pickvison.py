import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print("กำลังเชื่อมต่อกับ CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

# 1. ระบุ Path แบบเจาะจง ป้องกันการหยิบแก้วผิดใบ!
cup_path = '/20cmHighWallL[1]/Cup'
base_path = '/yaskawa'

try:
    cup_handle = sim.getObject(cup_path)
    base_handle = sim.getObject(base_path)
    
    # 2. ดึงพิกัด (Position) X, Y, Z เทียบกับฐานหุ่นยนต์
    pos = sim.getObjectPosition(cup_handle, base_handle)
    
    # 3. ดึงมุม (Orientation) Roll, Pitch, Yaw
    ori = sim.getObjectOrientation(cup_handle, base_handle)
    
    x, y, z = pos[0], pos[1], pos[2]
    roll, pitch, yaw = math.degrees(ori[0]), math.degrees(ori[1]), math.degrees(ori[2])
    
    print("\n🎯 === ข้อมูลสำหรับตาราง Work 2: Pick (หน้า 40) ===")
    print(f"X : {x:+.4f} m")
    print(f"Y : {y:+.4f} m")
    print(f"Z : {z:+.4f} m")
    print(f"Roll  : {roll:+.2f} องศา")
    print(f"Pitch : {pitch:+.2f} องศา")
    print(f"Yaw   : {yaw:+.2f} องศา")
    print("==================================================\n")

except Exception as e:
    print(f"❌ Error: หาวัตถุไม่เจอ เช็คชื่อ Path อีกทีครับ ({e})")