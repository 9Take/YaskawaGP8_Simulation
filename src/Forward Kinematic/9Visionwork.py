import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print("กำลังเชื่อมต่อกับ CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

# 1. เชื่อมต่อกล้อง
sensor_path = '/Vision_sensor'
vision_sensor_handle = sim.getObject(sensor_path)
print(f"✅ เชื่อมต่อกับ Vision Sensor สำเร็จ! (Handle: {vision_sensor_handle})")

# 2. 💡 แก้ไขตรงนี้: รับค่า resolution แบบตรงๆ ไม่ต้องมี result มารับแล้ว
resolution = sim.getVisionSensorResolution(vision_sensor_handle)
width = resolution[0]
height = resolution[1]
print(f"🖼️ ความละเอียดกล้อง: {width} x {height} Pixels")

print("\n--- เริ่มดึงภาพ Real-time (กด 'q' บนหน้าต่างรูปภาพเพื่อปิด) ---")

# --- ตั้งค่าระบบอ่าน ArUco (ดักไว้ให้รองรับทั้ง OpenCV เก่าและใหม่) ---
try:
    # สำหรับ OpenCV เวอร์ชันใหม่ (4.7+)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    is_new_opencv = True
except AttributeError:
    # สำหรับ OpenCV เวอร์ชันเก่า
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    is_new_opencv = False

print("\n--- เริ่มระบบ AI ตรวจจับ ArUco (กด 'q' เพื่อปิด) ---")

try:
    # สั่งให้ CoppeliaSim เริ่มเล่นอัตโนมัติ (จะได้ไม่ต้องสลับไปกดเอง)
    sim.startSimulation()
    
    while True:
        img_raw, res = sim.getVisionSensorImg(vision_sensor_handle)
        img = np.frombuffer(img_raw, dtype=np.uint8)
        img.shape = (res[1], res[0], 3)
        img = cv2.flip(img, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 💡 ขั้นตอนของ AI: 1. แปลงภาพเป็นขาวดำเพื่อให้อ่านง่ายขึ้น
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 💡 ขั้นตอนของ AI: 2. สแกนหา ArUco
        if is_new_opencv:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
        # 💡 ขั้นตอนของ AI: 3. ถ้าเจอ ArUco ให้วาดกรอบสีและพิมพ์ ID โชว์
        if ids is not None:
            # วาดกรอบสี่เหลี่ยมรอบๆ Marker
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            
            # ปริ้นท์บอกใน Console ว่าเจอ ID อะไรบ้าง
            ids_flatten = ids.flatten()
            # print(f"👁️ มองเห็น ArUco IDs: {ids_flatten}") # ปิดไว้ก่อนเดี๋ยวรก Console
            
        cv2.imshow('Robot Live View (AI Mode)', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    sim.stopSimulation() # สั่งหยุดซิมูเลชันตอนปิดโปรแกรมด้วย
    cv2.destroyAllWindows()
    print("🔚 ปิดโปรแกรมดึงภาพสำเร็จ")