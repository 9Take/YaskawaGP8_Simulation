"""
Python Script: Get Input Table 1 Data with Visual Overlay
Usage   : python visual_pick_data.py
Dependencies: numpy, opencv-python, coppeliasim-zmqremoteapi-client
"""

import time
import math
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# =========================================================
# 🔌 ส่วนการเชื่อมต่อ CoppeliaSim และระบุตำแหน่ง Scene Hierarchy
# =========================================================
print("🔌 เชื่อมต่อกับ CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

# กำหนด Handles สำหรับอุปกรณ์ต่างๆ (ตรวจสอบชื่อให้ตรงกับ Scene ของคุณ)
vision_sensor_handle = sim.getObject('/Vision_sensor') # กล้องด้านบน
base_handle = sim.getObject('/yaskawa')               # ฐานหุ่นยนต์ (เฟรมอ้างอิงหลักสำหรับค่า Input Table 1)
joints = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)] # ข้อต่อ h1-j6 สำหรับขยับแขนหลบกล้อง

# =========================================================
# 🧠 ส่วนโหลด AI และ OpenCV ArUco Detector
# =========================================================
print("🔄 กำลังโหลด OpenCV ArUco Detector...")
# ใช้ ArUco Dictionary 4x4 สำหรับแก้วน้ำใน Scenario
try:
    # สำหรับ OpenCV เวอร์ชันใหม่ (4.7.x+)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    is_new_opencv = True
except AttributeError:
    # สำหรับ OpenCV เวอร์ชันเก่า
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    is_new_opencv = False

# ประกาศตัวแปรเก็บพิกัดเริ่มต้น
cup_x, cup_y, cup_z = 0, 0, 0
cup_roll, cup_pitch, cup_yaw = 0, 0, 0
cup_found = False

try:
    print("🎬 เริ่มจำลองสถานการณ์...")
    sim.startSimulation()
    
    # 💡 ทริควิศวกร: วาร์ปแขนหุ่นยนต์หลบกล้องเพื่อให้มองเห็นแก้วน้ำชัดๆ และสแกนพิกัดได้แม่นยำที่สุด
    home_angles = [sim.getJointPosition(j) for j in joints] 
    sim.setJointPosition(joints[0], math.radians(-30))  # หมุนเอวไปทางขวา
    sim.setJointPosition(joints[1], math.radians(45))   # ยกแขนขึ้น
    sim.step()
    
    print("👀 กำลังมองหา ArUco ID=0 (แก้วน้ำ)...")
    while True:
        # ดึงภาพจาก Vision Sensor ของ CoppeliaSim
        img_raw, res = sim.getVisionSensorImg(vision_sensor_handle)
        
        # แปลงข้อมูล Buffer เป็น NumPy Array (รูปสี RGB)
        img = np.frombuffer(img_raw, dtype=np.uint8).reshape((res[1], res[0], 3))
        
        # ปรับแต่งรูปภาพ: พลิกภาพ (Sim ให้ภาพกลับหัวมา) และแปลงจาก RGB เป็น BGR สำหรับ OpenCV
        img = cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR)
        
        # แปลงเป็นภาพขาวดำ (Grayscale) เพื่อใช้ในการสแกน ID
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # เริ่มสแกน ArUco ID
        if is_new_opencv:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
        # รีเซ็ตสถานะการเจอแก้วในทุกรอบลูป
        cup_found = False
        
        if ids is not None:
            # วาดกรอบสี่เหลี่ยมสีเขียวและแสดง ID รอบ ArUco ที่เจอ (ค่าเริ่มต้นของ OpenCV)
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            
            # วนลูปหา ArUco ID=0 ที่เราต้องการ
            for i in range(len(ids)):
                current_id = ids[i][0]
                if current_id == 0:  # ✅ เจอ ArUco ID=0 (แก้วน้ำ) แล้ว!
                    cup_found = True
                    try:
                        # --- ใหม่: แสดงจุดหยิบบนรูปภาพ ---
                        # 1. คำนวณพิกัด pixel centroid (จุดศูนย์กลาง) ของ ArUco Marker ID=0
                        corner_points = corners[i][0] # NumPy array (4, 2)
                        centroid_x = int(corner_points[:, 0].mean())
                        centroid_y = int(corner_points[:, 1].mean())

                        # 2. วาดวงกลมสีแดง filled circle ที่จุด Centroid เพื่อบอกว่า "จุดหยิบ" อยู่ตรงไหน
                        cv2.circle(img, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

                        # --- ใหม่: ดึงพิกัดจริงจาก Sim และแสดงผลบนรูปภาพ ---
                        # 3. ระบุชื่อแก้วน้ำใน Scene Hierarchy เป๊ะๆ ตามที่คุณส่งมา
                        cup_handle = sim.getObject('/20cmHighWallL[1]/Cup')
                        
                        # 4. ดึงข้อมูลพิกัดและทิศทางแบบ "เรียลไทม์" เทียบกับฐานหุ่นยนต์ (base_handle)
                        pos = sim.getObjectPosition(cup_handle, base_handle) 
                        ori = sim.getObjectOrientation(cup_handle, base_handle) 
                        
                        cup_x, cup_y, cup_z = pos[0], pos[1], pos[2] # หน่วย: เมตร
                        # แปลงจากเรเดียนเป็นองศา (Degree) สำหรับกรอกตาราง
                        cup_roll, cup_pitch, cup_yaw = math.degrees(ori[0]), math.degrees(ori[1]), math.degrees(ori[2])

                        # 5. เตรียมข้อความข้อมูลพิกัด (formatted) เพื่อแสดงบนรูป
                        # ใช้หน่วย Meters และ Degree
                        text_lines = [
                            "Pick Pt Data:",
                            f"XYZ: [{cup_x:+.3f}, {cup_y:+.3f}, {cup_z:+.3f}]",
                            f"RPY: [{cup_roll:+.1f}, {cup_pitch:+.1f}, {cup_yaw:+.1f}]"
                        ]

                        # 6. แสดงข้อความบนรูปภาพ (Overlay Next to ID label)
                        # ตั้งค่าฟอนต์ สี (ฟ้า Cyan), ความหนา, และระยะห่างบรรทัด
                        x_draw = centroid_x + 20
                        y_draw = centroid_y
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        text_color = (255, 255, 0) # BGR=(255, 255, 0) => Cyan สีฟ้าอ่อน เห็นชัด
                        thickness = 2
                        line_height = 18

                        # cv2.putText ไม่รองรับ \n ในตัว ต้องแยกบรรทัดเอง
                        for index, line in enumerate(text_lines):
                            current_y = y_draw + (index * line_height)
                            cv2.putText(img, line, (x_draw, current_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

                    except Exception as e:
                        print(f"Error retrieving cup coordinates from Sim: {e}")

        # แสดงรูปภาพในหน้าต่าง OpenCV
        cv2.imshow('Get Input Table 1 Data', img)
        
        # กด q บนคีย์บอร์ดเพื่อหยุดลูปและรับค่าสุดท้าย
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 หยุดการสแกน...")
            if cup_found:
                # พิมพ์ผลลัพธ์สุดท้ายออกทาง Terminal (เพื่อความชัวร์ในการก๊อปปี้ไปกรอกตาราง)
                print("\n🎯 === ข้อมูลสำหรับ Input table 1 (หน้า 4) === (meters/degree)")
                print(f"X     : {cup_x:+.4f} m")
                print(f"Y     : {cup_y:+.4f} m")
                print(f"Z     : {cup_z:+.4f} m")
                print(f"Roll  : {cup_roll:+.2f} องศา")
                print(f"Pitch : {cup_pitch:+.2f} องศา")
                print(f"Yaw   : {cup_yaw:+.2f} องศา")
                print("===========================================")
            else:
                print("⚠️ มองไม่เห็น ArUco ID=0 (แก้วน้ำ) กรุณาลองใหม่")
            break
finally:
    # วาร์ปแขนกลับท่าเดิมก่อนปิดเพื่อความเรียบร้อย
    for i, j in enumerate(joints): sim.setJointPosition(j, home_angles[i])
    sim.step()
    sim.stopSimulation()
    cv2.destroyAllWindows()
    print("🔚 ปิดโปรแกรมเรียบร้อย")