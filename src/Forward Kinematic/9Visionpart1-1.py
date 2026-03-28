import numpy as np
import cv2
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

vision_sensor_handle = sim.getObject('/Vision_sensor')
base_handle = sim.getObject('/yaskawa') 
joints = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]

# โหลด AI
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    is_new_opencv = True
except AttributeError:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    is_new_opencv = False

cup_x, cup_y, cup_z = 0, 0, 0
cup_roll, cup_pitch, cup_yaw = 0, 0, 0
cup_found = False

try:
    sim.startSimulation()
    # วาร์ปแขนหลบกล้องให้เห็นแก้วชัดๆ
    home_angles = [sim.getJointPosition(j) for j in joints] 
    sim.setJointPosition(joints[0], math.radians(-30))  
    sim.setJointPosition(joints[1], math.radians(45))   
    sim.step()
    
    while True:
        img_raw, res = sim.getVisionSensorImg(vision_sensor_handle)
        img = np.frombuffer(img_raw, dtype=np.uint8).reshape((res[1], res[0], 3))
        img = cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if is_new_opencv:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
        cup_found = False
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            for i in range(len(ids)):
                if ids[i][0] == 0:  # ล็อกเป้าที่ ID=0 (แก้วน้ำ)
                    cup_found = True
                    try:
                        # 💡 เปลี่ยน Path ตามรูป Scene Hierarchy ที่คุณส่งมาเป๊ะๆ!
                        cup_handle = sim.getObject('/20cmHighWallL[1]/Cup')
                        
                        # ดึง Position และ Orientation เทียบกับฐานหุ่นยนต์ (base_handle)
                        pos = sim.getObjectPosition(cup_handle, base_handle) 
                        ori = sim.getObjectOrientation(cup_handle, base_handle) 
                        
                        cup_x, cup_y, cup_z = pos[0], pos[1], pos[2]
                        # แปลงจากเรเดียนเป็นองศา (Degree)
                        cup_roll, cup_pitch, cup_yaw = math.degrees(ori[0]), math.degrees(ori[1]), math.degrees(ori[2])
                    except Exception as e:
                        print("Error:", e)

        cv2.imshow('Get Input Table 1 Data', img)
        
        # กด q เพื่อโชว์ผลลัพธ์
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if cup_found:
                print("\n🎯 === ข้อมูลสำหรับ Input table 1 (หน้า 4) ===")
                print(f"X     : {cup_x:+.4f} m")
                print(f"Y     : {cup_y:+.4f} m")
                print(f"Z     : {cup_z:+.4f} m")
                print(f"Roll  : {cup_roll:+.2f} องศา")
                print(f"Pitch : {cup_pitch:+.2f} องศา")
                print(f"Yaw   : {cup_yaw:+.2f} องศา")
                print("===========================================")
            else:
                print("⚠️ มองไม่เห็น ArUco ID=0 (แก้วน้ำ)")
            break
finally:
    # วาร์ปแขนกลับท่าเดิม
    for i, j in enumerate(joints): sim.setJointPosition(j, home_angles[i])
    sim.step()
    sim.stopSimulation()
    cv2.destroyAllWindows()