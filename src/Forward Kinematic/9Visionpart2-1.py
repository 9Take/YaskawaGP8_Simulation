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

place_x, place_y, place_z = 0, 0, 0
place_roll, place_pitch, place_yaw = 0, 0, 0
place_found = False

try:
    sim.startSimulation()
    # วาร์ปแขนหลบกล้อง
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
            
        place_found = False
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            floor_corners = [corners[i] for i in range(len(ids)) if ids[i][0] != 0]
            
            if len(floor_corners) == 4:
                place_found = True
                # วาดจุดกึ่งกลางสีแดง
                cx = int(np.mean([np.mean(c[0, :, 0]) for c in floor_corners]))
                cy = int(np.mean([np.mean(c[0, :, 1]) for c in floor_corners]))
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(img, "PLACE ZONE", (cx-45, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # ดึงพิกัด 3D จุดกึ่งกลางแท่น
                px_base, py_base, pz_base = 0, 0, 0
                px_world, py_world, pz_world = 0, 0, 0
                
                for i in range(4):
                    try:
                        marker_h = sim.getObject(f'/20cmHighWallL[0]/arucoMarker[{i}]')
                        
                        # 1. ดึงเทียบกับ Base Frame (เอาไปใช้คำนวณ IK)
                        pos_b = sim.getObjectPosition(marker_h, base_handle) 
                        px_base += pos_b[0]; py_base += pos_b[1]; pz_base += pos_b[2]
                        
                        # 2. ดึงเทียบกับ World Frame (เอาไว้ดูเทียบกับตาเปล่าในโปรแกรม)
                        pos_w = sim.getObjectPosition(marker_h, sim.handle_world)
                        px_world += pos_w[0]; py_world += pos_w[1]; pz_world += pos_w[2]
                        
                    except Exception as e: 
                        print(f"หา Marker {i} ไม่เจอ: {e}")
                
                # เฉลี่ยค่า 4 มุม และชดเชยความสูงแก้วตอนวาง (+0.05m)
                place_x_base = px_base/4
                place_y_base = py_base/4
                place_z_base = (pz_base/4) + 0.05
                
                place_x_world = px_world/4
                place_y_world = py_world/4
                place_z_world = (pz_world/4) + 0.05

        cv2.imshow('Get Input Table 2 Data', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if place_found:
                print("\n🎯 === ข้อมูลสำหรับ Input table 2 (หน้า 7) ===")
                
                print(f"\n🌍 พิกัด World Frame (เทียบกับในจอ CoppeliaSim):")
                print(f"X : {place_x_world:+.4f} m, Y : {place_y_world:+.4f} m, Z : {place_z_world:+.4f} m")
                
                print(f"\n🤖 พิกัด Base Frame (ก๊อปปี้อันนี้ไปป้อนเข้า IK เท่านั้น!):")
                print(f"X : {place_x_base:+.4f} m, Y : {place_y_base:+.4f} m, Z : {place_z_base:+.4f} m")
                
                print("===========================================")
            break
finally:
    for i, j in enumerate(joints): sim.setJointPosition(j, home_angles[i])
    sim.step()
    sim.stopSimulation()
    cv2.destroyAllWindows()