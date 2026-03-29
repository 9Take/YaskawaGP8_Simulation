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
                px, py, pz = 0, 0, 0
                for i in range(4):
                    try:
                        marker_h = sim.getObject(f'/20cmHighWallL[0]/arucoMarker[{i}]')
                        pos = sim.getObjectPosition(marker_h, base_handle) 
                        px += pos[0]; py += pos[1]; pz += pos[2]
                    except: pass
                
                # ชดเชยความสูงแก้วตอนวาง (+0.05m) ไม่ให้กระแทกแผ่น
                place_x, place_y, place_z = px/4, py/4, (pz/4) + 0.05
                
                # ดึงมุมเอียงของแผ่นป้ายอ้างอิง (เพื่อหา Roll, Pitch, Yaw)
                try:
                    ref_marker = sim.getObject('/20cmHighWallL[0]/arucoMarker[0]')
                    ori = sim.getObjectOrientation(ref_marker, base_handle)
                    place_roll, place_pitch, place_yaw = math.degrees(ori[0]), math.degrees(ori[1]), math.degrees(ori[2])
                except: pass

        cv2.imshow('Get Input Table 2 Data', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if place_found:
                print("\n🎯 === ข้อมูลสำหรับ Input table 2 (หน้า 7) ===")
                print(f"X     : {place_x:+.4f} m")
                print(f"Y     : {place_y:+.4f} m")
                print(f"Z     : {place_z:+.4f} m")
                print(f"Roll  : {place_roll:+.2f} องศา")
                print(f"Pitch : {place_pitch:+.2f} องศา")
                print(f"Yaw   : {place_yaw:+.2f} องศา")
                print("===========================================")
            else:
                print("⚠️ สแกนไม่พบเป้าหมาย (ต้องเห็น ID 1,2,3,4 ครบ)")
            break
finally:
    for i, j in enumerate(joints): sim.setJointPosition(j, home_angles[i])
    sim.step()
    sim.stopSimulation()
    cv2.destroyAllWindows()