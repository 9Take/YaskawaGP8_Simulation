from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

# ดึงชื่อ Handle (ต้องเช็คชื่อใน Scene Hierarchy ของคุณว่าชื่ออะไร)
cup_handle = sim.getObject('/Cup')          # ชื่อแก้วน้ำ
robot_base_handle = sim.getObject('/yaskawa') # ชื่อฐานหุ่นยนต์ (ตัวล่างสุดที่ไม่ขยับ)

print("▶️ เริ่มจำลองเพื่อหาพิกัดที่วินาทีที่ 7.5...")
sim.startSimulation()

# ปล่อยให้สายพานวิ่งไปจนถึงวินาทีที่ 7.5
while sim.getSimulationTime() < 7.5:
    sim.step()

# 🎯 คำสั่งสำคัญ: หาพิกัดของแก้ว 'เทียบกับ' ฐานหุ่นยนต์
# (ตัวเลขที่ได้จะมองว่าฐานหุ่นยนต์คือจุด 0,0,0)
rel_pos = sim.getObjectPosition(cup_handle, robot_base_handle)

sim.stopSimulation()

print("\n--- พิกัดที่ต้องนำไปใส่ในโค้ด Jacobian ---")
print(f"px_target = {rel_pos[0]*1000:.2f}") # คูณ 1000 เพื่อแปลงเป็น mm
print(f"py_target = {rel_pos[1]*1000:.2f}")
print(f"pz_pick   = {rel_pos[2]*1000:.2f}")