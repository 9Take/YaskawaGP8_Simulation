from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

# ดึง Handle ของแก้วน้ำและฐานหุ่นยนต์
cup = sim.getObject('/Cup')
robot_base = sim.getObject('/yaskawa') # ใช้ตัวฐานของหุ่นเป็นจุดอ้างอิง

print("▶️ กำลังเริ่ม Simulation และรอให้สายพานเลื่อนไปที่วินาทีที่ 10.10...")
sim.startSimulation()

# ปล่อยให้ซิมรันไปเรื่อยๆ จนถึงวินาทีที่เราต้องการหยิบ
while sim.getSimulationTime() < 5.00:
    sim.step()

# ดึงพิกัดแก้วน้ำ "เทียบกับฐานหุ่นยนต์" (ตรงนี้แหละคือคีย์สำคัญ!)
rel_pos = sim.getObjectPosition(cup, robot_base)

print(f"\n🎯 พิกัดแก้วน้ำที่ถูกต้อง (เทียบกับฐานหุ่นยนต์):")
print(f"X = {rel_pos[0]*1000:.2f} mm")
print(f"Y = {rel_pos[1]*1000:.2f} mm")
print(f"Z = {rel_pos[2]*1000:.2f} mm")

sim.stopSimulation()
print("\n⏹️ สิ้นสุดการค้นหาพิกัด")