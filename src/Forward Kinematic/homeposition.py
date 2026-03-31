from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math

print("🔌 เชื่อมต่อกับ CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')

try:
    # 1. ดึง Handles
    base_handle = sim.getObject('/yaskawa')
    
    # 💡 ใส่ชื่อแท่นวาง (Place Table) ของคุณตรงนี้ให้เป๊ะๆ
    place_table_handle = sim.getObject('/20cmHighWallL[0]') 

    # 2. ดึงพิกัดแท่นวาง เทียบกับ "ฐานหุ่นยนต์"
    pos_base = sim.getObjectPosition(place_table_handle, base_handle)
    
    # 3. ดึงพิกัดแท่นวาง เทียบกับ "World Frame" (เพื่อเทียบกับตาเปล่า)
    pos_world = sim.getObjectPosition(place_table_handle, sim.handle_world)

    print("\n🎯 === พิกัดแท่นวาง (กึ่งกลางโต๊ะ) ===")
    print(f"🌍 พิกัดเทียบกับ World (เหมือนที่ตาเห็นในโปรแกรม):")
    print(f"   X: {pos_world[0]:+.4f}, Y: {pos_world[1]:+.4f}, Z: {pos_world[2]:+.4f}")
    
    print(f"\n🤖 พิกัดเทียบกับ ฐานหุ่นยนต์ (ตัวเลขที่ต้องเอาไปใช้จริง!):")
    print(f"   X: {pos_base[0]:+.4f}, Y: {pos_base[1]:+.4f}, Z: {pos_base[2]:+.4f}")
    
    # 💡 เผื่อความสูงให้รอดพื้นโต๊ะตอนวางแก้ว (+0.05 เมตร)
    print(f"\n✅ พิกัด Delivery ที่ต้องเอาไปใส่ในโค้ด IK:")
    print(f"   delivery_xyz = [{pos_base[0]:.4f}, {pos_base[1]:.4f}, {pos_base[2] + 0.05:.4f}]")
    print("===================================")

except Exception as e:
    print(f"⚠️ Error: {e} (เช็คชื่อแท่นวางในวงเล็บ getObject ให้ตรงกับใน Scene ด้วยนะครับ)")