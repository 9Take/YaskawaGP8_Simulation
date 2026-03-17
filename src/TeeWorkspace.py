import numpy as np
import matplotlib.pyplot as plt

# สมมติว่านี่คือสมการการเคลื่อนที่ของแก้วที่คุณได้มาจากสเตปแรก
# P(t) = V*t + P0 (คุณต้องเอาตัวเลขจริงที่คุณหาได้มาแทนที่ตรงนี้นะครับ)
t_steps = np.linspace(0, 20, 200) # จำลองเวลาตั้งแต่ 0 ถึง 20 วินาที
x_cup = 0.0 * t_steps + 800       # สมมติแก้ววิ่งมาที่ X = 800 mm
y_cup = 100.0 * t_steps - 1000    # สมมติวิ่งมาตามแกน Y ด้วยความเร็ว 100 mm/s
z_cup = 0.0 * t_steps + 150       # สมมติความสูงแก้วบนสายพานคือ 150 mm

# พิกัดหัวไหล่ของหุ่นยนต์ (อ้างอิงจาก Z = L1 ใน v1.py)
L1 = 330.0
robot_shoulder = np.array([0, 0, L1])

# คำนวณระยะห่างระหว่างแก้วกับหัวไหล่หุ่นยนต์ในทุกๆ ช่วงเวลา t
distances = np.sqrt(x_cup**2 + y_cup**2 + (z_cup - L1)**2)

# กำหนดระยะเอื้อมสูงสุด (Safety Margin แล้ว)
R_max_safe = 850.0 

# พล็อตกราฟเพื่อหา Time Window
plt.figure(figsize=(10, 5))
plt.plot(t_steps, distances, 'b-', linewidth=2, label='Distance to Cup')
plt.axhline(y=R_max_safe, color='r', linestyle='--', linewidth=2, label='Max Reach (Safe Limit)')

# แรเงาช่วงเวลาที่หุ่นยนต์สามารถเอื้อมถึง (Distance < R_max_safe)
valid_indices = np.where(distances <= R_max_safe)[0]
if len(valid_indices) > 0:
    t_enter = t_steps[valid_indices[0]]
    t_exit = t_steps[valid_indices[-1]]
    plt.axvspan(t_enter, t_exit, color='green', alpha=0.2, label=f'Pick Window: {t_enter:.1f}s - {t_exit:.1f}s')
    print(f"แก้วเข้ามาในระยะตอนวินาทีที่: {t_enter:.2f}")
    print(f"แก้วหลุดระยะตอนวินาทีที่: {t_exit:.2f}")
else:
    print("แก้วไม่เคยเข้ามาในระยะเอื้อมของหุ่นยนต์เลย! (ต้องเลื่อนสายพานเข้าใกล้ขึ้น)")

plt.title('Distance from Robot to Moving Cup vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance (mm)')
plt.legend()
plt.grid(True)
plt.show()