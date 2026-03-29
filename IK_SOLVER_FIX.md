# IK Solver - เอกสารแก้ไข Critical Fix

## ปัญหาที่เจอ

เมื่อทดสอบ IK solver พบว่า FK และ Simulation EE positions ห่างกันมากๆ:
- ที่ home position: 0.6mm error (ใกล้สมบูรณ์) ✓
- เคลื่อน J1 เพียงอย่างเดียว: 0.7mm error ✓
- เคลื่อน **J3 เพียงอย่างเดียว**: **344.8mm error** ✗✗✗
- เคลื่อน **J5 เพียงอย่างเดียว**: **143.0mm error** ✗
- ที่ IK solution (หัวจับ): **192mm error** ✗✗✗

## สาเหตุ

ทดสอบแต่ละ joint พบว่า **Joint 3 และ Joint 5 หมุนไปทางตรงข้าม ใน CoppeliaSim** เมื่อเทียบกับ DH standard

เมื่อลอง negate angle:
```
J3 +0.3 rad: Error 344.8mm (ปกติ) → 0.7mm (negate) ✓✓✓
J5 +0.3 rad: Error 143.0mm (ปกติ) → 0.6mm (negate) ✓✓✓
```

## วิธีแก้ปัญหา

แก้ไข J3 และ J5 แยกตรวจสอบ เมื่อโปรแกรมแปลง coordinate ระหว่าง:
1. **DH space มาตรฐาน** (ใช้ในการคำนวณ FK)
2. **CoppeliaSim space** (ใช้สำหรับ robot จริง - J3,J5 มีเครื่องหมายตรงข้าม)

### กฎการแปลง Coordinate:

```python
# อ่าน angle จาก simulation (แปลงจาก sim ไป DH):
angles_dh[2] *= -1  # J3: sim → DH
angles_dh[4] *= -1  # J5: sim → DH

# เขียน angle ไป simulation (แปลงจาก DH ไป sim):
angles_sim[2] *= -1  # J3: DH → sim
angles_sim[4] *= -1  # J5: DH → sim
```

### การใช้งานใน IK Solver:

```python
# 1. อ่าน angle จาก sim (แปลงไป DH space)
angles_sim = [sim.getJointPosition(h) for h in joint_handles]
angles_dh = angles_sim.copy()
angles_dh[2] *= -1  # J3
angles_dh[4] *= -1  # J5

# 2. แก้ IK ใน DH space โดยใช้ compute_fk_dh()
# ใช้ Damped Jacobian Pseudo-inverse method
converged, best_angles, error = damped_ik(target_pos, angles_dh, ...)

# 3. ส่ง angle ไป sim (แปลงกลับเป็น sim space)
angles_sim = best_angles.copy()
angles_sim[2] *= -1  # J3
angles_sim[4] *= -1  # J5
for h, angle in zip(joint_handles, angles_sim):
    sim.setJointPosition(h, angle)
```

## ผลการทดสอบ

หลังจากใช้ fix แล้ว IK solver ได้ผลลัพธ์:
- **FK convergence**: 4.456mm ✓
- **Sim position accuracy**: 4.142mm ✓
- **FK-Sim match**: 0.463mm ✓

**ก่อนแก้ไข**: 192mm error ✗
**หลังแก้ไข**: 4.2mm error ✓

### ทดสอบเพิ่มเติม (5 ตำแหน่ง):
- Home + X: 3.5mm ✓
- Home + Y: 4.0mm ✓
- Home + Z: 5.1mm ✓
- Diagonal: 3.9mm ✓
- Grab point: 2.7mm ✓

## Jacobian Method ที่ใช้

ใช้ **Damped Jacobian Pseudo-inverse** สำหรับ IK solver:
```python
# Damped pseudo-inverse:
J_pseudo = J.T @ inv(J @ J.T + λ * I)

# Parameters:
λ (lambda) = 0.01           # Damping factor
ΔQ_max = 3.5 rad/s          # Max velocity limit
Convergence = 0.005m (5mm)  # Threshold
```

**ข้อดี**:
- Fast convergence (7-9 iterations)
- Numerically stable (lambda damping)
- Avoid singularities (velocity limiting)
- Respects joint limits (clamping)

## ไฟล์ที่ปรับปรุง

- `src/main/ik_solver_production.py` - IK solver production-ready พร้อมอธิบาย
- `debug/ik_simple.py` - Debug version พร้อมหมายเหตุ
- `debug/validate_ik.py` - ทดสอบ 5 ตำแหน่งต่างๆ

## บทเรียนสำคัญ

⚠️ **สำคัญ**: ต้องทดสอบ FK ที่หลายๆ ตำแหน่ง ไม่ใช่เพียง home position เท่านั้น

ที่ home position J3=J5=0 ปัญหาถูกซ่อนไว้!

