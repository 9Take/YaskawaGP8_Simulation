# Yaskawa GP8 Pick-and-Place Project - Current Status

**Last Updated:** March 30, 2026  
**Project Phase:** DH Table Validation & Kinematics Implementation

---

## 📋 What We've Been Working On

### Phase Overview
After weeks of kinematic debugging, we finally identified and validated the **correct DH parameters** for the Yaskawa GP8 robot. The journey involved:

1. **Initial extraction** of robot link lengths from CoppeliaSim simulation
2. **Frame verification** testing to ensure kinematic frames are actual control frames
3. **FK/IK development** with multiple iterations to get accurate models
4. **DH parameter validation** - discovering the key issue: **q2 angle offset (-π/2)**
5. **FK accuracy confirmation** - TF Matrix.py achieving **0.24mm error** vs simulation

---

## ✅ Validated DH Parameters (CORRECT)

The **Yaskawa GP8 DH table** with actual measured values:

```
Link | α (rad)    | a (m)    | d (m)    | θ offset  | Note
-----|------------|----------|----------|-----------|------
  1  | -π/2       | 0.0400   | 0.3300   | 0         | Base link
  2  |  0.0       | 0.3450   | 0.0000   | -π/2      | ⭐ CRITICAL OFFSET!
  3  | -π/2       | 0.0400   | 0.0000   | 0         | 
  4  |  π/2       | 0.0      | 0.3400   | 0         | Wrist
  5  | -π/2       | 0.0      | 0.0000   | 0         | 
  6  |  0.0       | 0.0      | 0.2413   | 0         | End-effector
```

**Base Offset:** `[0.0187, 0, 0]` m (robot base to link 1)

### Key Discovery
- Joint 2 has a **-π/2 offset** in the DH convention
- This comes from the joint coordinate definition in CoppeliaSim
- Without this offset: 680mm-1392mm FK errors
- With correct offset: **0.24mm error** ✓

---

## 📁 Current Project Files

### Core Files (in `src/main/`)

#### 1. **TF Matrix.py** ⭐ (VERIFIED WORKING)
- **Purpose:** Compute forward kinematics using DH parameters
- **Accuracy:** **0.24mm error** vs CoppeliaSim
- **How it works:**
  - Reads live joint angles from simulation
  - Computes DH matrices for each link
  - Chains transformations: `T_total = T_Base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56`
  - Compares with simulation using `getObjectMatrix(ef_handle, sim.handle_world)`
  - Prints matrices and position error

**Output Example (at home position):**
```
DH Position:  [0.6400, 0.0000, 0.7150]
Sim Position: [0.6400, 0.0002, 0.7151]
Position Error = 0.2407 mm ✓
```

#### 2. **getcup trajectory.py**
- **Purpose:** Extract conveyor cup trajectory for pick operation
- **How it works:**
  - Records cup position for 600 time steps (30 seconds at 50ms/step)
  - Analyzes which steps have cup in robot's reachable workspace
  - Recommends optimal grab point (middle of reachable range)
  - Saves trajectory data to `trajectory_cup_data.npz`

**Input:** CoppeliaSim with moving conveyor + cup  
**Output:** `trajectory_cup_data.npz` containing:
- Cup position: `px`, `py`, `pz` (600 points each)
- Cup orientation: `ox`, `oy`, `oz` (Euler angles)
- Cup velocity: `vx`, `vy`, `vz`
- Grab step recommendation
- Robot base position
- Place conveyor position

---

## 🔧 Implementation Details

### DH Matrix Computation (from TF Matrix.py)

```python
def dh_matrix(alpha, a, d, theta):
    """Standard Denavit-Hartenberg transformation"""
    return np.array([
        [cos(θ), -sin(θ)cos(α),  sin(θ)sin(α), a·cos(θ)],
        [sin(θ),  cos(θ)cos(α), -cos(θ)sin(α), a·sin(θ)],
        [0,       sin(α),        cos(α),       d        ],
        [0,       0,             0,            1        ]
    ])
```

### Forward Kinematics Chain

```python
T_Base @ dh_matrix(...q1...) @ dh_matrix(...q2-π/2...) @ ... @ dh_matrix(...q6...)
```

**Critical Point:** `q2` is computed as `q2 - π/2` in the DH formula!

---

## 📊 Validation Results

### TF Matrix.py Test Run
```
====== DH Table (Yaskawa GP8) ======
| Link | alpha  | a     | d     | theta       |
|  1   |-1.5708 | 0.040 | 0.330 | q1 =  0.000 |
|  2   | 0.0000 | 0.345 | 0.000 | q2 = -1.571 |
|  3   |-1.5708 | 0.040 | 0.000 | q3 =  0.000 |
|  4   | 1.5708 | 0.000 | 0.340 | q4 =  0.000 |
|  5   |-1.5708 | 0.000 | 0.000 | q5 =  0.000 |
|  6   | 0.0000 | 0.000 | 0.241 | q6 =  0.000 |

DH Position:  [0.6400, 0.0000, 0.7150]
Sim Position: [0.6400, 0.0002, 0.7151]
✓ Position Error = 0.2407 mm
```

### getcup trajectory.py Test Run
```
EF home: [0.6156, 0.0000, -0.6406]
Cup start: [0.4554, -0.4886, 0.0050]
Robot base: [0.0000, 0.0000, 0.0000]

[INFO] Cup is in reach from step 150 to 450
[RECOMMEND] Best step to grab: 300
Position: [0.5000, -0.4887, 0.4261]
Time: 15.00 seconds

✓ Data saved to trajectory_cup_data.npz
```

---

## 🎯 Next Tasks

### Immediate (Priority 1)
- [ ] **IK Solver with Correct DH** - Implement Jacobian-based IK using validated DH table
  - Input: Target EE position `[x, y, z]`
  - Output: Joint angles `[q1, q2, q3, q4, q5, q6]`
  - Validation: FK(q_ik) should match target < 5mm error
  
  **Key Issue Found:** When applying IK solution angles to simulation, EE position doesn't update to target (coordinate frame mismatch needs investigation)

### Phase 2 (Priority 2)
- [ ] **Trajectory Generation** - Quintic polynomial spline from home → grab position
  - Use validated IK solver to find joint angles for grab point
  - Generate smooth 5s trajectory with Quintic interpolation
  - Apply to simulation and verify arm reaches cup

### Phase 3 (Priority 3)
- [ ] **Gripper Control** - Close gripper at grab point
- [ ] **Place Motion** - Move cup to drop location with Quintic trajectory
- [ ] **Full Pick-and-Place Cycle** - Home → Grab → Lift → Place → Return Home

---

## 🔍 Known Issues

### IK Solver Challenge
- IK algorithm converges correctly (error drops to ~4mm)
- Converged joint angles calculated and retrieved ✓
- BUT: When angles applied to simulation, EE doesn't reach target position
- **Root Cause Investigation Needed:** Possible coordinate frame mismatch or simulation joint control behavior

### Solution Approach
- Compare multiple EE position reading methods:
  - `getObjectPosition(ef_handle, robot_base)` - position relative to robot base
  - `getObjectMatrix(ef_handle, sim.handle_world)` - full transformation in world frame
  - `getObjectMatrix(ef_handle, robot_base)` - full transformation in robot frame
  
  **Finding:** Using `getObjectMatrix(..., sim.handle_world)` like TF Matrix.py does gives most accurate results

---

## 📚 Reference - File Locations

```
Final_Robot/
├── src/main/
│   ├── TF Matrix.py ⭐ (FK validation - WORKING)
│   └── getcup trajectory.py ⭐ (Cup trajectory extraction - WORKING)
├── trajectory_cup_data.npz (Cup trajectory dataset)
├── DH_TABLE_COPPELIASIM.md (DH parameters reference)
├── PROJECT_SUMMARY.md (Original project description)
└── PROJECT_STATUS.md (THIS FILE - Current progress)
```

---

## 💾 How to Use Current Code

### 1. Extract Cup Trajectory
```bash
python src/main/getcup\ trajectory.py
# Creates: trajectory_cup_data.npz
# Shows: Recommended grab point with visualization
```

### 2. Validate DH Parameters
```bash
python src/main/TF\ Matrix.py
# Output: Position error (should be ~0.24mm at home)
# Prints: All DH matrices for reference
```

### 3. Next: Implement IK Solver
Copy DH computation from `TF Matrix.py` and add Jacobian-based IK algorithm:
- Use same DH matrix chain
- Implement geometric Jacobian calculation
- Use damped pseudo-inverse for joint updates
- Apply joint angles to simulation with `setJointPosition`
- Validate FK matches target position

---

## 📖 Learning Timeline

| Task | Status | Error | Lesson |
|------|--------|-------|--------|
| Extract link lengths | ✅ Done | - | Robot has 860mm reach |
| Verify kinematic frames | ✅ Done | - | Frames are true control frames |
| Compute FK from DH | ✅ Done | 0.24mm | Q2 offset (-π/2) is critical! |
| Extract IK trajectory | 🔄 In Progress | 4mm conv | Coordinate frame mismatch |
| Full pick-and-place | ⏳ Pending | - | Waiting for working IK |

---

## 🚀 Success Criteria

- [x] DH parameters validated (0.24mm accuracy)
- [x] Cup trajectory extracted with grab point recommendation
- [ ] IK solver converges with < 5mm error
- [ ] Joint angles apply correctly to simulation
- [ ] Full pick-and-place cycle executes (grab cup → place → home)

---

**Questions or clarifications needed?** Check `TF Matrix.py` line-by-line for DH computation example.
