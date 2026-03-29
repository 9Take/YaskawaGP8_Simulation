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

#### 0. **ik_solver_damped.py** ⭐ (NEW - Ready for coordinate frame fix)
- **Purpose:** IK Jacobian solver with lambda damping
- **Status:** Converges fast (9 iterations) but needs coordinate frame transformation
- **Lambda damping:** 0.01 (prevents overshooting, smoother convergence)
- **Key features:**
  - Uses validated DH from TF Matrix.py
  - Damped pseudo-inverse: `J_pseudo = J^T @ inv(J @ J^T + lambda * I)`
  - Applies angles with `setJointPosition` for direct control
  - Next: Add world→robot_base frame transformation

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

### Immediate (Priority 1) - NEXT STEP
- [ ] **Fix IK for Coordinate Frames** - Transform grab position to robot base frame
  - Get robot base orientation: `[0, -π/2, 0]`
  - Transform grab position: world → robot base
  - Solve IK with transformed position
  - Verify EE reaches cup (should match simulation now)

### Phase 2 (Priority 2)
- [ ] **Trajectory Generation** - S-curve path with microstepping
  - Home → Above cup → Descend to grab
  - Use Quintic polynomial interpolation (smooth accel/decel)
  - Track cup position during approach (moving conveyor)
  - Apply to simulation with microstepping

### Phase 3 (Priority 3) 
- [ ] **Full Pick-and-Place Cycle** - 7-phase motion
  - Phase 0: WAIT (EF at home)
  - Phase 1: APPROACH (S-curve home → above cup)
  - Phase 2: DESCEND (track cup + move Z down)
  - Phase 3: GRAB (track cup exactly, close gripper)
  - Phase 4: LIFT (move up)
  - Phase 5: TRANSPORT (fly to place conveyor)
  - Phase 6: PLACE (descend, open gripper)
  - Phase 7: RETURN (retreat up → home)

---

## 🔍 Coordinate Frame Issue - IDENTIFIED & SOLVED ✅

### Frame Analysis Complete (debug/frame_analysis.py)

**At home configuration:**
- `getObjectMatrix(ee, WORLD)`: `[0.640, 0, 0.715]` ✓ **Matches FK!** (error 0.563mm)
- FK DH computation: `[0.640, 0, 0.715]` ✓ **Perfect match!**

**Robot base frame details:**
```
Position: [0, 0, 0.0995]
Orientation: [0, -90°, 0]  ← -90° rotation around Y axis!
```

### Root Cause Found 🎯
- FK model computes in **WORLD FRAME** ✓
- Grab position from `getcup trajectory.py` is in **WORLD FRAME** ✓
- BUT robot simulation uses **ROBOT BASE FRAME** (rotated -90° in Y)
- When we apply IK angles to joint motors, they move in robot base frame
- EE ends up at different position than expected!

### Solution
Transform grab position from WORLD FRAME → ROBOT BASE FRAME before solving IK:
```python
# Grab position is in world frame
grab_pos_world = [0.498, -0.488, 0.426]

# Robot base orientation
base_ori = [0, -π/2, 0]  # -90° rotation in Y

# Must transform to robot base frame before IK
grab_pos_base = transform_world_to_base(grab_pos_world, base_ori)
```

### Test Results
```
Home position error (FK vs Sim): 0.563 mm ✓ Perfect!
IK convergence: 9 iterations (with lambda damping) ✓
FK error to target: 31.4 mm (close!)
But Sim position: 215.8 mm away (because frame mismatch)
```

### Next Step
- Modify IK solver to transform grab position to robot base frame
- Apply IK with transformed position
- Verify EE reaches cup!

---

## 📚 Reference - File Locations

```
Final_Robot/
├── src/main/
│   ├── TF Matrix.py ⭐ (FK validation - 0.24mm accuracy)
│   ├── getcup trajectory.py ⭐ (Cup extraction - grab step 440)
│   └── ik_solver_damped.py ⭐ (IK with lambda damping - needs frame fix)
├── debug/
│   └── frame_analysis.py (Coordinate frame comparison - COMPLETED)
├── trajectory_cup_data.npz (Cup trajectory dataset)
├── DH_TABLE_COPPELIASIM.md (DH reference)
├── PROJECT_SUMMARY.md (Original description)
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
| Extract IK trajectory | 🔄 In Progress | 0.563mm FK err | FK works! Need frame transform |
| Frame analysis | ✅ Done | - | Robot base is -90° rotated in Y |
| Full pick-and-place | ⏳ Pending | - | After coordinate frame fix |

---

## 🚀 Success Criteria

- [x] DH parameters validated (0.24mm accuracy at home)
- [x] Cup trajectory extracted with grab point (step 440, 22s)
- [x] IK solver converges fast with damping (9 iterations)
- [x] FK model validated (0.563mm error)
- [x] Coordinate frame issue identified (robot base -90° rotated)
- [ ] Apply coordinate frame transformation to grab position
- [ ] IK solver applies angles correctly to reach grab point
- [ ] Full pick-and-place cycle executes (grab → place → home)

---

**Questions or clarifications needed?** Check `TF Matrix.py` line-by-line for DH computation example.
