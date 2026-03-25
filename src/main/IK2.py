"""
Part 2 : EF Trajectory Design  +  Jacobian Velocity IK  (OOP Version)
=======================================================================
Logic identical to part2_ik.py (procedural), rewritten in OOP style.

Classes:
  GraspFrameSolver   -- horizontal side-grasp orientation design
  TrajectoryPlanner  -- EF trajectory design (all phases, S-curve)
  OrientationPlanner -- per-step quaternion targets + Slerp
  JacobianIKSolver   -- numerical Jacobian + damped pseudo-inverse IK
  ResultRecorder     -- stores and plots IK results

Usage: python part2_ik_oop.py
Output: trajectory.npz, plot_traj_design.png, plot_jacobian_ik.png, plot_orientation.png
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ===================================================================
#  TUNABLE PARAMETERS
# ===================================================================
EF_Z_OFFSET      = 0.0
PRE_GRAB_HEIGHT  = 0.12
APPROACH_STEPS   = 120
DESCEND_STEPS    = 15
GRAB_STEPS       = 20
LIFT_STEPS       = 50
TRANSPORT_STEPS  = 90
PLACE_STEPS      = 30
RETURN_STEPS     = 50

LIFT_HEIGHT      = 0.20
PLACE_TARGET_Z   = 0.12

GRIP_OPEN  =  0.20
GRIP_CLOSE = -0.20
GRIP_HOLD  = -0.04
GRIP_IDLE  =  0.0

Kp          = 10.0
Kp_ori      = 8.0
W_ORI       = 0.8
DELTA_Q     = 1e-4
LAMBDA_BASE = 0.005
QD_MAX      = 3.5

Q_MIN = np.array([-2.97, -1.75, -3.14, -3.49, -2.09, -6.28])
Q_MAX = np.array([ 2.97,  2.62,  1.22,  3.49,  2.09,  6.28])
GP8_MAX_REACH = 0.727


# ===================================================================
#  CLASS 1: GraspFrameSolver
#  Computes horizontal side-grasp EF orientation from surface normal.
#  Strategy: minimal rotation to align gripper UP (local-X) with the
#  surface normal (same logic as _side_grasp_frame in part2_ik.py).
# ===================================================================
class GraspFrameSolver:
    def __init__(self, R_home_ref: R):
        self.R_home_ref = R_home_ref

    def solve(self, surface_normal: np.ndarray, max_pitch: float = 0.0) -> tuple:
        """
        Returns (R_grasp, z_horizontal, yaw, roll, pitch).
        max_pitch is accepted for API compatibility but not applied
        (same behaviour as part2_ik.py: pitch disabled at grab).
        """
        n = surface_normal.copy()
        if n[2] < 0:
            n = -n
        n = n / np.linalg.norm(n)

        home_x = self.R_home_ref.apply([1, 0, 0])
        cross_p = np.cross(home_x, n)
        norm_c  = np.linalg.norm(cross_p)

        if norm_c > 1e-6:
            axis     = cross_p / norm_c
            angle    = np.arccos(np.clip(np.dot(home_x, n), -1.0, 1.0))
            R_align  = R.from_rotvec(axis * angle)
            R_final  = R_align * self.R_home_ref
        else:
            R_final = self.R_home_ref

        z_horizontal = R_final.apply([0, 0, 1])
        return R_final, z_horizontal, 0.0, 0.0, 0.0


# ===================================================================
#  CLASS 2: TrajectoryPlanner
#  Designs EF position + gripper command arrays for all 8 phases.
# ===================================================================
class TrajectoryPlanner:
    def __init__(self, cup_data: dict, place_pos: np.ndarray,
                 grasp_solver: GraspFrameSolver,
                 R_grab: R, R_place: R,
                 cup_euler_xyz: np.ndarray):
        self.d              = cup_data
        self.place_pos      = place_pos
        self.grasp_solver   = grasp_solver
        self.R_grab         = R_grab
        self.R_place        = R_place
        self.cup_euler_xyz  = cup_euler_xyz

        # Phase boundaries (computed in _compute_phases)
        self.phases = {}
        self._compute_phases()
        self._allocate_arrays()

    # ------------------------------------------------------------------
    def _compute_phases(self):
        d = self.d
        global APPROACH_STEPS  # allow trim

        GRAB_STEP     = int(d['GRAB_STEP'])
        STEPS_COLLECT = int(d['STEPS_COLLECT'])

        descend_start  = GRAB_STEP - DESCEND_STEPS
        approach_start = descend_start - APPROACH_STEPS
        if approach_start < 0:
            globals()['APPROACH_STEPS'] = descend_start
            approach_start = 0

        wait_end      = approach_start
        grab_end      = GRAB_STEP + GRAB_STEPS
        lift_end      = grab_end   + LIFT_STEPS
        transport_end = lift_end   + TRANSPORT_STEPS
        place_end     = transport_end + PLACE_STEPS
        STEPS         = place_end  + RETURN_STEPS

        # Ensure cup data covers grab window
        need_idx = GRAB_STEP + GRAB_STEPS - 1
        if need_idx >= STEPS_COLLECT:
            gs = STEPS_COLLECT - GRAB_STEP
            grab_end      = GRAB_STEP + gs
            lift_end      = grab_end  + LIFT_STEPS
            transport_end = lift_end  + TRANSPORT_STEPS
            place_end     = transport_end + PLACE_STEPS
            STEPS         = place_end + RETURN_STEPS

        self.phases = dict(
            GRAB_STEP=GRAB_STEP, STEPS_COLLECT=STEPS_COLLECT,
            approach_start=approach_start, descend_start=descend_start,
            wait_end=wait_end, grab_end=grab_end,
            lift_end=lift_end, transport_end=transport_end,
            place_end=place_end, STEPS=STEPS,
        )

    def _allocate_arrays(self):
        STEPS = self.phases['STEPS']
        self.ef_x        = np.zeros(STEPS)
        self.ef_y        = np.zeros(STEPS)
        self.ef_z        = np.zeros(STEPS)
        self.gripper_cmd = np.zeros(STEPS)

    @staticmethod
    def scurve(a: float) -> float:
        return 0.5 * (1.0 - np.cos(np.pi * np.clip(a, 0.0, 1.0)))

    # ------------------------------------------------------------------
    def design(self, ef_home: np.ndarray,
               grab_pos: np.ndarray, pre_grab: np.ndarray,
               z_ee_grab: np.ndarray,
               place_target: np.ndarray, place_above: np.ndarray) -> None:
        """Fill ef_x/y/z and gripper_cmd for all phases."""
        d = self.d
        p = self.phases
        GRAB_STEP     = p['GRAB_STEP']
        STEPS_COLLECT = p['STEPS_COLLECT']
        approach_start = p['approach_start']
        descend_start  = p['descend_start']
        wait_end       = p['wait_end']
        grab_end       = p['grab_end']
        lift_end       = p['lift_end']
        transport_end  = p['transport_end']
        place_end      = p['place_end']
        STEPS          = p['STEPS']

        cup_px = np.asarray(d['cup_px'], dtype=float)
        cup_py = np.asarray(d['cup_py'], dtype=float)
        cup_pz = np.asarray(d['cup_pz'], dtype=float)

        sc = self.scurve

        # ---- Phase 0  WAIT ----
        for k in range(wait_end):
            self.ef_x[k], self.ef_y[k], self.ef_z[k] = ef_home
            self.gripper_cmd[k] = GRIP_OPEN

        # ---- Phase 1  APPROACH (S-curve home -> pre_grab) ----
        for k in range(APPROACH_STEPS):
            step = approach_start + k
            s = sc(k / max(APPROACH_STEPS - 1, 1))
            self.ef_x[step] = ef_home[0] + s * (pre_grab[0] - ef_home[0])
            self.ef_y[step] = ef_home[1] + s * (pre_grab[1] - ef_home[1])
            self.ef_z[step] = ef_home[2] + s * (pre_grab[2] - ef_home[2])
            self.gripper_cmd[step] = GRIP_OPEN

        # ---- Phase 2  DESCEND (horizontal slide-in, tracking cup XY) ----
        EARLY_CLOSE = 5
        for k in range(DESCEND_STEPS):
            step = descend_start + k
            s = sc(k / max(DESCEND_STEPS - 1, 1))
            cup_pos_k = np.array([cup_px[step], cup_py[step], cup_pz[step]])
            n_cup_k = R.from_euler('XYZ', self.cup_euler_xyz[step]).apply([0, 0, 1])
            _, z_k, _, _, _ = self.grasp_solver.solve(n_cup_k, max_pitch=0)
            pre_k = cup_pos_k - z_k * 0.15
            self.ef_x[step] = pre_k[0] * (1 - s) + cup_pos_k[0] * s
            self.ef_y[step] = pre_k[1] * (1 - s) + cup_pos_k[1] * s
            self.ef_z[step] = pre_k[2] * (1 - s) + cup_pos_k[2] * s
            self.gripper_cmd[step] = GRIP_CLOSE if k >= DESCEND_STEPS - EARLY_CLOSE else GRIP_OPEN

        # ---- Phase 3  GRAB (track cup center, close gripper) ----
        for k in range(GRAB_STEPS):
            step = GRAB_STEP + k
            idx  = min(step, STEPS_COLLECT - 1)
            self.ef_x[step] = cup_px[idx]
            self.ef_y[step] = cup_py[idx]
            self.ef_z[step] = cup_pz[idx]
            self.gripper_cmd[step] = GRIP_CLOSE

        # ---- Phase 4  LIFT (straight up in global Z) ----
        lift_s = np.array([self.ef_x[grab_end - 1],
                           self.ef_y[grab_end - 1],
                           self.ef_z[grab_end - 1]])
        lift_e = lift_s + np.array([0.0, 0.0, LIFT_HEIGHT])
        for k in range(LIFT_STEPS):
            step = grab_end + k
            s = sc(k / max(LIFT_STEPS - 1, 1))
            self.ef_x[step] = lift_s[0] + s * (lift_e[0] - lift_s[0])
            self.ef_y[step] = lift_s[1] + s * (lift_e[1] - lift_s[1])
            self.ef_z[step] = lift_s[2] + s * (lift_e[2] - lift_s[2])
            self.gripper_cmd[step] = GRIP_HOLD

        # ---- Phase 5  TRANSPORT (lift_e -> via -> place_above) ----
        via_x   = 0.45
        via_y   = (lift_e[1] + place_above[1]) / 2.0
        via_z   = max(lift_e[2], place_above[2]) + 0.05
        pos_via = np.array([via_x, via_y, via_z])
        self._via = pos_via  # store for printing

        seg_A = TRANSPORT_STEPS // 2
        seg_B = TRANSPORT_STEPS - seg_A
        for k in range(seg_A):
            step = lift_end + k
            s = sc(k / max(seg_A - 1, 1))
            self.ef_x[step] = lift_e[0] + s * (pos_via[0] - lift_e[0])
            self.ef_y[step] = lift_e[1] + s * (pos_via[1] - lift_e[1])
            self.ef_z[step] = lift_e[2] + s * (pos_via[2] - lift_e[2])
            self.gripper_cmd[step] = GRIP_HOLD
        for k in range(seg_B):
            step = lift_end + seg_A + k
            s = sc(k / max(seg_B - 1, 1))
            self.ef_x[step] = pos_via[0] + s * (place_above[0] - pos_via[0])
            self.ef_y[step] = pos_via[1] + s * (place_above[1] - pos_via[1])
            self.ef_z[step] = pos_via[2] + s * (place_above[2] - pos_via[2])
            self.gripper_cmd[step] = GRIP_HOLD

        # ---- Phase 6  PLACE ----
        for k in range(PLACE_STEPS):
            step = transport_end + k
            s = sc(k / max(PLACE_STEPS - 1, 1))
            self.ef_x[step] = place_above[0] + s * (place_target[0] - place_above[0])
            self.ef_y[step] = place_above[1] + s * (place_target[1] - place_above[1])
            self.ef_z[step] = place_above[2] + s * (place_target[2] - place_above[2])
            self.gripper_cmd[step] = GRIP_OPEN if k >= PLACE_STEPS - 5 else GRIP_HOLD

        # ---- Phase 7  RETURN (retreat -> via -> home) ----
        retreat_n = min(15, RETURN_STEPS // 4)
        via_ret_n = (RETURN_STEPS - retreat_n) // 2
        home_n    = RETURN_STEPS - retreat_n - via_ret_n
        ret_via   = np.array([0.45452, 0.0, 0.60])

        for k in range(retreat_n):
            step = place_end + k
            s = sc(k / max(retreat_n - 1, 1))
            self.ef_x[step] = place_target[0] + s * (place_above[0] - place_target[0])
            self.ef_y[step] = place_target[1] + s * (place_above[1] - place_target[1])
            self.ef_z[step] = place_target[2] + s * (place_above[2] - place_target[2])
            self.gripper_cmd[step] = GRIP_IDLE

        b = place_end + retreat_n
        for k in range(via_ret_n):
            step = b + k
            s = sc(k / max(via_ret_n - 1, 1))
            self.ef_x[step] = place_above[0] + s * (ret_via[0] - place_above[0])
            self.ef_y[step] = place_above[1] + s * (ret_via[1] - place_above[1])
            self.ef_z[step] = place_above[2] + s * (ret_via[2] - place_above[2])
            self.gripper_cmd[step] = GRIP_IDLE

        c = b + via_ret_n
        for k in range(home_n):
            step = c + k
            s = sc(k / max(home_n - 1, 1))
            self.ef_x[step] = ret_via[0] + s * (ef_home[0] - ret_via[0])
            self.ef_y[step] = ret_via[1] + s * (ef_home[1] - ret_via[1])
            self.ef_z[step] = ret_via[2] + s * (ef_home[2] - ret_via[2])
            self.gripper_cmd[step] = GRIP_IDLE


# ===================================================================
#  CLASS 3: OrientationPlanner
#  Builds per-step quaternion targets using Slerp and per-step
#  side-grasp frames — same schedule as part2_ik.py.
# ===================================================================
class OrientationPlanner:
    def __init__(self, phases: dict, cup_data: dict,
                 cup_euler_xyz: np.ndarray,
                 R_home: R, R_grab: R, R_place: R,
                 grasp_solver: GraspFrameSolver):
        self.phases         = phases
        self.cup_data       = cup_data
        self.cup_euler_xyz  = cup_euler_xyz
        self.R_home         = R_home
        self.R_grab         = R_grab
        self.R_place        = R_place
        self.grasp_solver   = grasp_solver

        self.quat_target    = None   # filled by build()
        self.euler_des_arr  = None

    @staticmethod
    def scurve(a):
        return 0.5 * (1.0 - np.cos(np.pi * np.clip(a, 0.0, 1.0)))

    def build(self) -> np.ndarray:
        """Returns quat_target array (STEPS, 4) in scipy [x,y,z,w] convention."""
        p  = self.phases
        STEPS          = p['STEPS']
        wait_end       = p['wait_end']
        approach_start = p['approach_start']
        descend_start  = p['descend_start']
        GRAB_STEP      = p['GRAB_STEP']
        grab_end       = p['grab_end']
        lift_end       = p['lift_end']
        transport_end  = p['transport_end']
        place_end      = p['place_end']
        STEPS_COLLECT  = p['STEPS_COLLECT']

        cup_px = np.asarray(self.cup_data['cup_px'], dtype=float)
        cup_py = np.asarray(self.cup_data['cup_py'], dtype=float)
        cup_pz = np.asarray(self.cup_data['cup_pz'], dtype=float)

        slerp_h2g = Slerp([0, 1], R.concatenate([self.R_home, self.R_grab]))
        slerp_g2p = Slerp([0, 1], R.concatenate([self.R_grab, self.R_place]))
        slerp_p2h = Slerp([0, 1], R.concatenate([self.R_place, self.R_home]))

        quat_target = np.zeros((STEPS, 4))
        sc = self.scurve

        for k in range(STEPS):
            if k < wait_end:
                quat_target[k] = self.R_home.as_quat()

            elif k < descend_start:
                s = sc((k - approach_start) / max(APPROACH_STEPS - 1, 1))
                quat_target[k] = slerp_h2g([s])[0].as_quat()

            elif k < GRAB_STEP:
                n_cup_k = R.from_euler('XYZ', self.cup_euler_xyz[k]).apply([0, 0, 1])
                R_k, _, _, _, _ = self.grasp_solver.solve(n_cup_k, max_pitch=0)
                quat_target[k] = R_k.as_quat()

            elif k < grab_end:
                idx = min(k, STEPS_COLLECT - 1)
                n_cup_k = R.from_euler('XYZ', self.cup_euler_xyz[idx]).apply([0, 0, 1])
                R_k, _, _, _, _ = self.grasp_solver.solve(n_cup_k, max_pitch=0)
                quat_target[k] = R_k.as_quat()

            elif k < lift_end:
                quat_target[k] = self.R_grab.as_quat()

            elif k < transport_end:
                frac = (k - lift_end) / max(TRANSPORT_STEPS - 1, 1)
                quat_target[k] = slerp_g2p([sc(frac)])[0].as_quat()

            elif k < place_end:
                quat_target[k] = self.R_place.as_quat()

            else:
                s = sc((k - place_end) / max(RETURN_STEPS - 1, 1))
                quat_target[k] = slerp_p2h([s])[0].as_quat()

        self.quat_target   = quat_target
        self.euler_des_arr = np.array([R.from_quat(q).as_euler('XYZ')
                                       for q in quat_target])
        return quat_target


# ===================================================================
#  CLASS 4: JacobianIKSolver
#  Numerical Jacobian (6x6 pos+ori) + damped pseudo-inverse IK.
#  Phase-varying orientation weights — identical to part2_ik.py.
# ===================================================================
class JacobianIKSolver:
    def __init__(self, sim, dt: float):
        self.sim  = sim
        self.dt   = dt
        self.ef   = sim.getObject('/yaskawa/gripperEF')
        self.joints = [sim.getObject(f'/yaskawa/joint{i}') for i in range(1, 7)]
        self.W_diag = np.array([1.0, 1.0, 1.0, W_ORI, W_ORI, W_ORI])

    def get_state(self):
        q    = np.array([self.sim.getJointPosition(j) for j in self.joints])
        p    = np.array(self.sim.getObjectPosition(self.ef, self.sim.handle_world))
        quat = self.sim.getObjectQuaternion(self.ef, self.sim.handle_world)
        return q, p, R.from_quat(quat)

    def _numerical_jacobian(self, q_cur: np.ndarray) -> np.ndarray:
        J = np.zeros((6, 6))
        for i in range(6):
            self.sim.setJointPosition(self.joints[i], float(q_cur[i] + DELTA_Q))
            p_plus = np.array(self.sim.getObjectPosition(self.ef, self.sim.handle_world))
            R_plus = R.from_quat(self.sim.getObjectQuaternion(self.ef, self.sim.handle_world))

            self.sim.setJointPosition(self.joints[i], float(q_cur[i] - DELTA_Q))
            p_minus = np.array(self.sim.getObjectPosition(self.ef, self.sim.handle_world))
            R_minus = R.from_quat(self.sim.getObjectQuaternion(self.ef, self.sim.handle_world))

            self.sim.setJointPosition(self.joints[i], float(q_cur[i]))
            J[0:3, i] = (p_plus  - p_minus)  / (2.0 * DELTA_Q)
            J[3:6, i] = (R_plus  * R_minus.inv()).as_rotvec() / (2.0 * DELTA_Q)
        return J

    def _phase_ori_weight(self, step: int, phases: dict) -> float:
        """Replicate part2_ik.py phase-dependent orientation weight."""
        if step >= phases['place_end']:
            return 0.10   # RETURN: position priority (103% reach)
        elif step >= phases['transport_end']:
            return 1.0    # PLACE: full ori weight
        elif step >= phases['lift_end']:
            return 1.0    # TRANSPORT: full ori weight
        elif step >= phases['grab_end']:
            return 0.3    # LIFT: some ori tracking
        return 1.0

    def solve_step(self, step: int, phases: dict,
                   P_des: np.ndarray, R_target: R,
                   q_cur: np.ndarray, P_cur: np.ndarray, R_cur: R,
                   V_ff: np.ndarray) -> np.ndarray:
        """One IK step. Returns q_dot (clamped)."""
        V_pos = V_ff + Kp * (P_des - P_cur)
        V_ori = Kp_ori * (R_target * R_cur.inv()).as_rotvec()
        V_full = np.concatenate([V_pos, V_ori])

        wait_end = phases['wait_end']
        if step < wait_end and np.linalg.norm(V_full) < 1e-5:
            return np.zeros(6)

        J = self._numerical_jacobian(q_cur)

        # Phase-varying orientation weight
        ori_w   = self._phase_ori_weight(step, phases)
        W_step  = self.W_diag.copy()
        W_step[3:6] *= ori_w

        J_w = np.diag(W_step) @ J
        V_w = W_step * V_full
        JJT = J_w @ J_w.T + LAMBDA_BASE * np.eye(6)
        q_dot = J_w.T @ np.linalg.solve(JJT, V_w)
        return np.clip(q_dot, -QD_MAX, QD_MAX)


# ===================================================================
#  CLASS 5: ResultRecorder
#  Stores IK data and generates the 3 plots from part2_ik.py.
# ===================================================================
class ResultRecorder:
    def __init__(self, STEPS: int, DT: float, phases: dict):
        self.STEPS  = STEPS
        self.DT     = DT
        self.phases = phases
        self.q_all        = np.zeros((STEPS, 6))
        self.qd_all       = np.zeros((STEPS, 6))
        self.ef_actual    = np.zeros((STEPS, 3))
        self.ef_err_arr   = np.zeros(STEPS)
        self.ori_err_arr  = np.zeros(STEPS)
        self.ef_euler_arr = np.zeros((STEPS, 3))

    def record(self, step: int, q_new, q_dot, P_actual, R_actual,
               P_des, R_target):
        self.q_all[step]        = q_new
        self.qd_all[step]       = q_dot
        self.ef_actual[step]    = P_actual
        self.ef_err_arr[step]   = np.linalg.norm(P_des - P_actual)
        self.ori_err_arr[step]  = np.linalg.norm(
            (R_target * R_actual.inv()).as_rotvec())
        self.ef_euler_arr[step] = R_actual.as_euler('XYZ')

    def _phase_label(self, step: int) -> str:
        p = self.phases
        if   step < p['wait_end']:       return "WAIT"
        elif step < p['descend_start']:  return "APPROACH"    # type: ignore[key]
        elif step < p['GRAB_STEP']:      return "DESCEND"
        elif step < p['grab_end']:       return "GRAB"
        elif step < p['lift_end']:       return "LIFT"
        elif step < p['transport_end']:  return "TRANSPORT"
        elif step < p['place_end']:      return "PLACE"
        return "RETURN"

    def print_progress(self, step: int, q_dot: np.ndarray):
        if step % 50 == 0 or step == self.STEPS - 1:
            print(f"    step {step:4d}/{self.STEPS}  t={step*self.DT:5.1f}s"
                  f"  pos_err={self.ef_err_arr[step]:.4f}m"
                  f"  ori_err={self.ori_err_arr[step]:.3f}"
                  f"  |qd|={np.linalg.norm(q_dot):.3f}"
                  f"  [{self._phase_label(step)}]")
        if self.ef_err_arr[step] > 0.05:
            print(f"  !! Large position error at step {step}: "
                  f"{self.ef_err_arr[step]:.3f} m")

    # ---- Plot 1: trajectory design ----
    def plot_trajectory_design(self, ef_x, ef_y, ef_z, gripper_cmd,
                                cup_px, cup_py, cup_pz, GRAB_STEP: int):
        t_traj = np.arange(self.STEPS) * self.DT
        t_cup  = np.arange(len(cup_px)) * self.DT
        p      = self.phases

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('EF Trajectory Design -- verify timing before IK', fontsize=13)

        for ci, (lab, ef_arr, cup_arr) in enumerate([
                ('X (m)', ef_x, cup_px), ('Y (m)', ef_y, cup_py),
                ('Z (m)', ef_z, cup_pz)]):
            ax = axes.flat[ci]
            ax.plot(t_cup,  cup_arr, 'r-',  alpha=0.5, label=f'Cup {lab[0].lower()}')
            ax.plot(t_traj, ef_arr,  'b-',  lw=1.5,    label=f'EF {lab[0].lower()} plan')
            ax.axvline(GRAB_STEP * self.DT, color='green', ls='--', alpha=.7,
                       label=f'GRAB @ step {GRAB_STEP}')
            for bd, clr in [(p['approach_start'], 'purple'),
                            (p['descend_start'],  'orange'),
                            (p['grab_end'],       'red'),
                            (p['lift_end'],       'cyan'),
                            (p['transport_end'],  'brown'),
                            (p['place_end'],      'gray')]:
                ax.axvline(bd * self.DT, color=clr, ls=':', alpha=0.35)
            ax.set_xlabel('t (s)'); ax.set_ylabel(lab); ax.legend(fontsize=7)

        ax = axes[1, 1]
        ax.plot(t_traj, gripper_cmd, 'k-', lw=1.5)
        ax.axvline(GRAB_STEP * self.DT, color='green', ls='--', alpha=.7)
        ax.set_xlabel('t (s)'); ax.set_ylabel('vel (m/s)')
        ax.set_title('Gripper command')

        plt.tight_layout()
        plt.savefig('plot_traj_design.png', dpi=120)
        print("  Saved -> plot_traj_design.png")
        plt.show()

    # ---- Plot 2: IK joints & error ----
    def plot_jacobian_ik(self, GRAB_STEP: int):
        t_traj = np.arange(self.STEPS) * self.DT
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Jacobian IK Results -- Joint Trajectories & Error', fontsize=13)

        for j in range(3):
            axes[0, 0].plot(t_traj, np.degrees(self.q_all[:, j]), label=f'j{j+1}')
        axes[0, 0].set_title('Joints 1-3 (deg)'); axes[0, 0].legend()
        axes[0, 0].set_xlabel('t (s)')

        for j in range(3, 6):
            axes[0, 1].plot(t_traj, np.degrees(self.q_all[:, j]), label=f'j{j+1}')
        axes[0, 1].set_title('Joints 4-6 (deg)'); axes[0, 1].legend()
        axes[0, 1].set_xlabel('t (s)')

        for j in range(3):
            axes[1, 0].plot(t_traj, self.qd_all[:, j], label=f'qd{j+1}')
        axes[1, 0].set_title('Joint vel 1-3 (rad/s)'); axes[1, 0].legend()
        axes[1, 0].set_xlabel('t (s)')

        for j in range(3, 6):
            axes[1, 1].plot(t_traj, self.qd_all[:, j], label=f'qd{j+1}')
        axes[1, 1].set_title('Joint vel 4-6 (rad/s)'); axes[1, 1].legend()
        axes[1, 1].set_xlabel('t (s)')

        axes[0, 2].plot(t_traj, self.ef_err_arr * 1000, 'r-', lw=1.2,
                        label='pos (mm)')
        axes[0, 2].plot(t_traj, self.ori_err_arr * 100,  'b--', lw=1.0,
                        label='ori x100')
        axes[0, 2].set_title('Tracking error')
        axes[0, 2].set_xlabel('t (s)'); axes[0, 2].legend(fontsize=7)

        axes[1, 2].plot(t_traj, (self.ef_actual[:, 0] - getattr(self, '_ef_x', np.zeros(self.STEPS))) * 1000, label='dx')
        axes[1, 2].plot(t_traj, (self.ef_actual[:, 1] - getattr(self, '_ef_y', np.zeros(self.STEPS))) * 1000, label='dy')
        axes[1, 2].plot(t_traj, (self.ef_actual[:, 2] - getattr(self, '_ef_z', np.zeros(self.STEPS))) * 1000, label='dz')
        axes[1, 2].set_title('Per-axis pos error (mm)'); axes[1, 2].legend()
        axes[1, 2].set_xlabel('t (s)')

        for ax in axes.flat:
            ax.axvline(GRAB_STEP * self.DT, color='green', ls='--', alpha=0.4)

        plt.tight_layout()
        plt.savefig('plot_jacobian_ik.png', dpi=120)
        print("  Saved -> plot_jacobian_ik.png")
        plt.show()

    # ---- Plot 3: orientation (rubric 2.4) ----
    def plot_orientation(self, euler_des_arr: np.ndarray, GRAB_STEP: int):
        t_traj = np.arange(self.STEPS) * self.DT
        euler_des_deg = np.degrees(euler_des_arr)
        euler_act_deg = np.degrees(self.ef_euler_arr)
        omega_des = np.gradient(euler_des_deg, self.DT, axis=0)
        omega_act = np.gradient(euler_act_deg, self.DT, axis=0)
        p = self.phases

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('EF Orientation & Angular Velocity  (rubric 2.4)', fontsize=13)

        lbl_ori = [r'$\alpha$ (deg)', r'$\beta$ (deg)', r'$\gamma$ (deg)']
        lbl_omg = [r'$\omega_\alpha$ (deg/s)', r'$\omega_\beta$ (deg/s)',
                   r'$\omega_\gamma$ (deg/s)']
        bds = [(p['descend_start'], 'orange'), (p['grab_end'], 'red'),
               (p['lift_end'], 'cyan'), (p['transport_end'], 'brown'),
               (p['place_end'], 'gray')]

        for ci in range(3):
            ax = axes[0, ci]
            ax.plot(t_traj, euler_des_deg[:, ci], 'b-',  lw=1.5, label='designed')
            ax.plot(t_traj, euler_act_deg[:, ci], 'r--', lw=1.0, label='actual (IK)')
            ax.set_xlabel('t (s)'); ax.set_ylabel(lbl_ori[ci])
            ax.set_title(lbl_ori[ci]); ax.legend(fontsize=7)
            ax.axvline(GRAB_STEP * self.DT, color='green', ls='--', alpha=0.4)
            for bd, clr in bds:
                ax.axvline(bd * self.DT, color=clr, ls=':', alpha=0.35)

            ax = axes[1, ci]
            ax.plot(t_traj, omega_des[:, ci], 'b-',  lw=1.5, label='designed')
            ax.plot(t_traj, omega_act[:, ci], 'r--', lw=1.0, label='actual (IK)')
            ax.set_xlabel('t (s)'); ax.set_ylabel(lbl_omg[ci])
            ax.set_title(lbl_omg[ci]); ax.legend(fontsize=7)
            ax.axvline(GRAB_STEP * self.DT, color='green', ls='--', alpha=0.4)
            for bd, clr in bds:
                ax.axvline(bd * self.DT, color=clr, ls=':', alpha=0.35)

        plt.tight_layout()
        plt.savefig('plot_orientation.png', dpi=120)
        print("  Saved -> plot_orientation.png")
        plt.show()


# ===================================================================
#  MAIN
# ===================================================================
def main():
    print("\n" + "=" * 60)
    print("  OOP Part 2 — Horizontal Side-Grasp + Jacobian IK")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 0. Connect to CoppeliaSim
    # ------------------------------------------------------------------
    client = RemoteAPIClient()
    sim    = client.require('sim')

    # ------------------------------------------------------------------
    # 1. Load cup data  (same keys as part2_ik.py)
    # ------------------------------------------------------------------
    print("\n  1.  LOAD CUP DATA")
    d = np.load('trajectory_cup_data.npz', allow_pickle=True)
    
    cup_px = np.asarray(d['px'], dtype=float)
    cup_py = np.asarray(d['py'], dtype=float)
    cup_pz = np.asarray(d['pz'], dtype=float)
    cup_euler_xyz = np.column_stack([
        np.asarray(d['ox'], dtype=float),
        np.asarray(d['oy'], dtype=float),
        np.asarray(d['oz'], dtype=float),
    ])

    ef_home       = np.asarray(d['ee_home'],   dtype=float)
    robot_base    = np.asarray(d['robot_base'], dtype=float)
    place_pos_npz = np.asarray(d['place_pos'],  dtype=float)
    GRAB_STEP     = int(d['grab_step'])
    DT            = float(d['timestep'])
    STEPS_COLLECT = int(d['total_steps'])

    cup_data = dict(cup_px=cup_px, cup_py=cup_py, cup_pz=cup_pz,
                    GRAB_STEP=GRAB_STEP, DT=DT, STEPS_COLLECT=STEPS_COLLECT)

    # Query /conveyor for dynamic place position (identical to part2_ik.py)
    try:
        _conv_h     = sim.getObject('/conveyor')
        place_pos   = np.array(sim.getObjectPosition(_conv_h, sim.handle_world))
        _conv_euler = np.array(sim.getObjectOrientation(_conv_h, sim.handle_world))
        R_conv      = R.from_euler('XYZ', _conv_euler)
        conv_normal = R_conv.apply([0, 0, 1])
        print(f"  /conveyor world pos = {place_pos.round(4)}")
    except Exception:
        place_pos   = place_pos_npz.copy()
        conv_normal = np.array([0.0, 0.0, 1.0])
        print(f"  /conveyor not found — using npz: {place_pos.round(4)}")

    # ------------------------------------------------------------------
    # 2. Read actual home orientation from CoppeliaSim
    # ------------------------------------------------------------------
    ef_h         = sim.getObject('/yaskawa/gripperEF')
    _quat_actual = sim.getObjectQuaternion(ef_h, sim.handle_world)
    R_home_approx = R.from_quat(_quat_actual)

    # ------------------------------------------------------------------
    # 3. Build grab & place orientations  (GraspFrameSolver)
    # ------------------------------------------------------------------
    print("\n  2.  DESIGN ORIENTATIONS")
    grasp_solver = GraspFrameSolver(R_home_approx)

    # --- GRAB ---
    n_cup = R.from_euler('XYZ', cup_euler_xyz[GRAB_STEP]).apply([0, 0, 1])
    if n_cup[2] < 0: n_cup = -n_cup
    R_grab, z_ee_grab, *_ = grasp_solver.solve(n_cup, max_pitch=0)
    x_ee_grab = R_grab.apply([1, 0, 0])
    grab_pos  = np.array([cup_px[GRAB_STEP], cup_py[GRAB_STEP], cup_pz[GRAB_STEP]])
    pre_grab  = grab_pos - z_ee_grab * 0.15

    # --- PLACE ---
    pos_place_target = np.array([0.45452, 0.500, 0.462])
    conv_R_full = R.from_euler('XYZ', [-15, -10, -180], degrees=True)
    n_conv = conv_R_full.apply([0, 0, 1])
    if n_conv[2] < 0: n_conv = -n_conv

    R_place, z_ee_place, *_ = grasp_solver.solve(n_conv, max_pitch=np.radians(20))

    # Cup-gripper offset compensation at PLACE
    cup_z_in_grip = R_grab.inv().apply(n_cup)
    cup_z_in_grip /= np.linalg.norm(cup_z_in_grip)
    _a = cup_z_in_grip; _b = np.array([1.0, 0.0, 0.0])
    _cross = np.cross(_a, _b)
    if np.linalg.norm(_cross) > 1e-10:
        _axis  = _cross / np.linalg.norm(_cross)
        _angle = np.arccos(np.clip(np.dot(_a, _b), -1, 1))
        R_corr = R.from_rotvec(_axis * _angle)
    else:
        R_corr = R.identity(); _angle = 0.0
    R_place = R_place * R_corr

    place_target = pos_place_target.copy()
    place_above  = place_target + np.array([0.0, 0.0, 0.15])
    print(f"  Grab pos    = {grab_pos.round(4)}")
    print(f"  Pre-grab    = {pre_grab.round(4)}")
    print(f"  Place target= {place_target.round(4)}")
    print(f"  R_grab euler= {np.degrees(R_grab.as_euler('XYZ')).round(1)}°")
    print(f"  R_place euler={np.degrees(R_place.as_euler('XYZ')).round(1)}°")

    # ------------------------------------------------------------------
    # 4. Position trajectory  (TrajectoryPlanner)
    # ------------------------------------------------------------------
    print("\n  3.  POSITION TRAJECTORY")
    planner = TrajectoryPlanner(cup_data, place_pos, grasp_solver,
                                R_grab, R_place, cup_euler_xyz)
    planner.design(ef_home, grab_pos, pre_grab, z_ee_grab,
                   place_target, place_above)

    phases = planner.phases
    ef_x, ef_y, ef_z = planner.ef_x, planner.ef_y, planner.ef_z
    gripper_cmd       = planner.gripper_cmd
    STEPS = phases['STEPS']
    DT    = float(d['timestep'])

    for lbl, pt in [('pre_grab',   pre_grab),   ('grab_pos',    grab_pos),
                    ('place_target', place_target), ('place_above', place_above)]:
        dist = np.sqrt(pt[0]**2 + pt[1]**2 + (pt[2]-0.33)**2)
        flag = '*** OUT OF REACH ***' if dist > GP8_MAX_REACH else ''
        print(f"  Reach {lbl:16s}: {dist:.3f}m  "
              f"({dist/GP8_MAX_REACH*100:.0f}% of max)  {flag}")

    # ------------------------------------------------------------------
    # 5. Orientation trajectory  (OrientationPlanner)
    # ------------------------------------------------------------------
    print("\n  4.  ORIENTATION TRAJECTORY")

    # Read actual home rotation just before simulation
    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(3): sim.step()

    _quat_home = sim.getObjectQuaternion(ef_h, sim.handle_world)
    R_home_actual = R.from_quat(_quat_home)
    q_cur, P_cur, _ = JacobianIKSolver(sim, DT).get_state()
    sim.stopSimulation()

    ori_planner = OrientationPlanner(phases, cup_data, cup_euler_xyz,
                                     R_home_actual, R_grab, R_place,
                                     grasp_solver)
    quat_target = ori_planner.build()
    euler_des_arr = ori_planner.euler_des_arr

    # ------------------------------------------------------------------
    # 6. Plot trajectory design
    # ------------------------------------------------------------------
    recorder = ResultRecorder(STEPS, DT, phases)
    recorder._ef_x = ef_x
    recorder._ef_y = ef_y
    recorder._ef_z = ef_z
    recorder.plot_trajectory_design(ef_x, ef_y, ef_z, gripper_cmd,
                                    cup_px, cup_py, cup_pz,
                                    phases['GRAB_STEP'])

    # ------------------------------------------------------------------
    # 7. Jacobian IK loop  (JacobianIKSolver)
    # ------------------------------------------------------------------
    print("\n  5.  JACOBIAN IK")
    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(3): sim.step()

    ik_solver = JacobianIKSolver(sim, DT)
    q_cur, _, _ = ik_solver.get_state()

    t0 = time.time()
    for step in range(STEPS):
        P_des = np.array([ef_x[step], ef_y[step], ef_z[step]])
        V_ff  = (np.array([ef_x[step]-ef_x[step-1],
                            ef_y[step]-ef_y[step-1],
                            ef_z[step]-ef_z[step-1]]) / DT
                 if step > 0 else np.zeros(3))

        P_cur, R_cur = (np.array(sim.getObjectPosition(ef_h, sim.handle_world)),
                        R.from_quat(sim.getObjectQuaternion(ef_h, sim.handle_world)))
        R_target_k = R.from_quat(quat_target[step])

        q_dot = ik_solver.solve_step(step, phases, P_des, R_target_k,
                                     q_cur, P_cur, R_cur, V_ff)
        q_new = q_cur + q_dot * DT
        for i in range(6):
            sim.setJointPosition(ik_solver.joints[i], float(q_new[i]))

        P_actual = np.array(sim.getObjectPosition(ef_h, sim.handle_world))
        R_actual = R.from_quat(sim.getObjectQuaternion(ef_h, sim.handle_world))
        recorder.record(step, q_new, q_dot, P_actual, R_actual,
                        P_des, R_target_k)
        recorder.print_progress(step, q_dot)
        q_cur = q_new
        sim.step()

    sim.stopSimulation()
    elapsed = time.time() - t0
    p = phases
    print(f"\n  >> IK complete  ({elapsed:.1f}s)")
    print(f"  max pos error  = {np.max(recorder.ef_err_arr):.4f} m")
    print(f"  mean pos error = {np.mean(recorder.ef_err_arr):.4f} m")
    print(f"  grab max pos   = {np.max(recorder.ef_err_arr[p['GRAB_STEP']:p['grab_end']]):.4f} m")
    print(f"  max ori error  = {np.max(recorder.ori_err_arr):.4f}")

    # ------------------------------------------------------------------
    # 8. Save trajectory.npz
    # ------------------------------------------------------------------
    np.savez('trajectory.npz',
             j1=recorder.q_all[:, 0], j2=recorder.q_all[:, 1],
             j3=recorder.q_all[:, 2], j4=recorder.q_all[:, 3],
             j5=recorder.q_all[:, 4], j6=recorder.q_all[:, 5],
             jd1=recorder.qd_all[:, 0], jd2=recorder.qd_all[:, 1],
             jd3=recorder.qd_all[:, 2], jd4=recorder.qd_all[:, 3],
             jd5=recorder.qd_all[:, 4], jd6=recorder.qd_all[:, 5],
             gripper_cmd=gripper_cmd,
             ef_x=ef_x, ef_y=ef_y, ef_z=ef_z,
             euler_des_alpha=euler_des_arr[:, 0],
             euler_des_beta=euler_des_arr[:, 1],
             euler_des_gamma=euler_des_arr[:, 2],
             ef_actual_x=recorder.ef_actual[:, 0],
             ef_actual_y=recorder.ef_actual[:, 1],
             ef_actual_z=recorder.ef_actual[:, 2],
             euler_actual_alpha=recorder.ef_euler_arr[:, 0],
             euler_actual_beta=recorder.ef_euler_arr[:, 1],
             euler_actual_gamma=recorder.ef_euler_arr[:, 2],
             ef_error=recorder.ef_err_arr,
             ori_error=recorder.ori_err_arr,
             times=np.arange(STEPS) * DT,
             GRAB_STEP=p['GRAB_STEP'], DT=DT, STEPS=STEPS,
             ef_home=ef_home, robot_base=robot_base, place_pos=place_pos)
    print("  Saved -> trajectory.npz")

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    recorder.plot_jacobian_ik(p['GRAB_STEP'])
    recorder.plot_orientation(euler_des_arr, p['GRAB_STEP'])

    print("\n" + "=" * 60)
    print("  Part 2 (OOP) complete.  Next: python part3_run.py")
    print("=" * 60)


if __name__ == '__main__':
    main()