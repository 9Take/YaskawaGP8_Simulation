import time
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from ik_solver import inverse_kinematics

# --- Configuration ---
CFG = {
    'ee_path': '/yaskawa/gripperEF',
    'joint_paths': [f'/yaskawa/joint{i}' for i in range(1, 7)],
    'dt': 0.05,
}

class TrajectoryRunner:
    def __init__(self):
        print("Connecting to CoppeliaSim for Lab: Perfect Circle...")
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        
        self.joints = [self.sim.getObject(p) for p in CFG['joint_paths']]
        # Get the end-effector handle so we can measure actual position
        self.ee_handle = self.sim.getObject(CFG['ee_path']) 

        self.client.setStepping(True)
        if self.sim.getSimulationState() == self.sim.simulation_stopped:
            self.sim.startSimulation()
        
        self.q_last = [np.degrees(self.sim.getJointPosition(j)) for j in self.joints]
        
        self.drawing_handle = self.sim.addDrawingObject(
            self.sim.drawing_linestrip, 5, 0.0, -1, 99999, [1, 0, 0]
        )

        # Arrays to collect graph data
        self.t_data = []
        self.q_data = []
        self.ref_data = []    # The math (Red Line)
        self.actual_data = [] # The simulator reality (Blue Line)

    def move_to_js(self, target_xyz, target_rpy, duration, label):
        print(f"Moving to {label}...")
        target_q = inverse_kinematics(
            X=target_xyz[0], Y=target_xyz[1], Z=target_xyz[2],
            Roll=target_rpy[0], Pitch=target_rpy[1], Yaw=target_rpy[2],
            current_joints=self.q_last
        )
        if target_q is None:
            print(f"   IK FAILED for {label}")
            return False

        steps = int(duration / CFG['dt'])
        start_q = np.array(self.q_last)
        end_q = np.array(target_q)
        
        # +1 ensures we reach the exact final coordinate
        for i in range(steps + 1): 
            t = i * CFG['dt']
            s = 10*(t/duration)**3 - 15*(t/duration)**4 + 6*(t/duration)**5
            if s > 1.0: s = 1.0  # Cap interpolation exactly at 1.0

            curr_q = start_q + (end_q - start_q) * s
            for j, val in zip(self.joints, curr_q.tolist()):
                self.sim.setJointPosition(j, np.radians(val))
            self.sim.step()
            
        self.q_last = target_q
        return True

    def draw_circle(self, center_xyz, radius, duration):
        print(f"Drawing Circle (Radius: {radius*1000} mm)...")
        steps = int(duration / CFG['dt'])
        xc, yc, zc = center_xyz

        self.t_data.clear()
        self.q_data.clear()
        self.ref_data.clear()
        self.actual_data.clear()

        # +1 ensures the loop reaches a full 360 degrees to close the circle
        for i in range(steps + 1): 
            t = i * CFG['dt']
            s = 10*(t/duration)**3 - 15*(t/duration)**4 + 6*(t/duration)**5
            if s > 1.0: s = 1.0 # Cap interpolation exactly at 1.0
            
            theta = np.pi + (2 * np.pi * s)

            x = xc + radius * np.cos(theta)
            y = yc + radius * np.sin(theta)

            # Log the PERFECT mathematical coordinate (Reference)
            self.ref_data.append([x, y, zc])

            # Lock the orientation to prevent IK Gimbal Lock
            target_rpy = [0.0, -89.9, 0.0]

            target_q = inverse_kinematics(
                X=x, Y=y, Z=zc,
                Roll=target_rpy[0], Pitch=target_rpy[1], Yaw=target_rpy[2],
                current_joints=self.q_last
            )

            if target_q is not None:
                for j, val in zip(self.joints, target_q):
                    self.sim.setJointPosition(j, np.radians(val))
                self.q_last = target_q
                self.sim.addDrawingObjectItem(self.drawing_handle, [x, y, zc])

                self.t_data.append(t)
                self.q_data.append(target_q)
            else:
                print(f"   IK Failed at theta = {np.degrees(theta):.1f} deg")

            self.sim.step()
            
            # Log the ACTUAL coordinate from the simulator after stepping
            actual_pos = self.sim.getObjectPosition(self.ee_handle, -1)
            self.actual_data.append(actual_pos)

    def plot_graphs(self):
        if not self.t_data:
            print("No data available to plot.")
            return
            
        print("Generating trajectory plots...")
        t_arr = np.array(self.t_data)
        q_arr = np.array(self.q_data)
        ref_arr = np.array(self.ref_data)
        act_arr = np.array(self.actual_data)

        # --- Plot 1: Standard 2D Graphs ---
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        for i in range(6):
            plt.plot(t_arr, q_arr[:, i], label=f'Joint {i+1}')
        plt.title('Joint Trajectory Profile')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(2, 1, 2)
        plt.plot(t_arr, act_arr[:, 0], label='X Position (m)')
        plt.plot(t_arr, act_arr[:, 1], label='Y Position (m)')
        plt.plot(t_arr, act_arr[:, 2], label='Z Position (m)')
        plt.title('End Effector Position Profile')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # --- Plot 2: 3D Trajectory Comparison ---
        fig3d = plt.figure(figsize=(8, 8))
        ax = fig3d.add_subplot(111, projection='3d')
        
        # Plot Reference Line (Solid Red)
        ax.plot(ref_arr[:, 0], ref_arr[:, 1], ref_arr[:, 2], 'r-', linewidth=2, label='Reference Trajectory')
        
        # Plot Actual Line (Dashed Blue)
        ax.plot(act_arr[:, 0], act_arr[:, 1], act_arr[:, 2], 'b--', linewidth=2, label='Actual Trajectory')
        
        ax.set_title('3D Trajectory of End-Effector')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # ---> ADD THESE LINES HERE <---
        # Lock the Z-axis to realistically show the height (0.40m to 0.50m)
        ax.set_zlim(0.40, 0.50) 
        
        # Lock X and Y so the circle doesn't look squished like an oval
        # Center X is 0.50, Radius is 0.10
        ax.set_xlim(0.35, 0.65) 
        # Center Y is 0.00, Radius is 0.10
        ax.set_ylim(-0.15, 0.15)

        ax.legend()
        plt.show()

def main():
    runner = TrajectoryRunner()
    try:
        CENTER_X = 0.50  
        CENTER_Y = 0.0
        DRAW_Z = 0.45    
        RADIUS = 0.10    
        
        start_x = CENTER_X - RADIUS  
        start_y = CENTER_Y

        # 1. Approach
        runner.move_to_js([start_x, start_y, DRAW_Z + 0.1], [0.0, -89.9, 0.0], 3.0, "Hover over Start")
        runner.move_to_js([start_x, start_y, DRAW_Z], [0.0, -89.9, 0.0], 2.0, "Lower to Start")

        # 2. Draw & Plot
        runner.draw_circle([CENTER_X, CENTER_Y, DRAW_Z], RADIUS, 10.0)

        # 3. Retreat
        runner.move_to_js([start_x, start_y, DRAW_Z + 0.1], [0.0, -89.9, 0.0], 2.0, "Retreat")
        runner.move_to_js([0.4, 0.0, 0.6], [0.0, -89.9, 0.0], 3.0, "Home")
        
        print("\nPerfect Circle Trajectory Complete!")
        
        runner.plot_graphs()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Stopping Simulation...")
        runner.sim.stopSimulation()

if __name__ == "__main__":
    main()