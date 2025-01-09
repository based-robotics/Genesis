import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

# Load the model directly for MuJoCo simulation
model = mj.MjModel.from_xml_path("four_bar_linkage.xml")
data = mj.MjData(model)

# Reset to initial position
mj.mj_resetDataKeyframe(model, data, 0)  # Reset to keyframe 0 (initial pose)

# Simulation parameters
duration = 10.0  # seconds
timestep = model.opt.timestep
num_steps = int(duration / timestep)

# Lists to store data for plotting
times = []
joint_angles = []
applied_torques = []

# Ramp up time for torques
ramp_duration = 2.0  # seconds
ramp_steps = int(ramp_duration / timestep)

# Simulation loop
for i in range(num_steps):
    time = i * timestep
    
    # Gradually ramp up the torques
    ramp = min(time / ramp_duration, 1.0) if time < ramp_duration else 1.0
    
    # Apply sinusoidal torques to both base joints
    torque1 = ramp * 50.0 * np.sin(2 * np.pi * time)
    torque2 = ramp * 25.0 * np.sin(2 * np.pi * time + np.pi/2)
    
    data.ctrl[0] = torque1
    data.ctrl[1] = torque2
    
    # Step the simulation
    mj.mj_step(model, data)
    
    # Store data for plotting
    times.append(time)
    joint_angles.append([data.qpos[0], data.qpos[2]])  # Store both base joint angles
    applied_torques.append([torque1, torque2])

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
joint_angles = np.array(joint_angles)
plt.plot(times, joint_angles[:, 0], label='Joint 1')
plt.plot(times, joint_angles[:, 1], label='Joint 3')
plt.title('Joint Angles vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
applied_torques = np.array(applied_torques)
plt.plot(times, applied_torques[:, 0], label='Torque 1')
plt.plot(times, applied_torques[:, 1], label='Torque 2')
plt.title('Applied Torques vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N⋅m)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
