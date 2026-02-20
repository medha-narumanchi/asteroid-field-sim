import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter



# Load simulation data
data = np.load("simulation_data.npz")
all_positions = data["positions"]      # shape: (frames, 800, 3)
final_velocities = data["final_velocities"]
sizes = data["sizes"]
GM = 100.0
distances_final = np.linalg.norm(all_positions[-1], axis=1)
NUM_ASTEROIDS = 800

# Calculate speeds for color mapping
speeds = np.linalg.norm(final_velocities, axis=1)
min_speed = speeds.min()
max_speed = speeds.max()
normalized_speeds = (speeds - min_speed) / (max_speed - min_speed)

# --- Plot 1: Asteroid field (final frame) ---
fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

pos = all_positions[-1]
ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1],
    c=normalized_speeds, cmap='coolwarm',
    s=sizes * (0.5 + normalized_speeds) * 15, alpha=0.8)
ax.scatter([0], [0], [0], c='white', s=200)
ax.set_title("Asteroid Field Simulation", color='white')
plt.tight_layout()
plt.savefig("asteroid_field.png", dpi=150, facecolor='black')
print("Saved asteroid_field.png")

# --- Plot 2: Energy conservation ---
# We need to recompute velocities at each frame
# Approximate KE and PE from position data
kinetic = []
potential = []
total = []

all_velocities = data["velocities"]

for frame_pos, frame_vel in zip(all_positions, all_velocities):
    distances = np.linalg.norm(frame_pos, axis=1)
    speeds_frame = np.linalg.norm(frame_vel, axis=1)
    ke = 0.5 * np.sum(speeds_frame ** 2)
    pe = np.sum(-GM / distances)
    kinetic.append(ke)
    potential.append(pe)
    total.append(ke + pe)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(kinetic, color='red', label='Kinetic Energy')
ax2.plot(potential, color='blue', label='Potential Energy')
ax2.plot(total, color='green', linestyle='--', label='Total Energy')
ax2.set_xlabel("Simulation Frame")
ax2.set_ylabel("Energy")
ax2.set_title("Energy Conservation Over Time")
ax2.legend()
ax2.grid(True)
energy_error = abs(total[-1] - total[0])
ax2.annotate(f"Energy conservation error: {energy_error:.4f}", 
    xy=(0.02, 0.05), xycoords='axes fraction',
    color='white', fontsize=10,
    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
ax2.set_facecolor('#111111')
fig2.patch.set_facecolor('#111111')
ax2.tick_params(colors='white')
ax2.xaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')
ax2.title.set_color('white')
ax2.legend(facecolor='#222222', labelcolor='white')
# Add stats text to the plot
ax.text2D(0.02, 0.98, 
    f"Asteroids: {NUM_ASTEROIDS}\nAvg speed: {speeds.mean():.3f}\nMax speed: {speeds.max():.3f}\nMin speed: {speeds.min():.3f}\nRadius: {distances_final.min():.1f} - {distances_final.max():.1f}",
    transform=ax.transAxes,
    color='white', fontsize=8,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
ax2.set_title(f"Energy Conservation Over Time\nAsteroids: {NUM_ASTEROIDS}  |  Avg speed: {speeds.mean():.3f}  |  Max: {speeds.max():.3f}  |  Min: {speeds.min():.3f}  |  Radius: {distances_final.min():.1f}–{distances_final.max():.1f}", color='white', fontsize=9)
plt.tight_layout()
plt.savefig("energy_conservation.png", dpi=150)
print("Saved energy_conservation.png")


# --- Animation ---
print("Creating animation...")

fig3 = plt.figure(figsize=(8, 8), facecolor='black')
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_facecolor('#222222')

def update(frame):
    ax3.cla()
    ax3.set_facecolor('#222222')
    pos = all_positions[frame]
    frame_speeds = np.linalg.norm(all_velocities[frame], axis=1)
    norm_speeds = (frame_speeds - frame_speeds.min()) / (frame_speeds.max() - frame_speeds.min() + 1e-8)
    ax3.scatter(pos[:, 0], pos[:, 2], pos[:, 1],
        c=norm_speeds, cmap='coolwarm',
        s=sizes * (0.5 + norm_speeds) * 15,
        alpha=0.8)
    ax3.scatter([0], [0], [0], c='white', s=200)
    ax3.set_title(f"Asteroid Field — Frame {frame+1}/{len(all_positions)}", color='white')
    ax3.tick_params(colors='white')

ani = FuncAnimation(fig3, update, frames=len(all_positions), interval=150)
ani.save("asteroid_animation.gif", writer=PillowWriter(fps=8))
print("Saved asteroid_animation.gif")