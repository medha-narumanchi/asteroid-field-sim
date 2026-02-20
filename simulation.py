import warp as wp
import numpy as np
from pxr import Usd, UsdGeom, Gf, UsdShade, Sdf

# Constants
NUM_ASTEROIDS = 800
NUM_STEPS = 200
DT = 0.01
GM = 100.0  # gravitational constant * planet mass

# Initialize warp
wp.init()

# Random starting positions in a ring/cloud around the origin
rng = np.random.default_rng(42)
asteroid_sizes = rng.uniform(0.5, 2.0, NUM_ASTEROIDS)
angles = rng.uniform(0, 2 * np.pi, NUM_ASTEROIDS)
radii = rng.uniform(15, 40, NUM_ASTEROIDS)

pos_np = np.zeros((NUM_ASTEROIDS, 3), dtype=np.float32)
pos_np[:, 0] = radii * np.cos(angles)
pos_np[:, 1] = rng.uniform(-2, 2, NUM_ASTEROIDS)
pos_np[:, 2] = radii * np.sin(angles)

# Give each asteroid an initial orbital velocity (perpendicular to radius)
vel_np = np.zeros((NUM_ASTEROIDS, 3), dtype=np.float32)
orbital_speed = np.sqrt(GM / radii) * 0.8
vel_np[:, 0] = -orbital_speed * np.sin(angles)
vel_np[:, 2] =  orbital_speed * np.cos(angles)

# Move to warp arrays
positions = wp.array(pos_np, dtype=wp.vec3)
velocities = wp.array(vel_np, dtype=wp.vec3)

print("Setup complete. Starting simulation...")

# Warp physics kernel - runs in parallel for every asteroid
@wp.kernel
def gravity_step(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    gm: float,
    dt: float
):
    i = wp.tid()  # which asteroid am I?

    pos = positions[i]
    vel = velocities[i]

    # Vector from asteroid to planet (at origin)
    direction = -pos
    dist = wp.length(direction)

    # Avoid division by zero
    if dist < 0.001:
        return

    # F = GM/r^2, in the direction of the planet
    acc = wp.normalize(direction) * (gm / (dist * dist))

    # Update velocity and position
    velocities[i] = vel + acc * dt
    positions[i] = pos + velocities[i] * dt

# Run the simulation
all_positions = []
all_velocities = []

for step in range(NUM_STEPS):
    wp.launch(kernel=gravity_step, dim=NUM_ASTEROIDS, inputs=[positions, velocities, GM, DT])
    if step % 5 == 0:
        all_positions.append(positions.numpy().copy())
        all_velocities.append(velocities.numpy().copy())
        print(f"Step {step}/{NUM_STEPS} done")

print("Simulation complete. Exporting to USD...")

# Calculate final speeds for color mapping
final_velocities = velocities.numpy()
speeds = np.linalg.norm(final_velocities, axis=1)
min_speed = speeds.min()
max_speed = speeds.max()

def speed_to_color(speed):
    # Slow = blue, fast = red
    t = (speed - min_speed) / (max_speed - min_speed)
    return Gf.Vec3f(float(t), 0.2, float(1.0 - t))

def make_color_material(stage, path, color):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material

np.savez("simulation_data.npz",
    positions=np.array(all_positions),
    velocities=np.array(all_velocities),
    final_velocities=velocities.numpy(),
    sizes=asteroid_sizes
)

# Simulation stats
distances_final = np.linalg.norm(all_positions[-1], axis=1)
print("\n--- Asteroid Field Stats ---")
print(f"Asteroids simulated:     {NUM_ASTEROIDS}")
print(f"Orbital radius range:    {distances_final.min():.2f} to {distances_final.max():.2f} units")
print(f"Average speed:           {speeds.mean():.3f} units/s")
print(f"Fastest asteroid speed:  {speeds.max():.3f} units/s")
print(f"Slowest asteroid speed:  {speeds.min():.3f} units/s")
print(f"Speed std deviation:     {speeds.std():.3f}")
print("----------------------------\n")



# Create USD file
stage = Usd.Stage.CreateNew("asteroid_field.usda")
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

# Add the planet at the origin
planet = UsdGeom.Sphere.Define(stage, "/Planet")
planet.GetRadiusAttr().Set(2.0)

# Create all asteroids with individual sizes and colors
asteroids = []

for i in range(NUM_ASTEROIDS):
    asteroid_path = f"/Asteroids/asteroid_{i}"
    asteroid = UsdGeom.Sphere.Define(stage, asteroid_path)
    asteroid.GetRadiusAttr().Set(1.0)
    xformable = UsdGeom.Xformable(asteroid)
    scale = float(asteroid_sizes[i])
    xformable.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    material = make_color_material(stage, f"/Materials/asteroid_mat_{i}", speed_to_color(speeds[i]))
    UsdShade.MaterialBindingAPI(asteroid).Bind(material)
    translate_op = xformable.AddTranslateOp()
    asteroids.append(translate_op)

# Set positions per frame
for frame, positions_at_step in enumerate(all_positions):
    for i, pos in enumerate(positions_at_step):
        asteroids[i].Set(
            Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])),
            frame
        )

stage.SetStartTimeCode(0)
stage.SetEndTimeCode(len(all_positions) - 1)
stage.Save()

print("Saved to asteroid_field.usda")